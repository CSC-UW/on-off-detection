import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy

from . import on_off
from .methods.exceptions import ALL_METHOD_EXCEPTIONS
from .utils import subset_sorted_train, kway_mergesort, slice_and_concat_sorted_train

SPATIAL_PARAMS = {
    # Windowing/pooling
	'window_min_size': 150,  # (um) Smallest allowed window size
	'window_min_fr': 100,  # (Hz) Smallest allowed within window population rate
	'window_fr_overlap': 0.75,  # (no unit) Population firing rate overlap between successive window
    # Merging of OFF states across windows
    "min_window_off_duration": 0.03,  # Remove OFFs shorter than this before merging
    "nearby_off_max_time_diff": 3,  # (sec)
    "min_shared_duration_overlap": 0.02, # (sec)
    "min_depth_overlap": 50, # (um)
}


def _run_detection(
    window_row,
    window_trains_list,
    window_cluster_ids,
    bouts_df,
    on_off_method,
    on_off_params,
    verbose=False,
):
    window_id = window_row.name
    if verbose:
        print(f'Run window {window_id}, window={window_row["window_lo"]}-{window_row["window_hi"]}')

    on_off_model = on_off.OnOffModel(
        window_trains_list,
        bouts_df,
        cluster_ids=window_cluster_ids,
        method=on_off_method,
        params=on_off_params,
        verbose=verbose,
    )
    try:
        window_on_off_df, window_output_info = on_off_model.run()
        window_output_info["raised_exception"] = False
        window_output_info["exception"] = None
    except ALL_METHOD_EXCEPTIONS as e:
        print(
            f"Caught the following exception for window={window_row['window_lo']}-{window_row['window_hi']}"
        )
        print(e)
        window_on_off_df = pd.DataFrame()
        window_output_info = {
            "raised_exception": True,
            "exception": None,
            "state": e,
        }

    # Store window information
    window_on_off_df["window_id"] = window_id
    window_on_off_df["lo"] = window_row["window_lo"]
    window_on_off_df["hi"] = window_row["window_hi"]
    window_on_off_df["span"] = window_row["window_span"]
    window_output_info["window_id"] = window_id
    window_output_info["lo"] = window_row["window_lo"]
    window_output_info["hi"] = window_row["window_hi"]
    window_output_info["span"] = window_row["window_span"]

    return window_on_off_df, window_output_info


class SpatialOffModel():
    """Run spatial OFF-state detection from MUA data.

    SpatialOffModel.run() runs on/off detection using OnOffModel at different
    spatial grains, and then merges all the detected off states between and across
    grains.

    Args:
            trains_list (list of array-like): Sorted MUA spike times for each cluster
            cluster_depths (array-like): Depth of each cluster in um
            Tmax: End time of recording.

    Kwargs:
            cluster_ids (list of array-like): Cluster ids. Added to output df if
                    provided. (default None)
            cluster_firing_rates (list of array-like): Firing rates for each cluster. Used to exclude
                    window based on threshold (default None)
            on_off_method: Name of method used for On/Off detection within each
                    spatial window (default hmmem)
            on_off_params: Dict of parameters passed to the on/off detection
                    algorithm within each spatial window. Mandatory params depend of
                    <method>. (default None)
            spatial_params: Dict of parameters describing spatial pooling and aggregation
                    of off periods (default None)
            bouts_df (pd.DataFrame): Frame containing bouts of interest. Must contain
                    'start_time', 'end_time', 'duration' and 'state' columns. If
                    provided, we consider only spikes within these bouts for on-off
                    detection (by cutting-and-concatenating the trains of each cluster).
                    The "state", "start_time" and "end_time" of the bout each on or off
                    period pertains to is saved in the "bout_state", "bout_start_time"
                    and "bout_end_time" columns. ON or OFF periods that are not STRICTLY
                    comprised within bouts are dismissed (default None)
            n_jobs (int or None): Number of parallel jobs.
                    No parallelization if None or 1
            verbose (bool): Default True
    """

    def __init__(
        self,
        trains_list,
        cluster_depths,
        bouts_df,
        cluster_ids=None,
        on_off_method="hmmem",
        on_off_params=None,
        spatial_params=None,
        n_jobs=1,
        verbose=True,
    ):
        # Main. Full checks repeated when initializing/runing the OnOffModel objects per window.
        # We don't simply inherit because the parameters/windows may be changed later on
        # Could be refactored..
        self.trains_list = [
            subset_sorted_train(bouts_df, np.sort(train)) for train in trains_list
        ]
        if cluster_ids is not None:
            assert len(cluster_ids) == len(trains_list)
            self.cluster_ids = np.array(cluster_ids)
        else:
            self.cluster_ids = np.array(["" for i in range(len(trains_list))])
        assert all(
            [
                c in bouts_df.columns
                for c in ["start_time", "end_time", "duration", "state"]
            ]
        )
        assert bouts_df.duration.sum(), f"Empty bouts"
        self.bouts_df = bouts_df
        self.method = on_off_method
        # Params
        self._per_window_on_off_params = None
        self.shared_on_off_params = on_off_params
        #
        self.verbose=verbose
        self.n_jobs=n_jobs
        #
        self._spatial_params = None
        self.spatial_params = spatial_params
        # Depths
        assert len(cluster_depths) == len(self.cluster_ids)
        self.cluster_depths = np.array(cluster_depths)
        # Spatial pooling info
        self.cluster_firing_rates = self.get_cluster_firing_rates()
        sumFR = np.sum(self.cluster_firing_rates)
        if  sumFR < self.spatial_params["window_min_fr"]:
            raise ValueError(
                "Cumulative firing rate = {sumFR}Hz, too low for requested `windows_min_fr` param"
            )
        self.initialize_default_windows_df()
        # Output
        self.all_windows_on_off_df = None  # Pre-merging
        self.off_df = None  # Final, post-merging
        self.window_output_infos = None  # {<window_id>: <window_output_info>}

    def get_cluster_firing_rates(self):
        total_duration = self.bouts_df.duration.sum()
        return np.array([
            len(train) / total_duration
            for train in self.trains_list
        ])

    @property
    def spatial_params(self):
        if self._spatial_params is None:
            raise ValueError("Spatial params were not assigned")
        return self._spatial_params

    @spatial_params.setter
    def spatial_params(self, params):
        unrecognized_params = set(params.keys()) - set(SPATIAL_PARAMS.keys())
        if len(unrecognized_params):
            raise ValueError(
                f"Unrecognized parameter keys for spatial algorithm: "
                f"{unrecognized_params}.\n\n"
                f"Default (recognized) parameters for spatial algo: {SPATIAL_PARAMS}"
            )
        missing_params = set(SPATIAL_PARAMS.keys()) - set(params.keys())
        if len(missing_params):
            print(
                f"Setting self.spatial_params: Use default value for params: {missing_params}"
            )
        self._spatial_params = {k: v for k, v in SPATIAL_PARAMS.items()}
        self._spatial_params.update(params)
        print(f"Spatial params: {self._spatial_params}")

    @property
    def per_window_on_off_params(self):
        if self._per_window_on_off_params is None:
            return [
                deepcopy(self.shared_on_off_params)
                for _ in self.windows_df.itertuples()
            ]
        if not len(self._per_window_on_off_params) == len(self.windows_df):
            raise ValueError(
                """Number of parameter dictionaries doesn't match number of windows."""
            )
        return self._per_window_on_off_params

    @per_window_on_off_params.setter
    def per_window_on_off_params(self, per_window_on_off_params):
        if not len(per_window_on_off_params) == len(self.windows_df):
            raise ValueError(
                """Number of parameter dictionaries doesn't match number of windows."""
            )
        print(f"Setting custom per-window parameters for N={len(per_window_on_off_params)} windows")
        self._per_window_on_off_params = deepcopy(per_window_on_off_params)

    @property
    def windows_df(self):
        return self._windows_df

    @windows_df.setter
    def windows_df(self, windows_df):
        assert all([c in windows_df.columns for c in ["window_lo", "window_hi", "window_span"]])
        windows_df = windows_df.copy()

        windows_df["window_cluster_indices"] = windows_df.apply(
            lambda row: np.where(
                    np.logical_and(
                        row.window_lo <= self.cluster_depths, self.cluster_depths <= row.window_hi
                    )
            )[0].astype(int),
            axis=1,
        )
        windows_df["window_cluster_ids"] = windows_df.apply(
            lambda row: self.cluster_ids[row.window_cluster_indices],
            axis=1
        )
        windows_df["window_sumFR"] = windows_df.apply(
            lambda row: np.sum(self.cluster_firing_rates[row.window_cluster_indices]),
            axis=1
        )

        self._windows_df = windows_df

    def initialize_default_windows_df(self):

        SPATIAL_RES = 20

        window_min_fr = self.spatial_params["window_min_fr"]
        window_min_size = self.spatial_params["window_min_size"]
        window_fr_overlap = self.spatial_params["window_fr_overlap"]

        # Get cumulative firing rate 
        depths = self.cluster_depths
        bins = np.arange(depths.min(), depths.max() + SPATIAL_RES, SPATIAL_RES) # Endpoint inclusive
        depthFR, bins = np.histogram(depths, weights=self.cluster_firing_rates, bins=bins)
        cumFR = np.cumsum(depthFR)

        lo_idx = 0
        hi_idx = 0
        all_window_los = []
        all_window_his = []
        all_window_spans = []

        # Iterate until reaching max depth
        while hi_idx < len(bins) - 1:

            # All candidate bin indices matching size and fr requirement
            idx_above = np.where(
                ((cumFR - cumFR[lo_idx]) > window_min_fr) 
                & ((bins[:-1] - bins[lo_idx]) > window_min_size)
            )[0]
            if not len(idx_above):
                # We reached the end
                hi_idx = len(bins) - 1
                # Ensure last window as well has the minimal required size and FR
                lo_idx = np.where(
                    ((cumFR - cumFR[-1]) < - window_min_fr)
                    & ((bins[:-1] - bins[hi_idx]) < - window_min_size)
                    # & 
                )[0][-1]
            else:
                hi_idx = idx_above[0]

            lo, hi = bins[lo_idx], bins[hi_idx]

            # Save information for this window
            all_window_spans.append(hi - lo)
            all_window_los.append(lo)
            all_window_his.append(hi)

            # Next window defined from overlap in cumulative firing rate
            lo_cumFR = cumFR[lo_idx]
            hi_cumFR = cumFR[min(hi_idx, len(cumFR) - 1)]
            lo_idx = np.where(cumFR > (hi_cumFR - lo_cumFR) * (1 - window_fr_overlap) + lo_cumFR)[0][0]

        self.windows_df = pd.DataFrame(
            {
                "window_span": all_window_spans,
                "window_lo": all_window_los,
                "window_hi": all_window_his,
            }
        ).reset_index(drop=True)

    def get_window_trains(self, window_row):
        """Return window_trains_list for a row of `self.windows_df`."""
        assert "window_cluster_indices" in window_row
        return [
            self.trains_list[cluster_idx]
            for cluster_idx in window_row["window_cluster_indices"]
        ]

    def get_window_cluster_ids(self, window_row):
        """Return window_cluster_ids for a row of `self.windows_df`."""
        assert "window_cluster_indices" in window_row
        ids = window_row["window_cluster_ids"]
        assert np.all(ids == self.cluster_ids[window_row["window_cluster_indices"]])
        return ids

    def dump(self, filepath):
        assert not Path(filepath).exists()
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def run(self):
        self.run_all_windows_on_off_df()
        self.run_off_df()
        return self.off_df

    def run_all_windows_on_off_df(self):
        print(
            f"Run on-off detection for each spatial window (N={len(self.windows_df)})"
        )

        per_window_on_off_params = self.per_window_on_off_params

        if self.n_jobs == 1:
            on_off_dfs = []
            window_output_infos = {}
            for i, (window_id, window_row) in tqdm(enumerate(self.windows_df.iterrows())):
                window_on_off_df, window_output_info = _run_detection(
                    window_row,
                    self.get_window_trains(window_row),
                    self.get_window_cluster_ids(window_row),
                    self.bouts_df,
                    self.method,
                    per_window_on_off_params[i],
                    self.verbose,
                )
                on_off_dfs.append(window_on_off_df)
                window_output_infos[window_id] = window_output_info
        else:
            from joblib import Parallel, delayed

            on_off_dfs, output_infos = zip(
                *Parallel(n_jobs=self.n_jobs, backend="multiprocessing")(
                    delayed(_run_detection)(
                        window_row,
                        self.get_window_trains(window_row),
                        self.get_window_cluster_ids(window_row),
                        self.bouts_df,
                        self.method,
                        per_window_on_off_params[i],
                        self.verbose,
                    )
                    for i, (window_id, window_row) in enumerate(self.windows_df.iterrows())
                )
            )
            window_output_infos = {
                window_id: output_info for (window_id, _), output_info in zip(
                    self.windows_df.iterrows(),
                    output_infos
                )
            }

        dfs_to_concat = [df for df in on_off_dfs if df is not None]
        if len(dfs_to_concat):
            self.all_windows_on_off_df = pd.concat(dfs_to_concat).reset_index(drop=True)
        else:
            self.all_windows_on_off_df = pd.DataFrame()
        self.window_output_infos = window_output_infos
        print(f"Done getting all windows on off periods.", end=" ")
        print(
            f"Found N={len(self.all_windows_on_off_df)} ON and OFF periods across windows."
        )

        return self.all_windows_on_off_df, self.window_output_infos

    def run_off_df(self):
        all_windows_on_off_df = self.all_windows_on_off_df
        if not len(all_windows_on_off_df[all_windows_on_off_df["state"] == "off"]):
            print("No OFF states to merge")
            self.off_df = pd.DataFrame()
        else:
            print("Merge off periods across windows.")
            off_df = self._merge_all_windows_offs(
                self.all_windows_on_off_df, self.spatial_params
            )
            print(f"Found N={len(off_df)} off periods after merging")
            self.off_df = off_df

        return self.off_df

    @classmethod
    def _merge_all_windows_offs(cls, all_windows_on_off_df, spatial_params):
        """Merge detected off states within and across spatial grains.

        The algorithm goes as follows:
        - Remove all ON periods (work only on OFFs)
        - Remove OFF periods shorter than `min_window_off_duration`
        - Sort OFFs by descending duration
        - For each OFF period in the initial df:
            - TODO


        Each OFF period in the final df have both:
                - `start_time_shared`/`end_time_shared`/`duration_shared` field: 
                    refer to the intersection of all merged OFF periods
                - `start_time_earliest`/`end_time_latest`/`duration_longest`:
                    Earliest/latest amongst merged offs.
                - `lo`/`hi`/`span`:
                    Span across merged offs
        """
        assert len(all_windows_on_off_df.index.unique()) == len(all_windows_on_off_df)

        # Remove ON periods and short offs
        all_windows_off_df = all_windows_on_off_df[
            all_windows_on_off_df["state"] == "off"
        ].copy()
        if spatial_params["min_window_off_duration"] is not None:
            all_windows_off_df = all_windows_off_df[
                all_windows_off_df["duration"] >= spatial_params["min_window_off_duration"]
            ]

        if not len(all_windows_off_df):
            return pd.DataFrame()

        # Sort by duration and break ties with start time
        off_df = all_windows_off_df.sort_values(
            by=["duration", "start_time"],
            ascending=False,
        )  # Don't reset index here so we keep same indices as all_windows_off_df

        # Initialize cols for extended off duration etc
        off_df["start_time_shared"] = off_df["start_time"]
        off_df["end_time_shared"] = off_df["end_time"]
        off_df["duration_shared"] = off_df["duration"]
        off_df["start_time_earliest"] = off_df["start_time"]
        off_df["end_time_latest"] = off_df["end_time"]
        off_df["duration_longest"] = off_df["duration"]
        off_df["N_merged_window_offs"] = 1
        off_df["merged_window_offs_indices"] = [[idx] for idx in off_df.index]
        # Keep only meaningful columns
        off_df = off_df.loc[
            :,
            [
                "state", "start_time_shared", "end_time_shared", "duration_shared",
                "start_time_earliest", "end_time_latest", "duration_longest",
                "lo", "hi", "span",
                "N_merged_window_offs", "merged_window_offs_indices",
            ]
        ]

        merged_off_rows_list = []  # Concatenate these rows to create final off df
        # Iterate on this and memorize whether an index is to be kept considered 
        # for merging in future iterations.
        initial_off_df = off_df.copy()
        # Bad practice to modify during iteration so we make this a separate series
        # rather than a column
        keep = pd.Series(True, index=initial_off_df.index)

        # Iterate on rows that were never merged so far
        for i, off_row in tqdm(list(initial_off_df.iterrows())):

            if not keep.loc[i]:  # Don't modify row directly
                continue
            keep.loc[i] = False  # Don't merge row with itself

            # Subselect only nearby offs for speed
            nearby_off_max_time_diff = spatial_params[
                "nearby_off_max_time_diff"
            ]  # Search for candidates for merging only amongst nearby Offs
            nearby_off_df = initial_off_df.loc[
                keep.values
                & (
                    initial_off_df["start_time_shared"]
                    >= off_row["start_time_shared"] - nearby_off_max_time_diff
                )
                & (
                    initial_off_df["end_time_shared"]
                    <= off_row["end_time_shared"] + nearby_off_max_time_diff
                )
            ].copy()
            nearby_off_df["keep"] = True

            # Merge rows until there's no remaining candidate
            # We merge rows one by one to ensure proper temporal overlap
            off_indice_to_merge = _find_off_indice_to_merge(
                off_row,
                nearby_off_df,
                spatial_params,
            )
            merged_off_row = off_row.copy()
            while len(off_indice_to_merge):

                merged_off_row = _merge_off_rows(
                    merged_off_row, initial_off_df.loc[off_indice_to_merge]
                )

                # Remove/ignore merged row in the future
                keep.loc[off_indice_to_merge] = False
                nearby_off_df.loc[off_indice_to_merge, "keep"] = False

                # Find next off to merge
                off_indice_to_merge = _find_off_indice_to_merge(
                    merged_off_row, nearby_off_df, spatial_params
                )

            # Save merged off and remove/ignore current starting off period in the future
            merged_off_rows_list.append(merged_off_row)
            keep.loc[i] = False

        return pd.DataFrame(
            merged_off_rows_list
        ).sort_values(
            by="start_time_shared",
            ascending=True,
        ).reset_index(
            drop=True
        )


def _merge_off_rows(base_off_row, selected_off_df):
    merged_off_row = base_off_row.copy()

    # New min/max depths
    los, his = selected_off_df["lo"].values, selected_off_df["hi"].values
    new_lo = min(*list(los), merged_off_row["lo"])
    new_hi = max(*list(his), merged_off_row["hi"])

    # New start/end time (restrictive and extensive)
    start_times_shared = list(selected_off_df["start_time_shared"].values)
    start_times_earliest = list(selected_off_df["start_time_earliest"].values)
    end_times_shared = list(selected_off_df["end_time_shared"].values)
    end_times_latest = list(selected_off_df["end_time_latest"].values)
    new_start_time_shared = max(*start_times_shared, merged_off_row["start_time_shared"])
    new_start_time_earliest = min(*start_times_earliest, merged_off_row["start_time_earliest"])
    new_end_time_shared = min(*end_times_shared, merged_off_row["end_time_shared"])
    new_end_time_latest = max(*end_times_latest, merged_off_row["end_time_latest"])
    assert ( new_start_time_shared <= new_end_time_shared)
    assert ( new_start_time_shared >= merged_off_row["start_time_shared"])
    assert ( new_end_time_shared <= merged_off_row["end_time_shared"])
    assert ( new_start_time_shared >= new_start_time_earliest)
    assert ( new_end_time_shared <= new_end_time_latest)

    # Update fields
    # times
    merged_off_row["start_time_shared"] = new_start_time_shared
    merged_off_row["start_time_earliest"] = new_start_time_earliest
    merged_off_row["end_time_shared"] = new_end_time_shared
    merged_off_row["end_time_latest"] = new_end_time_latest
    merged_off_row["duration_shared"] = new_end_time_shared - new_start_time_shared
    merged_off_row["duration_longest"] = new_end_time_latest - new_start_time_earliest
    # Depths
    merged_off_row["lo"] = new_lo
    merged_off_row["hi"] = new_hi
    merged_off_row["span"] = new_hi - new_lo
    # Origins
    merged_off_row["N_merged_window_offs"] += len(selected_off_df)
    merged_off_row["merged_window_offs_indices"] += list(selected_off_df.index)

    return merged_off_row


def _find_contiguous_off_indices(merged_off_row, nearby_off_df, spatial_params):
    merged_lo, merged_hi = merged_off_row["lo"], merged_off_row["hi"]
    min_depth_overlap = spatial_params["min_depth_overlap"]
    depth_overlap = nearby_off_df.apply(
        lambda row: (min(row["hi"], merged_hi) - max(row["lo"], merged_lo)),
        axis=1,
    )
    return nearby_off_df.index[
        nearby_off_df["keep"]
        & (depth_overlap >= min_depth_overlap)
    ]


def _find_concurrent_off_indices(merged_off_row, nearby_off_df, spatial_params):
    """OFFs that overlap temporally with merged OFFs' shared start/end time """
    start_time_shared, end_time_shared = merged_off_row["start_time_shared"], merged_off_row["end_time_shared"]
    min_shared_duration_overlap = spatial_params["min_shared_duration_overlap"]
    shared_duration_overlap = nearby_off_df.apply(
        lambda row: (min(row["end_time_shared"], end_time_shared) - max(row["start_time_shared"], start_time_shared)),
        axis=1,
    )
    return nearby_off_df.index[
        nearby_off_df["keep"]
        & (shared_duration_overlap >= min_shared_duration_overlap)
    ]


def _find_off_indices_to_merge(merged_off_row, nearby_off_df, spatial_params):
    return np.intersect1d(
        _find_contiguous_off_indices(merged_off_row, nearby_off_df, spatial_params),
        _find_concurrent_off_indices(merged_off_row, nearby_off_df, spatial_params),
    )


def _find_off_indice_to_merge(merged_off_row, nearby_off_df, spatial_params):
    off_indices_to_merge = _find_off_indices_to_merge(
        merged_off_row, nearby_off_df, spatial_params
    )
    if not len(off_indices_to_merge):
        return []
    # Sort all candidate offs by longest temporal overlap with shared start/end time
    start_time_shared = merged_off_row["start_time_shared"]
    end_time_shared = merged_off_row["end_time_shared"]
    to_merge_df = nearby_off_df.loc[off_indices_to_merge]
    to_merge_df["shared_duration_overlap"] = to_merge_df.apply(
        lambda row: (min(row["end_time_shared"], end_time_shared) - max(row["start_time_shared"], start_time_shared)),
        axis=1
    )
    assert np.all(to_merge_df["shared_duration_overlap"] >= 0) # They should already be contiguous
    return to_merge_df.sort_values(by="shared_duration_overlap", ascending=False).index[
        0:1
    ]  # Return index of row with highest overlap