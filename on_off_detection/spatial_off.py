import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import on_off
from .methods.exceptions import ALL_METHOD_EXCEPTIONS

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
    window_i = window_row.name
    if verbose:
        print(f'Run window #{window_i+1}/N, window={window_row["window_depths"]}')

    on_off_model = on_off.OnOffModel(
        window_trains_list,
        bouts_df,
        cluster_ids=window_cluster_ids,
        method=on_off_method,
        params=on_off_params,
        verbose=verbose,
    )
    try:
        window_on_off_df = on_off_model.run()
        window_on_off_df["raised_exception"] = False
        window_on_off_df["exception"] = None
    except ALL_METHOD_EXCEPTIONS as e:
        print(
            f"Caught the following exception for window={window_row['window_depths']}"
        )
        print(e)
        window_on_off_df = pd.DataFrame(
            {
                "raised_exception": [True],
                "exception": [None],
                "state": [None],
            }
        )

    # Store window information
    window_on_off_df["window_idx"] = window_i
    window_on_off_df = window_on_off_df.assign(
        **{k: [v] * len(window_on_off_df) for k, v in window_row.items()}
    )

    return window_on_off_df


class SpatialOffModel(on_off.OnOffModel):
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
        super().__init__(
            trains_list,
            bouts_df,
            cluster_ids=cluster_ids,
            method=on_off_method,
            params=on_off_params,
            verbose=verbose,
        )
        #
        self._spatial_params = None
        self.spatial_params = spatial_params
        self.n_jobs=n_jobs
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
        self.windows_df = self.initialize_windows_df()
        # Output
        self.all_windows_on_off_df = None  # Pre-merging
        self.off_df = None  # Final, post-merging

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

    def get_cluster_firing_rates(self):
        total_duration = self.bouts_df.duration.sum()
        return np.array([
            len(train) / total_duration
            for train in self.trains_list
        ])

    def initialize_windows_df(self):

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
        all_window_depths = []
        all_window_sizes = []
        all_window_cluster_indices = []
        all_window_cluster_ids = []
        all_window_sumFR = []

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
            window_cluster_indices = np.where(
                np.logical_and(
                    lo <= self.cluster_depths, self.cluster_depths <= hi
                )
            )[0].astype(int)

            # Save information for this window
            all_window_sizes.append(hi - lo)
            all_window_depths.append((lo, hi))
            all_window_cluster_indices.append(window_cluster_indices)
            all_window_cluster_ids.append(self.cluster_ids[window_cluster_indices])
            window_sumFR = np.sum(self.cluster_firing_rates[window_cluster_indices])
            all_window_sumFR.append(window_sumFR)


            # Next window defined from overlap in cumulative firing rate
            lo_cumFR = cumFR[lo_idx]
            hi_cumFR = cumFR[min(hi_idx, len(cumFR) - 1)]
            lo_idx = np.where(cumFR > (hi_cumFR - lo_cumFR) * (1 - window_fr_overlap) + lo_cumFR)[0][0]


        return pd.DataFrame(
            {
                "window_size": all_window_sizes,
                "window_depths": all_window_depths,
                "window_sumFR": all_window_sumFR,
                "window_cluster_indices": all_window_cluster_indices,
                "window_cluster_ids": all_window_cluster_ids,
            }
        ).reset_index()

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

        if self.n_jobs == 1:
            on_off_dfs = []
            for _, window_row in tqdm(self.windows_df.iterrows()):
                window_on_off_df = _run_detection(
                    window_row,
                    self.get_window_trains(window_row),
                    self.get_window_cluster_ids(window_row),
                    self.bouts_df,
                    self.method,
                    self.params,
                    self.verbose,
                )
                on_off_dfs.append(window_on_off_df)
        else:
            from joblib import Parallel, delayed

            on_off_dfs = Parallel(n_jobs=self.n_jobs, backend="multiprocessing")(
                delayed(_run_detection)(
                    window_row,
                    self.get_window_trains(window_row),
                    self.get_window_cluster_ids(window_row),
                    self.bouts_df,
                    self.method,
                    self.params,
                    self.verbose,
                )
                for _, window_row in self.windows_df.iterrows()
            )

        dfs_to_concat = [df for df in on_off_dfs if df is not None]
        if len(dfs_to_concat):
            self.all_windows_on_off_df = pd.concat(dfs_to_concat).reset_index(drop=True)
        else:
            self.all_windows_on_off_df = pd.DataFrame()
        print(f"Done getting all windows on off periods.", end=" ")
        print(
            f"Found N={len(self.all_windows_on_off_df)} ON and OFF periods across windows."
        )

        return self.all_windows_on_off_df

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
                - `depth_min`/`depth_max`/`depth_span`:
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
        off_df[["depth_min", "depth_max"]] = pd.DataFrame(
            off_df["window_depths"].to_list(), index=off_df.index
        )
        off_df["depth_span"] = off_df["depth_max"] - off_df["depth_min"]
        # Keep only meaningful columns
        off_df = off_df.loc[
            :,
            [
                "state", "start_time_shared", "end_time_shared", "duration_shared",
                "start_time_earliest", "end_time_latest", "duration_longest",
                "depth_min", "depth_max", "depth_span",
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
    depth_mins, depth_maxs = selected_off_df["depth_min"].values, selected_off_df["depth_max"].values
    new_depth_min = min(*list(depth_mins), merged_off_row["depth_min"])
    new_depth_max = max(*list(depth_maxs), merged_off_row["depth_max"])

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
    merged_off_row["depth_min"] = new_depth_min
    merged_off_row["depth_max"] = new_depth_max
    merged_off_row["depth_span"] = new_depth_max - new_depth_min
    # Origins
    merged_off_row["N_merged_window_offs"] += len(selected_off_df)
    merged_off_row["merged_window_offs_indices"] += list(selected_off_df.index)

    return merged_off_row


def _find_contiguous_off_indices(merged_off_row, nearby_off_df, spatial_params):
    merged_depth_min, merged_depth_max = merged_off_row["depth_min"], merged_off_row["depth_max"]
    min_depth_overlap = spatial_params["min_depth_overlap"]
    depth_overlap = nearby_off_df.apply(
        lambda row: (min(row["depth_max"], merged_depth_max) - max(row["depth_min"], merged_depth_min)),
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