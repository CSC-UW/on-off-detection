import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import on_off
from .methods.exceptions import ALL_METHOD_EXCEPTIONS

SPATIAL_PARAMS = {
	# Windowing/pooling
	'window_size_min': 200,  # (um) Smallest spatial "grain" for pooling
	'window_overlap': 0.5,  # (no unit) Overlap between windows within each spatial grain 
	'window_size_step': 200,  # (um) Increase in size of windows across successive spatial "grains"
	# Merging of OFF state between and across grain
	'merge_max_time_diff': 0.050, # (s). To be merged, off states need their start & end times to differ by less than this
	'merge_min_off_overlap': 0.5, # (no unit). To be merged, off states need to overlap by more than `merge_min_off_overlap` times the shortest OFF duration
}

def _run_detection(
	window_row,
	window_trains_list,
	window_cluster_ids,
	Tmax,
	bouts_df,
	on_off_method,
	on_off_params,
	output_dir,
	verbose=False,
):
	window_i = window_row.name
	if verbose:
		print(f'Run window #{window_i+1}/N, window={window_row["window_depths"]}')
	
	on_off_model = on_off.OnOffModel(
		window_trains_list,
		Tmax,
		cluster_ids=window_cluster_ids,
		method=on_off_method,
		params=on_off_params,
		bouts_df=bouts_df,
		pooled_detection=True,
		output_dir=output_dir,
		debug_plot_filename=None,
		n_jobs=1,
		verbose=verbose,
	)
	try:
		window_on_off_df = on_off_model.run()
		window_on_off_df['raised_exception'] = False
		window_on_off_df['exception'] = None
	except ALL_METHOD_EXCEPTIONS as e:
		print(f"Caught the following exception for window={window_row['window_depths']}")
		print(e)
		window_on_off_df = pd.DataFrame({
			'raised_exception': [True],
			'exception': [None],
		})

	# Store window information
	window_on_off_df['window_idx'] = window_i
	window_on_off_df = window_on_off_df.assign(
		**{
			k: [v] * len(window_on_off_df)
			for k, v in window_row.items()
		}
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
		output_dir: Where we save output figures and summary statistics.
		verbose (bool): Default True
	"""

	def __init__(
		self, trains_list, cluster_depths, Tmax, cluster_ids=None,
		on_off_method='hmmem', on_off_params=None, spatial_params=None,
		catch_numerical_errors=True, bouts_df=None, n_jobs=10, output_dir=None,
		verbose=True
	):
		super().__init__(
			trains_list,
			Tmax,
			cluster_ids=cluster_ids,
			method=on_off_method,
			params=on_off_params,
			bouts_df=bouts_df,
			pooled_detection=True,
			output_dir=output_dir,
			debug_plot_filename=None,
			n_jobs=n_jobs,
		)
		# Spatial parameters
		unrecognized_params = set(spatial_params.keys()) - set(SPATIAL_PARAMS.keys())
		if len(unrecognized_params):
			raise ValueError(
				f"Unrecognized parameter keys for spatial algorithm: "
				f"{unrecognized_params}.\n\n"
				f"Default (recognized) parameters for spatial algo: {SPATIAL_PARAMS}"
			)
		self.spatial_params = {k: v for k, v in SPATIAL_PARAMS.items()}
		self.spatial_params.update(spatial_params)
		# Spatial pooling info
		self.cluster_depths = cluster_depths
		self.windows_df = self.initialize_windows_df()
		# Output
		self.all_windows_on_off_df = None  # Pre-merging
		self.off_df = None  # Final, post-merging
	
	def initialize_windows_df(self):
		p = self.spatial_params
		# Window sizes (ie spatial grain)
		min_depth, max_depth = min(self.cluster_depths), max(self.cluster_depths)
		max_window_size = max_depth - min_depth
		unique_window_sizes = np.arange(p['window_size_min'], max_window_size, p['window_size_step']).tolist()
		if max_window_size not in unique_window_sizes:
			unique_window_sizes.append(max_window_size)
		# Moving window for each window size, which clusters, which min/max depths
		# Across all spatial grains:
		all_window_sizes = []
		all_window_depths = []
		all_window_cluster_indices = []
		all_window_cluster_ids = []
		all_window_cluster_depths = []
		# For each spatial grain...
		for window_size in unique_window_sizes:
			# Sliding windows
			step = window_size * (1 - p['window_overlap'])
			if step <= 0:
				raise ValueError("Invalid value for `window_overlap` spatial parameter")
			depth_starts = np.arange(min_depth, max_depth - window_size, step).tolist()
			if max_depth - window_size not in depth_starts:
				depth_starts.append(max_depth - window_size)
			depth_intervals = [(d, d+window_size) for d in depth_starts]
			# Which clusters for each sliding window
			# Inclusive on both sides out of laziness
			cluster_indices = [
				np.where(np.logical_and(start <= self.cluster_depths, self.cluster_depths <= end))[0]
				for start, end in depth_intervals
			]
			cluster_ids = [
				self.cluster_ids[indices]
				for indices in cluster_indices
			]
			cluster_depths = [
				self.cluster_depths[indices]
				for indices in cluster_indices
			]
			all_window_sizes += [window_size for _ in range(len(depth_intervals))]
			all_window_depths += depth_intervals
			all_window_cluster_indices += cluster_indices
			all_window_cluster_ids += cluster_ids
			all_window_cluster_depths += cluster_depths
		return pd.DataFrame({
			'window_size': all_window_sizes,
			'window_depths': all_window_depths,
			'window_cluster_indices': all_window_cluster_indices,
			'window_cluster_ids': all_window_cluster_ids,
			'window_cluster_depths': all_window_cluster_depths,
			'window_n_clusters': [len(cs) for cs in all_window_cluster_ids]
		}).reset_index()

	def get_window_trains(self, window_row):
		"""Return window_trains_list for a row of `self.windows_df`."""
		assert "window_cluster_indices" in window_row
		return [self.trains_list[cluster_idx] for cluster_idx in window_row['window_cluster_indices']]

	def get_window_cluster_ids(self, window_row):
		"""Return window_cluster_ids for a row of `self.windows_df`."""
		assert "window_cluster_indices" in window_row
		ids = window_row['window_cluster_ids']
		assert np.all(ids == self.cluster_ids[window_row['window_cluster_indices']])
		return ids
	
	def dump(self, filepath):
		assert not Path(filepath).exists()
		with open(filepath, 'wb') as f:
			pickle.dump(self, f)

	@classmethod
	def load(cls, filepath):
		with open(filepath, 'rb') as f:
			return pickle.load(f)

	def run(self):
		self.run_all_windows_on_off_df()
		self.run_off_df()
		return self.off_df

	def run_all_windows_on_off_df(self):
		print(f"Run on-off detection for each spatial window (N={len(self.windows_df)})")

		if self.n_jobs == 1:
			on_off_dfs = []
			for _, window_row in tqdm(self.windows_df.iterrows()):
				window_on_off_df = _run_detection(
					window_row,
					self.get_window_trains(window_row),
					self.get_window_cluster_ids(window_row),
					self.Tmax,
					self.bouts_df,
					self.method,
					self.params,
					self.output_dir,
					self.verbose,
				)
				on_off_dfs.append(window_on_off_df)
		else:
			from joblib import Parallel, delayed
			on_off_dfs = Parallel(n_jobs=self.n_jobs, backend='multiprocessing')(
				delayed(_run_detection)(
					window_row,
					self.get_window_trains(window_row),
					self.get_window_cluster_ids(window_row),
					self.Tmax,
					self.bouts_df,
					self.method,
					self.params,
					self.output_dir,
					self.verbose,
				)
				for _, window_row in self.windows_df.iterrows()
			)

		self.all_windows_on_off_df = pd.concat([
			df for df in on_off_dfs if df is not None
		]).reset_index(drop=True)
		print(f"Done getting all windows on off periods.", end=" ")
		print(f"Found N={len(self.all_windows_on_off_df)} ON and OFF periods across windows.")

		return self.all_windows_on_off_df
	
	def run_off_df(self):
		print("Merge off periods across windows.")
		off_df = self._merge_all_windows_offs(
			self.all_windows_on_off_df,
			self.spatial_params
		)
		print(f"Found N={len(off_df)} off periods after merging")

		self.off_df = off_df.sort_values(
			by='start_time'
		).reset_index().rename(
			columns={'level_0': 'original_idx'}
		)
		return self.off_df


	@classmethod
	def _merge_all_windows_offs(cls, all_windows_on_off_df, spatial_params):
		"""Merge detected off states within and across spatial grains.
		
		The algorithm goes as follows:
		- Remove all ON periods (work only on OFFs)
		- Sort OFFs by start time
		- For each OFF period in the initial df
			- Select all OFFs candidate for merging (those whose start/end time are close enough)
			- Find OFFs to merge to initial off (those that are contiguous & synchronous)
			- While there are OFFs to merge:
				- Merge them
				- remove them from candidate offs
				- find OFFs to merge to newly merged OFF
			- Remove OFFs that are contiguus and concurrent without being synchronous
			- Remove all the merged offs from the initial array and save merged off


		Each OFF period in the final df have both:
			- `start_time`/`end_time`/`duration` field: latest/earliest start time and end
				time of all merged OFFs (shortest OFF duration across grains)
			- and a `start_time_2`/`end_time_2`/`duration_2` field:
				earliest/latest start and end time of merged OFFs (longest OFF
				duration across grains)
		"""
		assert len(all_windows_on_off_df.index.unique()) == len(all_windows_on_off_df)

		# Remove ON periods
		# Sort by grain, and then by window depth
		off_df = all_windows_on_off_df[
			all_windows_on_off_df['state'] == 'off'
		].copy().sort_values(
			by=['start_time', 'end_time'],
			ascending=True
		)  # Don't reset index here so we keep same indices as all_windows_on_off_df

		# Initialize cols for extended off duration etc
		off_df['start_time_2'] = off_df['start_time']
		off_df['end_time_2'] = off_df['end_time']
		off_df['duration_2'] = off_df['duration']
		off_df['N_merged_window_offs'] = 1
		off_df['merged_window_offs_indices'] = off_df.index

		# Initialize utility col (split window_depths tuple column)
		off_df[['depth_min', 'depth_max']] = pd.DataFrame(
			off_df['window_depths'].to_list(),
			index = off_df.index
		)

		merged_off_rows_list = []  # Concatenate these rows to create final off df
		initial_off_df = off_df.copy() 
		initial_off_df['keep'] = True  # Set to False when OFF was merged
		NEARBY_OFF_MAX_TIME_DIFF = 2  # Search for contiguous/concurrent off states only amongst nearby Offs

		# Iterate on rows that were never merged so far
		for i, off_row in tqdm(list(initial_off_df.iterrows())):
			if not initial_off_df.at[i, 'keep']:  # Don't modify row directly
				continue

			# Subselect only nearby offs for speed
			nearby_off_df = initial_off_df.loc[
				initial_off_df['keep']
				& initial_off_df['start_time'].between(
					off_row['start_time'] - NEARBY_OFF_MAX_TIME_DIFF,
					off_row['start_time'] + NEARBY_OFF_MAX_TIME_DIFF,
				)
				& initial_off_df['end_time'].between(
					off_row['end_time'] - NEARBY_OFF_MAX_TIME_DIFF,
					off_row['end_time'] + NEARBY_OFF_MAX_TIME_DIFF,
				)
			].copy()

			# Merge rows until there's no remaining off contiguous and concurrent to
			# the current merged off
			off_indices_to_merge = _find_off_indices_to_merge(
				off_row,
				nearby_off_df,
				spatial_params,
			)
			merged_off_row = off_row.copy()
			while len(off_indices_to_merge):

				merged_off_row = _merge_off_rows(
					merged_off_row,
					initial_off_df.loc[off_indices_to_merge]
				)

				# Remove/ignore merged indices in the future
				initial_off_df.loc[off_indices_to_merge, 'keep'] = False
				nearby_off_df.loc[off_indices_to_merge, 'keep'] = False

				# Remove/ignore OFFs that 
				# Find new candidates for merging
				off_indices_to_merge = _find_off_indices_to_merge(
					merged_off_row,
					nearby_off_df,
					spatial_params
				)
			# Save merged off and remove/ignore current starting off period in the future
			merged_off_rows_list.append(merged_off_row)
			initial_off_df.at[i, 'keep'] = False  # https://stackoverflow.com/questions/23330654/update-a-dataframe-in-pandas-while-iterating-row-by-row
		
		return pd.DataFrame(merged_off_rows_list)


def _merge_off_rows(base_off_row, selected_off_df):
	merged_off_row = base_off_row.copy()

	# New min/max depths and window size
	depth_mins, depth_maxs = zip(*selected_off_df['window_depths'].values)
	new_depth_min = min(*list(depth_mins), merged_off_row['depth_min'])
	new_depth_max = max(*list(depth_maxs), merged_off_row['depth_max'])

	# New start/end time (restrictive and extensive)
	start_times = list(selected_off_df['start_time'].values)
	end_times = list(selected_off_df['end_time'].values)
	new_start_time = max(*start_times, merged_off_row['start_time'])
	new_start_time_2 = min(*start_times, merged_off_row['start_time_2'])
	new_end_time = min(*end_times, merged_off_row['end_time'])
	new_end_time_2 = max(*end_times, merged_off_row['end_time_2'])
	# TODO
	# assert new_start_time < new_end_time # Modify logic for concurrent off indices otherwise

	# Update fields
	# times
	merged_off_row['start_time'] = new_start_time
	merged_off_row['start_time_2'] = new_start_time_2
	merged_off_row['end_time'] = new_end_time
	merged_off_row['end_time_2'] = new_end_time_2
	merged_off_row['duration'] = new_end_time - new_start_time
	merged_off_row['duration_2'] = new_end_time_2 - new_start_time_2
	# depths
	merged_off_row['depth_min'] = new_depth_min
	merged_off_row['depth_max'] = new_depth_max
	merged_off_row['window_depths'] = (new_depth_min, new_depth_max)  # TODO Maybe rename?
	merged_off_row['window_size'] = new_depth_max - new_depth_min  # TODO maybe rename?
	#
	merged_off_row['N_merged_window_offs'] += len(selected_off_df)
	# TODO: Cluster indices etc

	return merged_off_row


def _find_contiguous_off_indices(off_row, nearby_off_df, spatial_params):
	depth_min, depth_max = off_row['window_depths']
	return nearby_off_df.index[
		nearby_off_df['keep']
		& (
			(
				(nearby_off_df['depth_min'] <= depth_min)
				& (nearby_off_df['depth_max'] >= depth_min)
			) # Contiguous below
			| (
				(nearby_off_df['depth_min'] <= depth_max)
				& (nearby_off_df['depth_max'] >= depth_max)
			) # Contiguous above
			| (
				(nearby_off_df['depth_min'] >= depth_min)
				& (nearby_off_df['depth_max'] <= depth_max)
			) # fully within
		)
	]


def _find_concurrent_off_indices(off_row, nearby_off_df, spatial_params):
	start_time, end_time = off_row['start_time'], off_row['end_time']
	start_time_2, end_time_2 = off_row['start_time_2'], off_row['end_time_2']
	max_time_diff = spatial_params['merge_max_time_diff']
	min_overlap = spatial_params['merge_min_off_overlap']  # TODO
	return nearby_off_df.index[
		nearby_off_df['keep']
		& (
			(
				(
					nearby_off_df['start_time'].between(
						start_time - max_time_diff,
						start_time + max_time_diff,
					)
				)
				& (
					nearby_off_df['end_time'].between(
						end_time - max_time_diff,
						end_time + max_time_diff,
					)
				)
			) # Start/end time match with (merged-)off start_time/end_time
			| (
				(
					nearby_off_df['start_time'].between(
						start_time_2 - max_time_diff,
						start_time_2 + max_time_diff,
					)
				)
				& (
					nearby_off_df['end_time'].between(
						end_time_2 - max_time_diff,
						end_time_2 + max_time_diff,
					)
				)
			) # Start/end time match with (merged-)off start_time_2/end_time_2
		)
	]


def _find_off_indices_to_merge(off_row, nearby_off_df, spatial_params):
	return np.intersect1d(
		_find_contiguous_off_indices(off_row, nearby_off_df, spatial_params),
		_find_concurrent_off_indices(off_row, nearby_off_df, spatial_params)
	)