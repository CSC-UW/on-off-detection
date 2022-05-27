from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .methods.threshold import THRESHOLD_PARAMS, run_threshold
from .methods.hmmem import HMMEM_PARAMS, run_hmmem 
from .utils import subset_trains_list

METHODS = {
	'threshold': run_threshold,
	'hmmem': run_hmmem,
}

DF_PARAMS = {
	'threshold': THRESHOLD_PARAMS,
	'hmmem': HMMEM_PARAMS,
}


def _run_detection(
	cluster_ids,
	detection_func,
	trains_list,
	Tmax,
	bouts_df,
	params,
	output_dir,
	debug_plot_filename,
	verbose=True,
	i=None,
):
	if i is not None:
		if verbose:
			print(f'Run #{i+1}/N, cluster_ids={cluster_ids}')

	if bouts_df is not None:
		trains_list = subset_trains_list(
			trains_list,
			bouts_df
		)  # Times in cut-and-concatenated bouts
		Tmax = bouts_df.duration.sum()
		if verbose:
			print(f"Cut and concatenate bouts: subselect T={Tmax} seconds within bouts")

	on_off_df = detection_func(
		trains_list, Tmax, params,
		save=True, output_dir=output_dir,
		filename=f'{debug_plot_filename}_clusters={cluster_ids}',
		verbose=verbose,
	)
	on_off_df['cluster_ids'] = [cluster_ids] * len(on_off_df)

	if bouts_df is not None:
		if verbose:
			print("Recover original start/end times from non-cut-and-concat data")
		# Add bout info for computed on_off periods
		# - 'state' from original bouts_df
		# - Mark on/off periods that span non-consecutive bouts as 'interbout'
		# - Mark first and last bout as 'interbout'
		# - recover start/end time in original (not cut/concatenated) time (Kinda nasty)
		on_off_orig = on_off_df.copy()
		on_off_df['bout_state'] = 'interbout'
		bout_concat_start_time = 0  # Start time in cut and concatenated data
		for i, row in bouts_df.iterrows():
			bout_concat_end_time = bout_concat_start_time + row['duration']
			bout_on_off = (
				(on_off_orig['start_time'] > bout_concat_start_time)
				& (on_off_orig['end_time'] < bout_concat_end_time)
			)  # Strict comparison also excludes first and last bout
			# start and end time in cut-concatenated data
			on_off_df.loc[bout_on_off, 'start_time_relative_to_concatenated_bouts'] = on_off_df.loc[bout_on_off, 'start_time']
			on_off_df.loc[bout_on_off, 'end_time_relative_to_concatenated_bouts'] = on_off_df.loc[bout_on_off, 'end_time']
			# Start and end time in original recording 
			bout_offset = - bout_concat_start_time + row['start_time'] # Offset from concat to real time for this bout
			# print('offset', bout_offset)
			on_off_df.loc[bout_on_off, 'start_time'] = on_off_df.loc[bout_on_off, 'start_time'] + bout_offset
			on_off_df.loc[bout_on_off, 'end_time'] = on_off_df.loc[bout_on_off, 'end_time'] + bout_offset
			# bout information
			bout_state = row['state']
			on_off_df.loc[bout_on_off, 'bout_state'] = bout_state
			on_off_df.loc[bout_on_off, 'bout_idx'] = row.name
			on_off_df.loc[bout_on_off, 'bout_concat_start_time'] = row['start_time']
			on_off_df.loc[bout_on_off, 'bout_end_time'] = row['end_time']
			on_off_df.loc[bout_on_off, 'bout_duration'] = row['duration']
			# Go to next bout
			bout_concat_start_time = bout_concat_end_time

		# Total state time per condition
		for bout_state in bouts_df['state'].unique():
			total_state_time = bouts_df[bouts_df['state'] == bout_state].duration.sum()
			on_off_df.loc[on_off_df['bout_state'] == bout_state, 'bout_state_total_time'] = total_state_time
		
		on_off_df = on_off_df[on_off_df['bout_state'] != 'interbout'].reset_index(drop=True)

	return on_off_df


class OnOffModel(object):
	"""Run ON and OFF-state detection from MUA data.
	
	Args:
		trains_list (list of array-like): Sorted MUA spike times for each cluster
		cluster_ids (list of array-like): Cluster ids
		pooled_detection (bool): Single on-off detection using all clusters,
			or run a on-off detection for each cluster independently
		Tmax: End time of recording.
		params: Dict of parameters. Mandatory params depend of <method>
		output_dir: Where we save output figures and summary statistics.
	
	Kwargs:
		bouts_df (pd.DataFrame): Frame containing bouts of interest. Must contain
			'start_time', 'end_time', 'duration' and 'state' columns. If
			provided, we consider only spikes within these bouts for on-off
			detection (by cutting-and-concatenating the trains of each cluster).
			The "state", "start_time" and "end_time" of the bout each on or off
			period pertains to is saved in the "bout_state", "bout_start_time"
			and "bout_end_time" columns. ON or OFF periods that are not STRICTLY
			comprised within bouts are dismissed ()
	"""

	def __init__(
		self, trains_list, cluster_ids=None, pooled_detection=True,
		params=None, Tmax=None, method='hmmem', bouts_df=None,
		output_dir=None, debug_plot_filename=None, n_jobs=50,
		verbose=True
	):

		self.trains_list = [sorted(train) for train in trains_list]
		if cluster_ids is not None:
			assert len(cluster_ids) == len(trains_list)
			self.cluster_ids = cluster_ids
		else:
			self.cluster_ids = ['' for i in range(len(trains_list))]
		self.pooled_detection = pooled_detection
		if Tmax is None:
			Tmax = max(self.train)
		self.Tmax = Tmax
		if bouts_df is not None:
			assert all([c in bouts_df for c in ['start_time', 'end_time', 'state', 'duration']])
		self.bouts_df = bouts_df
		self.n_jobs = n_jobs

		# Method and params
		self.method=method
		if self.method not in METHODS.keys():
			raise ValueError('Unrecognized method.')
		self.detection_func = METHODS[method]
		if params is None:
			params = {}
		unrecognized_params = set(params.keys()) - set(DF_PARAMS[method].keys())
		if len(unrecognized_params):
			raise ValueError(
				f"Unrecognized parameter keys for on-off detection method `{method}`: "
				f"{unrecognized_params}.\n\n"
				f"Default (recognized) parameters for this method: {DF_PARAMS[method]}"
			)
		self.params = {k: v for k, v in DF_PARAMS[method].items()}
		self.params.update(params)

		# Output stuff
		self.verbose=verbose
		if output_dir is None:
			output_dir='.'
		self.output_dir = Path(output_dir)
		self.debug_plot_filename=debug_plot_filename
		self.on_off_df = None
		self.stats = None

	def run(self):
		if self.pooled_detection:
			if self.verbose:
				print("Run on-off detection on pooled data")
			self.on_off_df = _run_detection(
				self.cluster_ids,
				self.detection_func,
				self.trains_list,
				self.Tmax,
				self.bouts_df,
				self.params,
				self.output_dir/'plots',
				self.debug_plot_filename,
				self.verbose,
				i=None,
			)
		else:
			if self.verbose:
				print(f"Run on-off detection for each cluster independently (N={len(self.cluster_ids)})")


			if self.n_jobs == 1:
				on_off_dfs = []
				for i, cluster_id in tqdm(enumerate(range(len(self.cluster_ids)))):
					cluster_on_off_df = _run_detection(
						[cluster_id],
						self.detection_func,
						[self.trains_list[i]],
						self.Tmax,
						self.bouts_df,
						self.params,
						self.output_dir/'plots',
						self.debug_plot_filename,
						self.verbose,
						i=i,
					)
					on_off_dfs.append(cluster_on_off_df)
			else:
				from joblib import Parallel, delayed
				on_off_dfs = Parallel(n_jobs=self.n_jobs, backend='multiprocessing')(
					delayed(_run_detection)(
						cluster_id,
						self.detection_func,
						[self.trains_list[i]],
						self.Tmax,
						self.bouts_df,
						self.params,
						self.output_dir/'plots',
						self.debug_plot_filename,
						self.verbose,
						i,
					)
					for i, cluster_id in enumerate(self.cluster_ids)
				)

			print("Done getting all units' on off periods")
			self.on_off_df = pd.concat(on_off_dfs)

		return self.on_off_df
	
	# def save():
	# 	if self.output_dir is None:
	# 		raise ValueError()
	# 	self.output_dir.mkdir(exist_ok=True, parents=True)
	# 	self.res.to_csv(self.output_dir/'on-off-times.csv')
	# 	self.stats.to_csv(self.output_dir/'on-off-stats.csv')

