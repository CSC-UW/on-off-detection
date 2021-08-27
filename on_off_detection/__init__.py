from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .threshold import THRESHOLD_PARAMS, run_threshold

METHODS = {
	'threshold': run_threshold,
}

DF_PARAMS = {
	'threshold': THRESHOLD_PARAMS,
}


def run_cluster(
	i,
	cluster_id,
	detection_func,
	spike_times_list,
	Tmax,
	params,
	output_dir,
	debug_plot_filename,
):
	print(f'Run #{i+1}/N')
	cluster_on_off_df = detection_func(
		spike_times_list, Tmax, params,
		save=True, output_dir=output_dir,
		filename=f'{debug_plot_filename}_cluster={cluster_id}',
	)
	cluster_on_off_df['cluster_id'] = cluster_id
	return cluster_on_off_df


class OnOffModel(object):
	"""Run ON and OFF-state detection from MUA data.
	
	Args:
		spike_times_list (list of array-like): Sorted MUA spike times for each cluster
		cluster_ids (list of array-like): Cluster ids
		pooled_detection (bool): Single on-off detection using all clusters,
			or run a on-off detection for each cluster independently
		Tmax: End time of recording.
		params: Dict of parameters. Mandatory params depend of <method>
		output_dir: Where we save output figures and summary statistics.
	"""

	def __init__(
		self, spike_times_list, cluster_ids=None, pooled_detection=True,
		params=None, Tmax=None, method='threshold',
		output_dir=None, debug_plot_filename=None, hyp=None, n_jobs=50,
	):
		self.spike_times_list = [sorted(spike_times) for spike_times in spike_times_list]
		if cluster_ids is not None:
			assert len(cluster_ids) == len(spike_times_list)
			self.cluster_ids = cluster_ids
		else:
			self.cluster_ids = ['' for i in range(len(spike_times_list))]
		self.pooled_detection = pooled_detection
		self.Tmax = Tmax
		if self.Tmax is None:
			self.Tmax = max(self.spike_times)
		self.method=method
		if self.method not in METHODS.keys():
			raise ValueError('Unrecognized method.')
		self.detection_func = METHODS[method]
		self.params = {k: v for k, v in DF_PARAMS[method].items()}
		if params is None:
			params = {}
		self.params.update(params)
		self.hyp = hyp
		self.n_jobs = n_jobs
		# Output stuff
		self.output_dir = output_dir
		if output_dir is not None:
			self.output_dir = Path(output_dir)
		self.debug_plot_filename=debug_plot_filename
		self.on_off_df = None
		self.stats = None

	def run(self):
		if self.pooled_detection:
			print("Run on-off detection on pooled data")
			self.on_off_df = self.detection_func(
				self.spike_times_list, self.Tmax, self.params,
				save=True, output_dir=self.output_dir/'plots',
				filename=self.debug_plot_filename,
			)
		else:
			print(f"Run on-off detection for each cluster (N={len(self.cluster_ids)})")


			if self.n_jobs == 1:
				on_off_dfs = []
				for i, cluster_id in tqdm(enumerate(range(len(self.cluster_ids)))):
					print(f'Run #{i+1}/N')
					cluster_on_off_df = self.detection_func(
						[self.spike_times_list[i]], self.Tmax, self.params,
						save=True, output_dir=self.output_dir/'plots',
						filename=f'{self.debug_plot_filename}_cluster={cluster_id}',
					)
					cluster_on_off_df['cluster_id'] = cluster_id
					on_off_dfs.append(cluster_on_off_df)
			else:
				from joblib import Parallel, delayed
				on_off_dfs = Parallel(n_jobs=self.n_jobs, backend='multiprocessing')(
					delayed(run_cluster)(
						i,
						cluster_id,
						self.detection_func,
						[self.spike_times_list[i]],
						self.Tmax,
						self.params,
						self.output_dir/'plots',
						self.debug_plot_filename
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

