from pathlib import Path

import pandas as pd

from .methods.threshold import THRESHOLD_PARAMS, run_threshold

METHODS = {
	'threshold': run_threshold,
}

DF_PARAMS = {
	'threshold': THRESHOLD_PARAMS,
}


def OnOffModel(object):
	"""Run ON and OFF-state detection from MUA data.
	
	Args:
		spike_times (list-like): Sorted MUA spike times
		Tmax: End time of recording.
		params: Dict of parameters. Mandatory params depend of <method>
		output_dir: Where we save output figures and summary statistics.
	"""

	def __init__(spike_times, output_dir, params=None, Tmax=None, method='threshold'):
		# Params
		self.spike_times = spike_times
		if not all(
			self.spike_times[i] <= self.spike_times[i+1]
			for i in xrange(len(self.spike_times)-1)
		):
			self.spike_times = sorted(self.spike_times)
		self.Tmax = Tmax
		if self.Tmax is None:
			self.Tmax = max(self.spike_times)
		self.method=method
		if self.method not in METHODS.keys:
			raise ValueError('Unrecognized method.')
		self.detection_func = METHODS[method]
		self.params = {k: v for k, v in DF_PARAMS[method]}
		if params is None:
			params = {}
		self.params.update(params)
		# Output stuff
		self.output_dir = Path(output_dir)
		self.res = None
		self.stats = None

	def run():
		self.res = self.detection_func(
			self.spike_times, self.Tmax, self.params, self.output_dir
		)
		self.stats = self.compute_stats(self.res)
	
	def save():
		self.output_dir.mkdir(exist_ok=True, parents=True)
		self.res.to_csv(self.output_dir/'on-off-times.csv')
		self.stats.to_csv(self.output_dir/'on-off-stats.csv')



