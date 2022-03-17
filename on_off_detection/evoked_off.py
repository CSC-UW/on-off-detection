from pathlib import Path

import pandas as pd
from tqdm import tqdm
from scipy.ndimage import median_filter
import numpy as np

STDPERIOD_PARAMS = {
	'start_tmin': 0,
	'start_tmax': 500,
	'medfilt_size_t': 20,
	'std_threshold': 2,
}

DF_PARAMS = {
	'stdperiod': STDPERIOD_PARAMS,
}


def run_stdperiod_off_detection(cluster_psth_data, times, params, debug=False):
	"""Return pd.DataFrame containing OFF period info.

	
	First, we smooth the array with a median filter of size
	`params['medfilt_size_t']` (defined in seconds).  The evoked OFF period is
	the longest period starting `params['start_tmin']` and
	`params['start_tmax']` during which data is below the 
	baseline average by more than `params['std_thres']` times the baseline
	standard deviation.

    Args:
        cluster_psth_data (list-like): (1 x nbins) psth
        times (list-like): Timestamp for each bin in seconds
        params (dict): Algorithm parameters. All keys are mandatory:
			`start_tmin`, `start_tmax`: Min and max time within which the OFF period may start
			`medfilt_size_t`: Size of the median filter applied to the data (same unit as binsize)
			`std_threshold`: Threshold for period detection, in number of standard deviations below baseline 

	Return
		pd.DataFrame: frame containing the following columns:
			`start_time`, `end_time`, `start_idx`, `end_idx`, `duration`: OFF
				start/end time/indices and duration
			`min_raw_idx`, `min_raw_time`, `min_raw_value`,
				`min_raw_value_zscore`: Idx, time, value and zscored value of period's minimum (in raw data)
			`min_filt_idx`, `min_filt_time`, `min_filt_value`,
				`min_filt_value_zscore`: Same for filtered data
			`threshold`: Threshold used for OFF detection (derived from baseline mean/std and `std_threshold`)
	"""

	# Check input
	assert cluster_psth_data.squeeze().ndim == 1
	dat = cluster_psth_data.squeeze()  # True 1D
	nbins = len(times)
	binsize = times[1] - times[0]
	assert len(dat) == nbins
	assert times[0] < 0
	assert times[-1] > 0
	assert params['start_tmin'] < params['start_tmax']
	assert params['start_tmax'] < times[-1]
	assert params['std_threshold'] > 0

	# Baseline std and mean
	baseline_idx = np.array([idx for idx, t in enumerate(times) if t < -binsize])
	baseline_mean = np.mean(dat[baseline_idx])
	baseline_std = np.std(dat[baseline_idx])

	# Smooth data
	filtsize_t = params['medfilt_size_t']
	filtsize_idx = int(filtsize_t / binsize)
	datfilt = median_filter(dat, filtsize_idx)

	thres = baseline_mean - params['std_threshold'] * baseline_std

	# Find start and end time of all below-threshold periods
	# that start within the time-range of interest

	periods_df = pd.DataFrame()  # Fill info for all periods
	# Initialize at params['start_tmin'], end when we're out of bounds
	start_idx = next(idx for idx in range(nbins) if times[idx] > params['start_tmin'])
	while times[start_idx] <= params['start_tmax']:

		if datfilt[start_idx] > thres: # Above threshold -> Start at next sample
			start_idx += 1
			continue

		# End period at next above-threshold bin
		end_idx = next(
			idx for idx in range(start_idx+1, nbins)
			if datfilt[idx] > thres
		)
		# Fill in period info
		min_filt_idx = start_idx + np.argmin(datfilt[start_idx:end_idx])
		min_filt_time = times[min_filt_idx]
		min_filt_value = datfilt[min_filt_idx]
		min_filt_value_zscore = (min_filt_value - baseline_mean) / baseline_std
		min_raw_idx = start_idx + np.argmin(dat[start_idx:end_idx])
		min_raw_time = times[min_raw_idx]
		min_raw_value = dat[min_raw_idx]
		min_raw_value_zscore = (min_raw_value - baseline_mean) / baseline_std
		periods_df = periods_df.append(pd.DataFrame({
			'evoked_off': True,
			'start_time': times[start_idx],
			'start_idx': start_idx,
			'end_time': times[end_idx],
			'end_idx': end_idx,
			'duration': times[end_idx] - times[start_idx],
			'threshold': thres,
			'min_filt_idx': min_filt_idx,
			'min_filt_time': min_filt_time,
			'min_filt_value': min_filt_value,
			'min_filt_value_zscore': min_filt_value_zscore,
			'min_raw_idx': min_raw_idx,
			'min_raw_time': min_raw_time,
			'min_raw_value': min_raw_value,
			'min_raw_value_zscore': min_raw_value_zscore,
			**params,
		}, index=[0]))

		# Go to next period
		start_idx = end_idx + 1
	
	if debug and len(periods_df):
		import matplotlib.pyplot as plt
		plt.plot(times, dat)
		plt.plot(times, datfilt)
		for _, row in periods_df.iterrows():
			plt.axvline(x=row['start_time'], color='r')
			plt.axvline(x=row['end_time'], color='r')
			plt.plot(row['min_filt_time'], row['min_filt_value'], '+')
			plt.plot(row['min_raw_time'], row['min_raw_value'], '+')
		print('Save debugging plot at ./debug.png')
		plt.savefig('debug.png')
	
	if len(periods_df):
		# Return only the longest period
		off_idx = periods_df['duration'].idxmax()
		off_df = periods_df.iloc[off_idx:off_idx+1, :]
	else:
		# No detected off
		off_df = pd.DataFrame({
			'evoked_off': False,
			'start_time': None,
			'start_idx': None,
			'end_time': None,
			'end_idx': None,
			'duration': None,
			'threshold': thres,
			'min_filt_idx': None,
			'min_filt_time': None,
			'min_filt_value': None,
			'min_filt_value_zscore': None,
			'min_raw_idx': None,
			'min_raw_time': None,
			'min_raw_value': None,
			'min_raw_value_zscore': None,
			**params,
		}, index=[0])

	return off_df


METHODS = {
	'stdperiod': run_stdperiod_off_detection,
}



class EvokedOffModel(object):
	"""Run OFF-state detection from PSTH data
	
	Args:
		psth_data (array-like): (nclusters x nbins) array containing PSTH values
		binsize (float): Binsize of psth
		window (float): window of PSTH around event.
		cluster_ids (list of array-like): Cluster ids
		params: Dict of parameters. Mandatory params depend of <method>
	"""

	def __init__(
		self, psth_data, binsize, window, cluster_ids=None,
		params=None, method='stdperiod',
		debug=False, n_jobs=1,
	):
		if psth_data.ndim == 1:  # Make 2D even if there's 1 cluster
			psth_data = np.reshape(psth_data, (1, len(psth_data)))
		nclusters, nbins = psth_data.shape
		self.nclusters = nclusters
		self.nbins = nbins
		self.window = window
		self.binsize = window
		self.times = np.arange(window[0], window[1], binsize)
		self.psth_data = psth_data
		if cluster_ids is not None:
			assert len(cluster_ids) == nclusters
			self.cluster_ids = cluster_ids
		else:
			self.cluster_ids = ['' for i in range(nclusters)]
		self.method=method
		if self.method not in METHODS.keys():
			raise ValueError('Unrecognized method.')
		self.detection_func = METHODS[method]
		self.params = {k: v for k, v in DF_PARAMS[method].items()}
		if params is None:
			params = {}
		self.params.update(params)
		self.n_jobs = n_jobs
		# Output stuff
		self.debug = debug
		self.off_df = None

	def run(self):
		print(f"Run on-off detection for each cluster (N={self.nclusters})")

		if self.n_jobs == 1:
			off_dfs = []
			for i, cluster_id in tqdm(enumerate(range(len(self.cluster_ids)))):
				print(f'Run #{i+1}/N')
				cluster_off_df = self.detection_func(
					self.psth_data[i:i+1, :], self.times, self.params,
					debug=self.debug,
				)
				cluster_off_df['cluster_id'] = cluster_id
				off_dfs.append(cluster_off_df)
		else:
			raise NotImplementedError

		print("Done getting all clusters off periods")
		self.off_df = pd.concat(off_dfs)

		return self.off_df


if __name__ == "__main__":

	# Test 
	binsize = 2
	window = (-1000, 2000)
	times = np.arange(window[0], window[1], binsize)
	dat = np.random.normal(2, 2, len(times))

	dat[550:650] = dat[550:650] - 10
	dat[700:800] = dat[700:800] - 10

	run_stdperiod_off_detection(
		dat, times, STDPERIOD_PARAMS, debug=True
	)

	model = EvokedOffModel(dat, binsize, window, params=STDPERIOD_PARAMS)
	print(model.run())
