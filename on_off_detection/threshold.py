"""
Implements:

Zhe Chen, Sujith Vijayan, Riccardo Barbieri, Matthew A. Wilson, Emery N. Brown;
Discrete- and Continuous-Time Probabilistic Models and Algorithms for Inferring Neuronal UP and DOWN States. Neural Comput 2009; 21 (7): 1797–1862.
doi: https://doi.org/10.1162/neco.2009.06-08-799

Appendix B:
The standard threshold-based method for determining the UP and DOWN states based
on MUA spike trains (Ji & Wilson, 2007) consists of three major steps.

First, we bin the spike trains into 10 ms windows and calculate the raw spike
counts for all time intervals. The raw count signal is smoothed by a gaussian
window with an SD of 30 ms to obtain the smoothed count signal over time. We
then calculate the first minimum (count threshold value) of the smoothed spike
count histogram during SWS. As the spike count has been smoothed, the count
threshold value may be a noninteger value.

Second, based on the count threshold value, we determine the active and silent
periods for all 10 ms bins. The active periods are set to 1, and silent periods
are set to 0. Next, the duration lengths of all silent periods are computed. We
then calculate the first local minimum (gap threshold) of the histogram of the
silent period durations.

Third, based on the gap threshold value, we merge those active periods separated
by silent periods in duration less than the gap threshold. The resultant active
and silent periods are classified as the UP and DOWN states, respectively.
Finally, we recalculate the duration lengths of all UP and DOWN state periods
and compute their respective histograms and sample statistics (min, max, median,
mean, SD).

In summary, the choices of the spike count threshold and the gap threshold will
directly influence the UP and DOWN state classification and their statistics (in
terms of duration length and occurrence frequency). However, the optimal choices
of these two hand-tuned parameters are rather ad hoc and dependent on several
issues (e.g., kernel smoothing, bin size; see Figure 16 for an illustration). In
some cases, no minimum can be found in the smoothed histogram, and then the
choice of the threshold is problematic. Note that the procedure will need to be
repeated for different data sets such that the UP and DOWN states statistics can
be compared.
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from ecephys.plot import plot_on_off_overlay, plot_spike_train
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema

from . import utils


THRESHOLD_PARAMS = {
	'binsize': 0.010,
	'smooth_sd_counts': 3, # Units of binsize
	# 'smooth_sd_hist': 3, # No units
}


def run_threshold(spike_times_list, Tmax, params, output_dir=None, show=True, save=False, filename=None):
	"""Return dataframe of on/off periods.

	Args:
		spike_times_list: List of array likes.

	Return:
		pd.DataFrame: df with 'state', 'start_time', 'end_time' and 'duration' columns
	"""
	print(f"method=threshold, params: {params}")
	srate = int(1/params['binsize'])
	N_off_durations_bins = 100 # 100 bins/sec
	N_count_bins = 100 # 100 bins


	spike_times = utils.merge_spike_times(spike_times_list)
	print(f"Merged N={len(spike_times_list)} spike trains for on/off detection")
	print(f"Merged pop. rate = {len(spike_times)/Tmax}Hz, N={len(spike_times)} spikes")


	## Get "count threshold" from histogram of smoothed bin counts
	print("Get count histogram...", end="")
	nbins = int(Tmax/params['binsize'])
	bin_counts, bins = np.histogram(spike_times, bins=nbins, range=[0, Tmax])
	bin_centers = np.array(bins[0:-1]) + params['binsize']/2
	smoothed_bin_counts = gaussian_filter1d(
		bin_counts, params['smooth_sd_counts'], output=float,
	)

	count_hist, hist_bins = np.histogram(
		smoothed_bin_counts,
		bins=100,
	)
	count_threshold = hist_bins[argrelextrema(count_hist, np.less)[0]][0]
	print(f'Count threshold = {count_threshold}')

	# Active/silent periods
	active_bin = np.array([count >= count_threshold for count in bin_counts])

	# Get "gap threshold" from all off periods durations
	print("Get off period duration histogram...", end="")
	off_durations_pre = utils.state_durations(
		active_bin, 0, srate=srate,
	) # (sec)
	nbins = int(max(off_durations_pre)*N_off_durations_bins)
	duration_hist_pre, duration_bins_pre = np.histogram(
		off_durations_pre,
			bins=max(nbins, 1)
	) # 100 bins / sec
	duration_bin_centers_pre = (np.array(duration_bins_pre[0:-1]) + np.array(duration_bins_pre[0:-1])) / 2
	local_min_idx = argrelextrema(duration_hist_pre, np.less)[0]
	if not len(local_min_idx):
		from warnings import warn
		warn("No strict local min in off-durations histogram. try with flat minima.")
		local_min_idx = argrelextrema(duration_hist_pre, np.less_equal)[0]
		if not len(local_min_idx):
			warn("No flat local min in off-durations histogram. Set gap threshold to 0.")
			gap_threshold = 0.0
		else:
			gap_threshold = duration_bin_centers_pre[local_min_idx[0]]
	else:
		gap_threshold = duration_bin_centers_pre[local_min_idx[0]]
	print(f'Gap threshold = {gap_threshold}(s)')

	# Merge active states separated by less than gap_threshold
	print("Merge closeby on-periods...", end="")
	off_starts = utils.state_starts(active_bin, 0)
	off_ends = utils.state_ends(active_bin, 0)
	N_merged = 0
	for i, off_dur in enumerate(off_durations_pre):
		if off_dur <= gap_threshold:
			active_bin[off_starts[i]:off_ends[i]+1] = 1
			N_merged += 1
	print(f'Merged N={N_merged} active periods')
	
	# Return df
	# all in (sec)
	print("Get final on/off periods df...", end="")
	on_starts = utils.state_starts(active_bin, 1) / srate
	off_starts = utils.state_starts(active_bin, 0) / srate
	on_ends = utils.state_ends(active_bin, 1) / srate
	off_ends = utils.state_ends(active_bin, 0) / srate
	on_durations = utils.state_durations(
		active_bin, 1, srate=srate, starts=on_starts, ends=on_ends,
	)
	off_durations = utils.state_durations(
		active_bin, 0, srate=srate, starts=off_starts, ends=off_ends,
	)
	N_on = len(on_starts)
	N_off = len(off_starts)

	on_off_df = pd.DataFrame({
		'state': ['on' for i in range(N_on)] + ['off' for i in range(N_off)],
		'start_time': list(on_starts) + list(off_starts),
		'end_time': list(on_ends) + list(off_ends),
		'duration': list(on_durations) + list(off_durations),
	}).sort_values(by='start_time').reset_index(drop=True)
	print("Done.")

	if save or show:

		print("Generate summary figure")

		# Smoothed spike count
		fig = plt.figure(figsize=(25, 15), constrained_layout=True)
		spec = gridspec.GridSpec(ncols=4, nrows=4, figure=fig)
		ax = fig.add_subplot(spec[0,0])
		ax.plot(hist_bins[:-1], count_hist, color='blue')
		ax.set_yscale('log')
		plt.axvline(x=count_threshold)
		ax.set_title(f'Smoothed spike count histogram ;\n count threshold = {count_threshold}')

		# Off period durations before merging
		ax = fig.add_subplot(spec[0,1])
		ax.plot(duration_bins_pre[:-1], duration_hist_pre, color='blue')
		ax.set_ylabel('N')
		ax.set_xlabel('Off period duration (s) (log-scaled)')
		ax.set_xscale('log')
		plt.axvline(x=gap_threshold)
		ax.set_title(f'Off period duration histogram (pre-merging) ;\n gap threshold = {gap_threshold}')

		# Final Off period durations
		nbins = int(max(off_durations)*N_off_durations_bins)
		duration_hist_post, duration_bins_post = np.histogram(
			off_durations,
			bins=max(nbins, 1)
		) # 100 bins / sec
		duration_bin_centers_post = (np.array(duration_bins_post[0:-1]) + np.array(duration_bins_post[0:-1])) / 2
		ax = fig.add_subplot(spec[0,2])
		ax.plot(duration_bins_post[:-1], duration_hist_post, color='blue')
		ax.set_title('Off period duration histogram (final)')
		ax.set_ylabel('N')
		ax.set_xlabel('Off period duration (s)')

		# Final on period durations
		nbins = int(max(on_durations)*N_off_durations_bins)
		on_duration_hist_post, on_duration_bins_post = np.histogram(
			on_durations,
			bins=max(nbins, 1)
		) # 100 bins / sec
		on_duration_bin_centers_post = (np.array(on_duration_bins_post[0:-1]) + np.array(on_duration_bins_post[0:-1])) / 2
		ax = fig.add_subplot(spec[0,3])
		ax.plot(on_duration_bins_post[:-1], on_duration_hist_post, color='blue')
		ax.set_title('On period duration histogram (final)')
		ax.set_ylabel('N')
		ax.set_xlabel('On period duration (s)')

		# Binned count
		ax = fig.add_subplot(spec[1,:])
		ax.plot(bin_centers, bin_counts, color='blue', linewidth=0.1)
		ax.set_xlim(0, Tmax)
		ax2=ax.twinx()
		ax2.plot(bin_centers, smoothed_bin_counts, color='red', linewidth=0.1)
		plot_on_off_overlay(on_off_df, ax=ax, alpha=0.2)
		ax.set_title('Binned spike count ; smoothed count')

		# raster
		ax = fig.add_subplot(spec[2,:])
		plot_spike_train(spike_times_list, ax=ax, linewidth=0.1, Tmax=Tmax)
		plot_on_off_overlay(on_off_df, ax=ax)
		ax.set_title('Spike raster')

		# Zoomed raster
		zoom = (30, 90)
		ax = fig.add_subplot(spec[3,:])
		plot_spike_train(spike_times_list, ax=ax, linewidth=1.0, Tmax=Tmax)
		plot_on_off_overlay(on_off_df, ax=ax)
		ax.set_xlim(*zoom)
		ax.set_title('Spike raster')
		if show:
			plt.show()

		if save:
			assert output_dir is not None
			output_dir = Path(output_dir)
			output_dir.mkdir(parents=True, exist_ok=True)
			if filename is None:
				filename = 'on_off_summary_threshold'
			print(f"Save summary fig at {output_dir/filename}")
			fig.savefig(output_dir/filename)
	
	return on_off_df