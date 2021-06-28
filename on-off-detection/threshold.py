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

from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema
import pandas as pd
import numpy as np

from . import utils


THRESHOLD_PARAMS = {
	'binsize': 0.010,
	'smooth_sd': 3, # Units of binsize
}

def run_threshold(spike_times, Tmax, params, output_dir):

	## Get "count threshold" from histogram of smoothed bin counts
	bin_counts, bin_times = utils.bin(spike_times, params['binsize'])
	smoothed_bin_counts = gaussian_filter(
		bin_counts,
		params['smooth_sd']
	)
	hist, hist_bins = np.histogram(
		smoothed_bin_counts,
		bins=int((max(smoothed_bin_counts) - min(smoothed_bin_counts)/10))
	) # bin width = 10 (ms)
	count_threshold = min(hist) # Or second min?

	# Active/silent periods
	active_bin = [count >= count_threshold for count in bin_counts]

	# Get "gap threshold" from all off periods durations
	off_durations = utils.state_durations(
		active_bin, 0
	)
	duration_hist, _ = np.histogram(
		off_durations,
		bins=int((max(off_durations) - min(off_durations)/1))
	) # bin width = 1 * binsize
	# TODO ? smooth duration hist
	local_min_idx = argrelextrema(duration_hist, np.less)
	gap_threshold = duration_hist[local_min_idx[0]]

	# Merge active states separated by less than gap_threshold
	off_starts = utils.state_starts(active_bin, 0)
	off_ends = utils.state_ends(active_bin, 0)
	for i, off_dur in enumerate(off_durations):
		if off_durations[i] <= gap_threshold:
			active_bin[off_starts[i]:off_ends[i]] = 1
	
	period = ['on' if a else 'off' for a in active_bin]

	return pd.DataFrame({
		'period': ['on' for i in ]

	})






