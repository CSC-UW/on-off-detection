import pandas as pd
import numpy as np


def subset_spike_times_list(spike_times_list, bouts_df):
	"""Subset spikes within bouts.
	
	Return spike times relative to concatenated bouts.
	"""
	assert "start_time" in bouts_df.columns
	assert "end_time" in bouts_df.columns
	# TODO Validate that ther's no overlapping bout

	return [
		subset_spike_times(spike_times, bouts_df) for spike_times in spike_times_list
	]


def subset_spike_times(spike_times, bouts_df):
	res = []
	current_concatenated_start = 0
	for _, row in bouts_df.iterrows():
		start, end = row.start_time, row.end_time
		duration = end - start
		res += [
			s - start + current_concatenated_start
			for s in spike_times
			if s >= start and s <= end
		]
		current_concatenated_start += duration
	return res


def merge_spike_times(spike_times_list):
	return sorted([inner for outer in spike_times_list for inner in outer])


def state_starts(states_list, state):
	"""Indices of transition to `state` (inclusive)."""
	return np.array([
		i for i, s in enumerate(states_list)
		if s == state and (i == 0 or (states_list[i-1] != state))
	])


def state_ends(states_list, state):
	"""Indices of transition off `state`."""
	starts = list(state_starts(states_list, state))
	other_starts = []
	for s in [s for s in np.unique(states_list) if s != state]:
		other_starts += list(state_starts(states_list, s))
	other_starts = sorted(other_starts)

	ends = []
	for i, start_i in enumerate(starts):
		# next_other_starts = [st for st in other_starts if st > start_i]
		next_other_start = next((st for st in other_starts if st > start_i), None)
		if not next_other_start:
			# Last bout
			ends.append(len(states_list))
		else:
			ends.append(next_other_start)
		# For speed
		if i % 5000 == 0:
			other_starts = [st for st in other_starts if st >= start_i]


	return np.array(ends)


def state_durations(states_list, state, srate=None, starts=None, ends=None):
	"""Duration for each state.
	
	Converted to seconds if srate is provided.
	"""
	from time import time
	if starts is None:
		starts = state_starts(states_list, state)
	if ends is None:
		ends = state_ends(states_list, state)
	assert len(starts) == len(ends)
	res = np.array(
		[ends[i] - starts[i] for i in range(len(starts))],
		dtype='float'
	)
	if srate is None:
		return res
	else:
		return res / srate