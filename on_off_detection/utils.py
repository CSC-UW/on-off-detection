import pandas as pd
import numpy as np


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
		next_other_starts = [st for st in other_starts if st > start_i]
		if not next_other_starts:
			# Last bout
		    ends.append(len(states_list))
		else:
			ends.append(min(next_other_starts))

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