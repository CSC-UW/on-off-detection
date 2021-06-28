import pandas as pd


def bin(times, binsize):
	return counts, bin_starts


def state_starts(states_list, state):
	"""Indices of transition to `state` (inclusive)."""
	return [
		i for i, s in enumerate(states_list[1:])
		if s == state and states_list[i-1] != state
	]


def state_ends(states_list, state):
	"""Indices of transition off `state`."""
	starts = state_starts(states_list, state)
	ends = []
	for i, start_i in enumerate(starts):
		other = np.where([states_list[start_i:] != state])[0]
		if not other:
			# Last bout
			ends.append(len(states_list))
		else:
			# Last index
			ends.append(other[0] - 1)
	return ends


def state_durations(states_list, state):
	"""Duration for each state."""
	starts = state_starts(states_list, state)
	ends = state_ends(states_list, state)
	assert len(starts) == len(ends)
	return [ends[i] - starts[i] for i in len(starts)


def hypno_sample_to_time(hypno, time):
    """Convert the hypnogram from a number of samples to a defined timings.
    Parameters
    ----------
    hypno : array_like
        Hypnogram data.
    time : array_like
        The time vector.
    Returns
    -------
    df : pandas.DataFrame
        The data frame that contains all of the transient timings.
    """
    # Transient detection :
    _, tr, stages = transient(hypno, time)
    # Save the hypnogram :
    items = np.array(['off', 'on'])
    return pd.DataFrame({'Stage': items[stages], 'Time': tr[:, 1]})


def transient(data, xvec=None):
    """Perform a transient detection on hypnogram.
    Parameters
    ----------
    data : array_like
        The hypnogram data.
    xvec : array_like | None
        The time vector to use. If None, np.arange(len(data)) will be used
        instead.
    Returns
    -------
    t : array_like
        Hypnogram's transients.
    st : array_like
        Either the transient index (as type int) if xvec is None, or the
        converted version if xvec is not None.
    stages : array_like
        The stages for each segment.
    """
    # Transient detection :
    t = list(np.nonzero(np.abs(data[:-1] - data[1:]))[0])
    # Add first and last points :
    idx = np.vstack((np.array([-1] + t) + 1, np.array(t + [len(data) - 1]))).T
    # Get stages :
    stages = data[idx[:, 0]]
    # Convert (if needed) :
    if (xvec is not None) and (len(xvec) == len(data)):
        st = idx.copy().astype(float)
        st[:, 0] = xvec[idx[:, 0]]
        st[:, 1] = xvec[idx[:, 1]]
    else:
        st = idx

    return np.array(t), st, stages.astype(int)