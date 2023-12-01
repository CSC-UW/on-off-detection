import pandas as pd
import numpy as np
from tqdm import tqdm

def subset_sorted_train(bouts_df: pd.DataFrame, train: np.ndarray[float]) -> np.ndarray[bool]:
    """Return spikes covered by bouts."""

    assert np.all(np.diff(train) >= 0), "The times must be increasing."
    assert train.ndim == 1

    epoch_idxs = np.searchsorted(
        train, np.c_[bouts_df.start_time.to_numpy(), bouts_df.end_time.to_numpy()]
    )
    result = np.full_like(train, fill_value=False, dtype="bool")
    for i, ep in enumerate(epoch_idxs):
        result[ep[0] : ep[1]] = True
    return train[result]

def slice_and_concat_sorted_train(train, bouts_df):
    "Spike times relative to cumulative time within bouts."
    train = np.array(train)
    current_concatenated_start = 0
    res = []
    for row in bouts_df.itertuples():
        start, end = row.start_time, row.end_time
        duration = end - start
        start_i = np.searchsorted(train, start, side="left")
        end_i = np.searchsorted(train, end, side="right")
        res.append(train[start_i:end_i] - start + current_concatenated_start)
        current_concatenated_start += duration
    return np.hstack(res)


def merge_trains_list(trains_list):
    return sorted([inner for outer in trains_list for inner in outer])


def state_starts(states_list, state):
    """Indices of transition to `state` (inclusive)."""
    return np.array(
        [
            i
            for i, s in enumerate(states_list)
            if s == state and (i == 0 or (states_list[i - 1] != state))
        ]
    )


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
    res = np.array([ends[i] - starts[i] for i in range(len(starts))], dtype="float")
    if srate is None:
        return res
    else:
        return res / srate
