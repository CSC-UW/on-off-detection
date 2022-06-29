# on_off_detection: Detect ON/OFF periods from MUA or single-unit activity.


## Examples

### Detect all ON and OFF periods from MUA


```python
from on_off_detection import OnOffModel
import numpy as np

Tmax = 3600 # (s)
cluster_ids = [
    '1',
    '2',
]
trains_list = (
    (np.random.randint(0, Tmax, 300) # 300 Spikes for cluster 1
    (np.random.randint(0, Tmax, 300) # 300 Spikes for cluster 2
)

#### Threshold based method #######
on_off_method = 'threshold'  # Implements: Zhe Chen, Sujith Vijayan, Riccardo Barbieri, Matthew A. Wilson, Emery N. Brown; Discrete- and Continuous-Time Probabilistic Models and Algorithms for Inferring Neuronal UP and DOWN States. Neural Comput 2009; 21 (7): 1797–1862.  doi: https://doi.org/10.1162/neco.2009.06-08-799
on_off_params = {
	'binsize': 1,
	'smooth_sd_counts': 3, # Units of binsize
	'binsize_smoothed_count_hist': 0.02, # 
	'smooth_sd_smoothed_count_hist': 1, # Units of "binsize_smoothed_count_hist"
	'binsize_duration_hist': 0.01, # 
	'count_threshold': None,
	'gap_threshold': None,
}

#### HMM-EM ########
on_off_method = 'hmmem'  # Implements: Zhe Chen, Sujith Vijayan, Riccardo Barbieri, Matthew A. Wilson, Emery N. Brown; Discrete- and Continuous-Time Probabilistic Models and Algorithms for Inferring Neuronal UP and DOWN States. Neural Comput 2009; 21 (7): 1797–1862.  doi: https://doi.org/10.1162/neco.2009.06-08-799
on_off_params = {
    'binsize': 0.010, # (s) (Discrete algorithm)
    'history_window_nbins': 3, # Size of history window IN BINS
    'n_iter_EM': 200,  # Number of iterations for EM
    'n_iter_newton_ralphson': 100,
    'init_A': np.array([[0.1, 0.9], [0.01, 0.99]]), # Initial transition probability matrix
    'init_mu': None,  # ~ OFF rate. Fitted to data if None
    'init_alphaa': None,  # ~ difference between ON and OFF rate. Fitted to data if None
    'init_betaa': None, # ~ Weight of recent history firing rate. Fitted to data if None,
    'gap_threshold': None, # Merge active states separated by less than gap_threhsold
}
##########

on_off_model = OnOffModel(
    trains_list,
    cluster_ids,
    Tmax,
    method=method,
    params=params,
)
on_off_df = model.run()
on_off_df = model.on_off_df # Pandas frame with 'state', 'start_time', 'end_time', 'duration' columns
```

### Spatial OFF detection

```python
from on_off_detection import SpatialOffModel
import numpy as np

Tmax = 3600 # (s)
cluster_ids = [
    '1', '2', '3'
]
cluster_depths = [
    1, 2, 3,
]
trains_list = (
    (np.random.randint(0, Tmax, 300) # 300 Spikes for cluster 1
    (np.random.randint(0, Tmax, 300) # 300 Spikes for cluster 2
    (np.random.randint(0, Tmax, 300) # 300 Spikes for cluster 3
)


# Method used for ON-OFF detection within each spatial window
# (Same as OnOffModel)

#### HMM-EM ########
on_off_method = 'hmmem'  # Implements: Zhe Chen, Sujith Vijayan, Riccardo Barbieri, Matthew A. Wilson, Emery N. Brown; Discrete- and Continuous-Time Probabilistic Models and Algorithms for Inferring Neuronal UP and DOWN States. Neural Comput 2009; 21 (7): 1797–1862.  doi: https://doi.org/10.1162/neco.2009.06-08-799
on_off_params = {
    'binsize': 0.010, # (s) (Discrete algorithm)
    'history_window_nbins': 3, # Size of history window IN BINS
    'n_iter_EM': 200,  # Number of iterations for EM
    'n_iter_newton_ralphson': 100,
    'init_A': np.array([[0.1, 0.9], [0.01, 0.99]]), # Initial transition probability matrix
    'init_mu': None,  # ~ OFF rate. Fitted to data if None
    'init_alphaa': None,  # ~ difference between ON and OFF rate. Fitted to data if None
    'init_betaa': None, # ~ Weight of recent history firing rate. Fitted to data if None,
    'gap_threshold': None, # Merge active states separated by less than gap_threhsold
}

# Parameters for aggregation of OFF states across windows
spatial_params = {
	# Definition of windows
	'window_size_min': 1,  # (um) Smallest spatial "grain" for pooling
	'window_overlap': 0.5,  # (no unit) Overlap between windows within each spatial grain 
	'window_size_step': 1,  # (um) Increase in size of windows across successive spatial "grains"
	# Merging of OFF state between and across grain
	'merge_max_time_diff': 0.050, # (s). To be merged, off states need their start & end times to differ by less than this
	'nearby_off_max_time_diff': 3, # (sec)
	'sort_all_window_offs_by': ['off_area', 'duration', 'start_time', 'end_time'],  # # How to sort all OFFs before iteratively merging
	'sort_all_window_offs_by_ascending': [False, False, True, True],
}

spatial_off_model = OnOffModel(
    trains_list,
    cluster_ids,
    Tmax,
    method=method,
    params=params,
)
off_df = model.run()

all_windows_on_off_df = spatial_off_model.all_windows_on_off_df # On/Off across all windows. Pandas frame with 'state', 'start_time', 'end_time', 'duration', 'window_depths' columns
off_df = spatial_off_model.off_df # Offs after merging. Pandas frame with 'state', 'start_time', 'end_time', 'duration', 'window_depths' columns
```


### Detect OFF state evoked by stimulation

#TODO 
Check out the "EvokedOff" class