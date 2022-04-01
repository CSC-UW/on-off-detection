# on_off_detection: Detect, ON/OFF periods from MUA or single-unit activity.


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
spike_times_list = (
    (np.random.randint(0, Tmax, 300) # 300 Spikes for cluster 1
    (np.random.randint(0, Tmax, 300) # 300 Spikes for cluster 2
)

method = 'threshold'  # Implements: Zhe Chen, Sujith Vijayan, Riccardo Barbieri, Matthew A. Wilson, Emery N. Brown; Discrete- and Continuous-Time Probabilistic Models and Algorithms for Inferring Neuronal UP and DOWN States. Neural Comput 2009; 21 (7): 1797–1862.  doi: https://doi.org/10.1162/neco.2009.06-08-799
params = {
	'binsize': 1,
	'smooth_sd_counts': 3, # Units of binsize
	'binsize_smoothed_count_hist': 0.02, # 
	'smooth_sd_smoothed_count_hist': 1, # Units of "binsize_smoothed_count_hist"
	'binsize_duration_hist': 0.01, # 
	'count_threshold': None,
	'gap_threshold': None,
}

model = OnOffModel(
    spike_times_list,
    cluster_ids,
    method=method,
    params=params,
    pooled_detection=True,  # Pool spikes from all clusters
    Tmax=Tmax,  # Pool spikes from all clusters
)
on_off_df = model.run()
on_off_df = model.on_off_df # Pandas frame with 'state', 'start_time', 'end_time', 'duration' columns
```



### Detect OFF state evoked by stimulation

#TODO