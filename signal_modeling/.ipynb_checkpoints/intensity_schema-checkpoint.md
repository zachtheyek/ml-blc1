Schemas for data products:

### `ts_info.npy`: timestamps for all observations
```
{
    'tstart': 58602.52766203704,
    'dt': 16.777216,
    'ts_list': [
        array(...),
        array(...),
        ...
    ]
}
```
This is calculated via `extract_signals.get_ts_info()`.

To get contiguous array of all timestamps, do
```
import extra_signals as es
ts_info = np.load('ts_info.npy', allow_pickle=True)).item()
ts = es.get_full_ts(ts_info)
```


### `intensity_data.npy`: Extracted (and arbitrarily normalized) intensities. 
List is per candidate, candidate signal list is per observation.
```
[
    {
        'name': 'blc1',
        'signal': [
            {
                'drift_rate': None,
                'center_freq': None,
                'scaled_ys': None,
                'noise_mean': None,
                'noise_std': None,
                'px_width': None
            },
            ...,
            {...}
        ]
    },
    ...,
    {...}
]
```
This is calculated via `extract_signals.all_time_series()`.

To load data, do
```
import extra_signals as es
all_y = np.load('intensity_data.npy', allow_pickle=True))
```

Note that since this is a list, we do *not* need to use ".item" to access the data.