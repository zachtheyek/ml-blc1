import matplotlib.pyplot as plt
import json
import setigen as stg
import blscint as bls
import blimpy as bl
import pandas as pd
import numpy as np
from astropy.time import Time
from astropy.stats import sigma_clip
import astropy.units as ua
import tqdm

from params import DATA_PREFIX, META, DRIFTS, CENTERS


def get_ts_info():
    """
    Get timestamps for observations.
    
    Returns a dictionary with key 'ts_list', containing a list of timestamp
    arrays matching our observations. Also has 'dt' and 'tstart' for 
    convenience.
    """
    ts_info = {}
    ts_list = []
    for j in tqdm.tqdm(range(len(DRIFTS))):
        filename = DRIFTS['filename'].loc[j]
        # Just use first candidate to get times
        full_path = f'{DATA_PREFIX}/{filename}'.replace('blcxx', 
                                                        META.loc[0]['node'])

        container = bl.Waterfall(full_path, load_data=False).container
        tchans = container.file_shape[0]
        dt = container.header['tsamp']
        tstart = container.header['tstart']
        # Log first MJD
        if j == 0:
            ts_info['tstart'] = tstart
            ts_info['dt'] = dt
            
        t = Time(tstart, format='mjd')
        x = np.linspace(0, tchans * dt, tchans, endpoint=False) + t.unix
#         x = Time(x, format='unix').mjd

        # Subtract out tstart so that ts starts at 0
        ts_list.append(x - Time(ts_info['tstart'], format='mjd').unix)
        
    ts_info['ts_list'] = ts_list
    return ts_info


def get_full_ts(ts_info):
    """
    Return full timestamp array for use with setigen.
    """
    return np.concatenate(ts_info['ts_list'])


def single_obs_time_series(candidate, obs, fchans=256, frame_params=None):
    """
    Dedrift and set bounding boxes around signals to estimate intensity
    over time.
    
    Arguments
    ---------
    candidate : int
        Index in META dataframe, for each candidate
    obs : int
        Index in DRIFTS dataframe, represent different exposures
        
    Returns
    -------
    output : dict
        Dict with extraction details. 
    """
    meta_row = META.loc[candidate]
    filename = DRIFTS['filename'].loc[obs]
    full_path = f'{DATA_PREFIX}/{filename}'.replace('blcxx', 
                                                    meta_row['node'])
    drift_rate = DRIFTS[meta_row['lookalike']].loc[obs]
    center_freq = CENTERS[meta_row['lookalike']].loc[obs]
    
    frame = bls.centered_frame(fn=full_path,
                               drift_rate=drift_rate,
                               center_freq=center_freq,
                               fchans=fchans,
                               frame_params=frame_params)
    frame = stg.dedrift(frame, drift_rate=drift_rate)
    noise_mean, _ = frame.get_noise_stats()

    l, r, _ = bls.threshold_bounds(frame.integrate())
    n_frame = bls.t_norm_frame(frame)
    tr_frame = n_frame.get_slice(l, r)
    tr_y = tr_frame.integrate('f', mode='sum')
#             tr_y /= tr_y.mean()
    # normalize tr_y by noise levels
    noise_std = np.std(sigma_clip(n_frame.get_data()))
    tr_y /= noise_std
    
    output = {
        'scaled_ys': tr_y,
        'drift_rate': drift_rate,
        'center_freq': center_freq,
        'noise_mean': noise_mean,
        'noise_std': noise_std,
        'px_width': r - l
    }

    return output


def all_time_series(fchans=256):
    """
    Compute time series of all candidates, over all exposures.
    """
    meta_row = META.loc[0]
    filename = DRIFTS['filename'].loc[0]
    frame_params = bls.get_frame_params(f'{DATA_PREFIX}/{filename}'.replace('blcxx', 
                                                                            meta_row['node']))
    
    all_y = []
    for i, meta_row in tqdm.tqdm(META.iterrows()):
        signal_info = {
            'name': meta_row['lookalike'],
            'signal' : []
        }
        indices = json.loads(meta_row['indexes'])
        for j in range(len(DRIFTS)):
            if j in indices:
                info = single_obs_time_series(i, j, fchans=fchans, frame_params=frame_params)
            else:
                info = {
                    'scaled_ys': None,
                    'drift_rate': None,
                    'center_freq': None,
                    'noise_mean': None,
                    'noise_std': None,
                    'px_width': None
                }
            signal_info['signal'].append(info)
        all_y.append(signal_info)
    return all_y


def generate_analysis_products(prefix='', fchans=256):
    """
    Compute and save analysis products in one function.
    """
    print('Generating ts_info...')
    ts_info = get_ts_info()
    np.save(f'{prefix}ts_info.npy', ts_info)
    print('')
    print('Generating intensity_data...')
    all_y = all_time_series(fchans=fchans)
    np.save(f'{prefix}intensity_data.npy', all_y)
    
    
if __name__ == '__main__':
    generate_analysis_products(fchans=256)