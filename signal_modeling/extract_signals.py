import matplotlib.pyplot as plt
import json
import setigen as stg
import blscint as bls
import blimpy as bl
import pandas as pd
import numpy as np
from astropy.time import Time
from astropy.stats import sigma_clip
import astropy.units as u

DATA_PREFIX = '/mnt_blpd2/datax/PKSUWL/blcxx/PKSUWL'

META = pd.read_csv('first_11_meta_info.csv', sep=';')
META['indexes'] = META['indexes'].str.strip()

DRIFTS = pd.read_csv('first_11_drift_rates.csv', sep=';')
CENTERS = pd.read_csv('first_11_start_frequencies.csv', sep=';')


def get_ts_info():
    """
    Get timestamps for observations.
    
    Returns a dictionary with key 'ts_list', containing a list of timestamp
    arrays matching our observations. Also has 'dt' and 'tstart' for 
    convenience.
    """
    ts_info = {}
    ts_list = []
    for j in range(len(DRIFTS)):
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
        ts_list.append(x - ts_info['tstart'])
        
    ts_info['ts_list'] = ts_list
    return ts_info


def get_full_ts(ts_info):
    """
    Return full timestamp array for use with setigen.
    """
    return np.concatenate(ts_info['ts_list'])


def single_obs_time_series(candidate, obs):
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
    output = {}
        
    meta_row = META.loc[candidate]
    filename = DRIFTS['filename'].loc[obs]
    full_path = f'{DATA_PREFIX}/{filename}'.replace('blcxx', 
                                                    meta_row['node'])

    drift_rate = DRIFTS[meta_row['lookalike']].loc[obs]
    center_freq = CENTERS[meta_row['lookalike']].loc[obs]
    if pd.isnull(drift_rate):
        output['scaled_ts'] = None
        output['drift_rate'] = None
        output['center_freq'] = None
    else:
        frame = bls.centered_frame(fn=full_path,
                                   drift_rate=drift_rate,
                                   center_freq=center_freq,
                                   fchans=256)
        frame = stg.dedrift(frame, drift_rate=drift_rate)
        l, r, _ = bls.threshold_bounds(frame.integrate())
        n_frame = bls.t_norm_frame(frame)
        tr_frame = n_frame.get_slice(l, r)
        tr_y = tr_frame.integrate('f', mode='sum')
#             tr_y /= tr_y.mean()
        # normalize tr_y by noise levels
        tr_y /= np.std(sigma_clip(tr_frame.get_data()))

        output['scaled_ts'] = tr_y
        output['drift_rate'] = drift_rate
        output['center_freq'] = center_freq

    return output


def all_time_series():
    """
    Compute time series of all candidates, over all exposures.
    """
    all_y = []
    for i, meta_row in META.iterrows():
        signal_info = {
            'name': meta_row['lookalike'],
            'signal' : []
        }
        for j in range(len(DRIFTS)):
            signal_info['signal'].append(single_obs_time_series(i, j))
        all_y.append(signal_info)
    return all_y
