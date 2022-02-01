import os, errno
import tqdm
from astropy import units as u
import numpy as np
import pandas as pd
import blimpy as bl
import setigen as stg
import argparse


from params import DATA_PREFIX, META, DRIFTS
import extract_signals as es


def mkdir(d):
    try:
        os.makedirs(d)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def get_frame_params(fn):
    """
    Get relevant parameters for grabbing data. 
    """
    container = bl.Waterfall(fn, load_data=False).container
    return {
        'tchans': container.file_shape[0],
        'df': abs(container.header['foff']) * 1e6,
        'dt': container.header['tsamp']
    }

def sinfit(t): 
    """
    Sine fit to blc1 signal. Work in 5_synthetic_dataset.ipynb.
    """
    return 444 * np.sin(2 * np.pi * 1.13492353e-05 * (t-1556542955)) + 982002391.815186

def dr_sinfit(t):
    """
    Derivative of sine fit..
    """
    return 444 * np.cos(2 * np.pi * 1.13492353e-05 * (t-1556542955)) * 2 * np.pi * 1.13492353e-05


# Compute params to match from blc1 observation
meta_row = META.loc[0]
filename = DRIFTS['filename'].loc[0]
full_path = f'{DATA_PREFIX}/{filename}'.replace('blcxx', 
                                                meta_row['node'])
frame_params = get_frame_params(full_path)
df = frame_params['df']
dt = frame_params['dt']

f_start = meta_row['f_start']
f_stop = meta_row['f_stop']

gen_info = []
for i in range(len(DRIFTS)):
    filename = DRIFTS['filename'].loc[i]
    full_path = f'{DATA_PREFIX}/{filename}'.replace('blcxx', 
                                                   meta_row['node'])
    
    frame = stg.Frame(full_path,
                      f_start=f_start,
                      f_stop=f_stop)
    gen_info.append([frame.noise_mean, 
                     frame.t_start, 
                     frame.tchans])
    
    
def base_cadence():
    c = stg.Cadence()
    for i in range(len(DRIFTS)):
        noise_mean, t_start, tchans = gen_info[i]
        frame = stg.Frame(fchans=512,
                          tchans=tchans,
                          df=df,
                          dt=dt,
                          fch1=f_start*u.MHz,
                          ascending=True,
                          t_start=t_start)
        frame.add_noise(noise_mean)
        c.append(frame)
    return c


def target_frame():
    """
    Make target candidate lookalike.
    """
    snr = np.random.uniform(15, 100)
    start_index = np.random.randint(0, 512)
    if start_index < 128:
        direction = 1
    elif start_index >= 384:
        direction = -1
    else:
        direction = np.random.choice([-1, 1])
    
    c = base_cadence()
    c.add_signal(stg.sine_path(f_start=c[0].get_frequency(index=start_index),
                               drift_rate=0,
                               period=1/(1.13492353e-05),
                               amplitude=direction*444),
                              stg.constant_t_profile(level=c[0].get_intensity(snr=snr)),
                              stg.sinc2_f_profile(width=c[0].df),
                              stg.constant_bp_profile(level=1))
    final_frame = c.consolidate()
    frame_info = {
        'label': 1,
        'class': 'target',
        'snr': snr,
        'start_index': start_index,
        'direction': direction
    }
    return final_frame, frame_info


def noise_frame():
    c = base_cadence()
    final_frame = c.consolidate()
    frame_info = {
        'label': 0,
        'class': 'noise',
        'snr': None,
        'start_index': None,
        'direction': None
    }
    return final_frame, frame_info


def random_frame():
    """
    Make false, randomized signal.
    1/4 chance no signal in pointing, random drift rates and positions.
    """
    c = base_cadence()

    for frame in c:
        snr = np.random.uniform(25, 100)
        if np.random.rand() < 0.25:
            snr = 0
            
        start_index = np.random.randint(0, 512)
        drift_rate = np.random.uniform(-0.05, 0.05)
    
        frame.add_signal(stg.constant_path(f_start=frame.get_frequency(index=start_index),
                                           drift_rate=drift_rate),
                         stg.constant_t_profile(level=frame.get_intensity(snr=snr)),
                         stg.sinc2_f_profile(width=frame.df),
                         stg.constant_bp_profile(level=1))
    final_frame = c.consolidate()
    frame_info = {
        'label': 0,
        'class': 'random',
        'snr': None,
        'start_index': None,
        'direction': None
    }
    return final_frame, frame_info

def vertical_frame():
    """
    Make signal with RFI frequency variation, nearly vertical.
    """
    c = base_cadence()

    snr = np.random.uniform(25, 100)
    start_index = np.random.randint(0, 512)
    drift_rate = np.random.uniform(-0.001, 0.001)
    
    c.add_signal(stg.simple_rfi_path(f_start=frame.get_frequency(index=start_index),
                                       drift_rate=drift_rate,
                                     spread=1*frame.df,
                                     spread_type='normal'),
                     stg.constant_t_profile(level=frame.get_intensity(snr=snr)),
                     stg.sinc2_f_profile(width=frame.df),
                     stg.constant_bp_profile(level=1))
    final_frame = c.consolidate()
    frame_info = {
        'label': 0,
        'class': 'vertical',
        'snr': None,
        'start_index': None,
        'direction': None
    }
    return final_frame, frame_info


def make_frame():
    t_val = np.random.rand()
    if t_val < 0.1:
        return noise_frame()
    elif t_val < 0.4:
        return random_frame()
    elif t_val < 0.7:
        return vertical_frame()
    else:
        return target_frame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('n_train', type=int)
    parser.add_argument('n_test', type=int)
    args = parser.parse_args()
    
    # Convert args to dictionary
    params = vars(args)
    
    data_path = params['data_path']
    n_train = params['n_train']
    n_test = params['n_test']
    
    splits = [('train', n_train), ('test', n_test)]
    for split_name, n_split in splits:
        mkdir(f'{data_path}/{split_name}/')
        
        info_list = []
        for j in tqdm.tqdm(range(n_split)):
            frame, frame_info = make_frame()
            frame.save_npy(f'{data_path}/{split_name}/{j:06d}.npy')                   
            
            # Directly link filename to the frame info for convenience
            frame_info['filename'] = f'{j:06d}.npy'
            info_list.append(frame_info)

            print(f'Saved frame {j}')
            
        # Save out labels using pandas
        df = pd.DataFrame(info_list)
        df.to_csv(f'{data_path}/{split_name}/labels.csv', index=False)
        
        
if __name__ == "__main__":
    
    main()