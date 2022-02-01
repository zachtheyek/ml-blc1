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
import tqdm
import pathlib

from params import DATA_PREFIX, META, DRIFTS, CENTERS
import extract_signals as es


def plot_sequences(candidate=0, f_start=982.0015, f_stop=982.0035):
    """
    Collect real observations of blc1 candidate, and produce synthetic copies using extracted
    properties. Plot real and synthetic cadences side-by-side for easy comparison.
    """
    # Collect real files
    file_list = []
    height_ratios = []
    for i, row in es.DRIFTS.iterrows():
        filename = row['filename']
        full_path = f'{es.DATA_PREFIX}/{filename}'.replace('blcxx', 
                                                           es.META.loc[candidate]['node'])
        file_list.append(full_path)

        container = bl.Waterfall(full_path, load_data=False).container
        tchans = container.file_shape[0]
        df = abs(container.header['foff']) * 1e6
        dt = container.header['tsamp']
        height_ratios.append(tchans * dt)
        
    F_min, F_max = bl.stax.sort2(f_start, f_stop)

    # Compute the midpoint frequency for the x-axis.
    F_mid = np.abs(F_min + F_max) / 2
    
    # Create plot grid
    n_plots = len(file_list)
    fig_array, axs = plt.subplots(nrows=n_plots,
                                  ncols=2,
                             sharex=True,
                             sharey=False, 
                             dpi=200,
                             figsize=(24, 2*n_plots),
                             gridspec_kw={"height_ratios" : height_ratios})
    
    
    # Iterate over data for min/max values, real
    for i, filename in enumerate(file_list):
        wf = bl.Waterfall(filename, f_start=F_min, f_stop=F_max)
        data = stg.Frame(wf).data
        if i == 0:
            px_min = np.min(data)
            px_max = np.max(data)
        else:
            if px_min > np.min(data):
                px_min = np.min(data)
            if px_max < np.max(data):
                px_max = np.max(data)
    
    # Produce and collect synthetic files
    all_y = np.load('intensity_data.npy', allow_pickle=True)
    candidate_data = all_y[candidate]['signal']
    ts_info = np.load('ts_info.npy', allow_pickle=True).item()

    fchans = wf.data.shape[2]
    synthetic_file_list = []
    for i, ts in enumerate(ts_info['ts_list']):
        ys = candidate_data[i]['scaled_ys']
        noise_mean = candidate_data[i]['noise_mean']

        frame = stg.Frame(fchans=fchans,
                          tchans=len(ts),
                          df=df,
                          dt=dt,
                          fch1=F_min*u.MHz,
                          ascending=True)
        
        if noise_mean is None:
            filename = es.DRIFTS['filename'].loc[i]
            full_path = f'{es.DATA_PREFIX}/{filename}'.replace('blcxx', 
                                                               es.META.loc[candidate]['node'])
            fr = stg.Frame(bl.Waterfall(full_path, f_start=F_min, f_stop=F_max))
            noise_mean = fr.noise_mean
        frame.add_noise(noise_mean)

        if ys is not None:
            # Make sure values are nonnegative
            frame.add_signal(stg.constant_path(f_start=candidate_data[i]['center_freq']*u.MHz,
                                                drift_rate=candidate_data[i]['drift_rate']),
                             frame.noise_std * np.maximum(ys, 0) / 1,
                             stg.sinc2_f_profile(width=1.*frame.df),
                             stg.constant_bp_profile(level=1))
        fn = f"data/synth_{es.META.loc[candidate]['lookalike']}_{i}.fil"
        frame.save_fil(fn)
        synthetic_file_list.append(fn)
        
    # Iterate over data for min/max values, synthetic
    for i, filename in enumerate(synthetic_file_list):
        wf = bl.Waterfall(filename, f_start=F_min, f_stop=F_max)
        data = stg.Frame(wf).data
        
        if px_min > np.min(data):
            px_min = np.min(data)
        if px_max < np.max(data):
            px_max = np.max(data)
    
    # Plot real observations
    for i, filename in enumerate(file_list):
        plt.sca(axs[i, 0])
        if i == 0:
            plt.title(f"Real: {es.META.loc[candidate]['lookalike']}")
            
        wf = bl.Waterfall(filename, f_start=F_min, f_stop=F_max)
#         last_plot = stg.Frame(wf).plot(use_db=True, cb=False)
        last_plot = plot_waterfall(wf, vmin=stg.db(px_min), vmax=stg.db(px_max))
        
    factor = 1e6
    units = "Hz"
    xloc = np.linspace(F_min, F_max, 5)
    xticks = [round(loc_freq) for loc_freq in (xloc - F_mid) * factor]
    if np.max(xticks) > 1000:
        xticks = [xt / 1000 for xt in xticks]
        units = "kHz"
    plt.xticks(xloc, xticks)
    plt.xlabel("Relative Frequency [%s] from %f MHz" % (units, F_mid))
    plt.ylabel("Time [s]")
    
    # Plot synthetic observations
    for i, filename in enumerate(synthetic_file_list):
        plt.sca(axs[i, 1])
        if i == 0:
            plt.title(f"Synthetic: {es.META.loc[candidate]['lookalike']}")
            
        wf = bl.Waterfall(filename, f_start=F_min, f_stop=F_max)
        plot_waterfall(wf, vmin=stg.db(px_min), vmax=stg.db(px_max))

        
    plt.xticks(xloc, xticks)
    plt.xlabel("Relative Frequency [%s] from %f MHz" % (units, F_mid))
    plt.ylabel("Time [s]")
    
    # Adjust plots
    plt.subplots_adjust(hspace=0.02, wspace=0.1)    
        
    # Add colorbar.
    cax = fig_array.add_axes([0.94, 0.11, 0.03, 0.77])
    fig_array.colorbar(last_plot, cax=cax, label="Power (dB)")
    

    
def generate_comparison_plots():
    """
    Produce and save all comparison plots, into 'data' folder.
    """
    pathlib.Path('data/real_synth_img/').mkdir(parents=True, exist_ok=True)
    
    for i, meta_row in tqdm.tqdm(META.iterrows()):
        plot_sequences(candidate=i, f_start=meta_row['f_start'], f_stop=meta_row['f_stop'])
        plt.savefig(f"data/real_synth_img/{i:02d}_real_synth_{meta_row['lookalike']}.pdf", bbox_inches='tight')
        plt.close("all")


def plot_waterfall(wf, f_start=None, f_stop=None, **kwargs):
    """
    Version of blimpy.stax plot_waterfall method without normalization.
    """
    MAX_IMSHOW_POINTS = (4096, 1268)
    from blimpy.utils import rebin
    
    # Load in the data from fil
    plot_f, plot_data = wf.grab_data(f_start=f_start, f_stop=f_stop)

    # Make sure waterfall plot is under 4k*4k
    dec_fac_x, dec_fac_y = 1, 1

    # rebinning data to plot correctly with fewer points
    try:
        if plot_data.shape[0] > MAX_IMSHOW_POINTS[0]:
            dec_fac_x = plot_data.shape[0] / MAX_IMSHOW_POINTS[0]
        if plot_data.shape[1] > MAX_IMSHOW_POINTS[1]:
            dec_fac_y =  int(np.ceil(plot_data.shape[1] /  MAX_IMSHOW_POINTS[1]))
        plot_data = rebin(plot_data, dec_fac_x, dec_fac_y)
    except Exception as ex:
        print("\n*** Oops, grab_data returned plot_data.shape={}, plot_f.shape={}"
              .format(plot_data.shape, plot_f.shape))
        print("Waterfall info for {}:".format(wf.filename))
        wf.info()
        raise ValueError("*** Something is wrong with the grab_data output!") from ex

    # determine extent of the plotting panel for imshow
    nints = plot_data.shape[0]
    bottom = (nints - 1) * wf.header["tsamp"] # in seconds
    extent=(plot_f[0], # left
            plot_f[-1], # right
            bottom, # bottom
            0.0) # top

    # plot and scale intensity (log vs. linear)
    kwargs["cmap"] = kwargs.get("cmap", "viridis")
    plot_data = 10.0 * np.log10(plot_data)

    # display the waterfall plot
    this_plot = plt.imshow(plot_data,
        aspect="auto",
        rasterized=True,
        interpolation="nearest",
        extent=extent,
        **kwargs
    )

    # add source name
    ax = plt.gca()
    plt.text(0.03, 0.8, wf.header["source_name"], transform=ax.transAxes, bbox=dict(facecolor="white"))

    return this_plot
        
        
if __name__ == '__main__':
    generate_comparison_plots()