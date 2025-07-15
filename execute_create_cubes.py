# %% [markdown]
# # Analysis of detection algorithms for SIRENA
# 
# Check results of possible input parameters
# 
# * samplesUp (fixed): number of consecutive samples in the derivative above the threshold
# * samplesDown (fixed): number of consecutive samples in the derivative below the threshold to start triggering again
# * threshold (fixed): value to be crossed by the derivative
# 
# * window: size (samples) of the window to calculate average derivative and do a subtraction   
#   Ex. window = 3  :  
#   ```
#   deriv[i] => deriv[i] - mean(deriv[i-1], deriv[i-2], deriv[i-3])
#   ```
# 
# * offset: offset (samples) of the subtracting window   
#   Ex. window = 3 && offset = 2  :   
#   ```
#   deriv[i] => deriv[i] - mean(deriv[i-3], deriv[i-4], deriv[i-5])
#   ```
# 
# ## Procedure   
# 1) (*external*) XIFUSIM files with 100 pairs of pulses are simulated:    
# ```python
#     Eprimary = [0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9 , 10, 11 ,12]    
#     Esecondary = [0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9 , 10, 11 ,12]    
#     Separations = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30,35,40,45,50,60,70,80,90,100,126]    
# ```
# 2) (*external*) SIRENA reconstructed (xifusim) files using combinations of window and offset:    
# ```python
#     samplesUp = 3
#     samplesDown = 2   
#     threshold = 6
#     window = [0, 1, 2, 3, 4, 5, 6, 10, 15, 20]
#     offset = [0, 1, 2, 3, 4, 5, 6]
#     The combination window=0 & offset=0 corresponds to the traditional method (no derivative subtraction)   
# ```
# 3) (*external*) For each window/offset combination a pickle object (file) is created with the following information:   
# ```python
#     | separation | energy1 | energy2 | window | offset | ndetected | nfake |
#     
# ```
# 4) Analysis in this notebook:    
#    
#    a) For each window/offset:
#     * Read pickle file   
#     * (Optionally) create a FITS data cube of number of detected photons: AXIS1-ENERGY1, AXIS2-ENERGY2, AXIS3-separations   
#     * Save number of photons lost: numpy[separations, window, offset]  (**Warning**: indexing in numpy and FITS is reversed)  
#     * Plot an image of E2 vs E1 for a given separation (data cube slice)   
# 
#    b) Create a mosaic of images with all the windows and offsets    
#    c) Write a FITS cube with number of lost photons: AXIS1-window, AXIS2-offset, AXIS3-separations    
#    d) Collapse the cube in separations and take mean value: plot image of lost photons (offset vs window)    
# 

# %% [markdown]
# 
# ***
# > **NOTE**:   
# > to convert this notebook into a Python script (for Slurm), just "*Export as*" -> Python and comment the line: `%matplotlib widget`
# 
# ***

# %% [markdown]
# ## Import modules

# %%
# import python modules
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
# import for pickling
import pickle
from astropy import table
from astropy.io import fits
from auxiliary import create_fits_cube
import matplotlib.colors as mcolors
import ipywidgets as widgets
#%matplotlib widget
#%matplotlib qt
#%config InlineBackend.figure_format = 'retina'

# %% [markdown]
# ## Running Jupyter or Python script?   
# * It tries to call get_ipython() (only available in IPython environments, like Jupyter).   
# * If the shell class name is "ZMQInteractiveShell", it confirms that you're in a Jupyter notebook (or JupyterLab).      
# * If it's a regular Python interpreter, the function returns False.

# %%
# detect whether running in Jupyter Notebook or as a script
def is_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell == 'ZMQInteractiveShell'
    except (NameError, ImportError):
        return False

# %%
# parameter handling
def get_parameters():
    """
    Get parameters for pairs detection analysis.
    If running in a Jupyter Notebook, use default parameters.
    If running as a script (e.g., SLURM), parse command line arguments.
    """
    global th, sUp, sDown, windows, offsets, relevant_separations, xifu_config, create_cubes, sep_for_plot_mosaic

    # Check if running in a Jupyter Notebook or as a script 
    if is_notebook():
        # Default parameters for interactive use
        print("Running in notebook mode for pairs detection analysis")
        return {
            "threshold": 6.0, # threshold for detection
            "samplesUp": 3,  # samples up for detection
            "samplesDown": 2, # samples down for detection
            "windows": [0, 1, 2, 3, 4, 5, 10, 15, 20], # subtraction derivative window for detection
            "offsets": [0, 1, 2, 3, 4, 5],  # offset for subtraction window
            "relevant_separations": [8, 20, 50, 126, 317, 797], # relevant separations for the analysis
            "config_version": 'v5_20250621',  # XIFU configuration
            "create_cubes": False,  # flag to create cubes
            "sep_for_plot_mosaic": 797 # samples separation for plotting the mosaic of slices of the data cube (if negative, no plotting)
        }
    else:
        # Parameters from command line (e.g., for SLURM)
        parser = argparse.ArgumentParser(
            description='Execute the python script for pairs detection analysis',
            prog='create_cubes.py')
        parser.add_argument('--windows', required=False, type=int,
                            nargs='*', default=[0, 1, 2, 3, 4, 5, 10, 15, 20],
                            help='Subtraction derivative window for detection')
        parser.add_argument('--offsets', required=False, type=int,
                            nargs='*', default=[0, 1, 2, 3, 4, 5],
                            help='Offset for subtraction window')           
        parser.add_argument('--threshold', required=False, type=float, default=0.5,
                            help='Threshold for detection')
        parser.add_argument('--samplesUp', required=False, type=int, default=2,
                            help='Samples up for detection')
        parser.add_argument('--samplesDown', required=False, type=int, default=2,
                            help='Samples down for detection')
        parser.add_argument('--config_version', required=False, type=str, default='v5_20250621',
                            help='XIFU configuration version')
        parser.add_argument('--create_cubes', action='store_true',
                            help='Flag to create cubes')
        parser.add_argument('--relevant_separations', required=False, type=int,
                            nargs='*', default=[8, 20, 50, 126, 317, 797],
                            help='Relevant separations for the analysis')
        parser.add_argument('--sep_for_plot_mosaic', required=False, type=int, default=-1,
                            help='Samples separation for plotting the mosaic of slices of the data cube (if negative, no plotting)')
        
        args = parser.parse_args()
      
        return vars(args)

# %% [markdown]
# ## Get parameters

# %%
params = get_parameters()
th = params['threshold']
sUp = params['samplesUp']
sDown = params['samplesDown']
windows = params['windows']
offsets = params['offsets']
xifu_config = params['config_version']
create_cubes = params['create_cubes']
relevant_separations = params['relevant_separations']
sep_for_plot_mosaic = params['sep_for_plot_mosaic']

# %% [markdown]
# ### Secondary parameters

# %%
min_detected = 100
max_detected = 200

# %%
# print parameters
print(f"Parameters: th={th}, sUp={sUp}, sDown={sDown}, windows={windows}, offsets={offsets}, xifu_config={xifu_config}, create_cubes={create_cubes}, relevant_separations={relevant_separations}, sep_for_plot_mosaic={sep_for_plot_mosaic}")

# %% [markdown]
# ## Create FITS cubes (for DS9) from pickle files   
# 1. Data reconstruction results are saved in pickle files for each combination of window and offset    
# 2. Create Data cubes:    
# 
#  | NAXIS3(sep)   
#  |        
#  |____ NAXIS1(e1)     
#  /         
# NAXIS2(e2)
# 
# 3. Save nlost_pulses [separation, window, offset]    
# 4. Plot E2 vs E1 mosaic of images   

# %%
first_read = True  # flag to indicate if we are reading the first pickle file
separations = None  # to store unique separations
energies1 = None  # to store unique energies1
energies2 = None  # to store unique energies2
nrows = len(offsets)  # number of rows for the mosaic plot
ncols = len(windows)  # number of columns for the mosaic plot

# initialize a numpy array 3-D (sep, offset, window) for nlost_pulses
nlost_pulses = np.zeros((len(relevant_separations), len(offsets), len(windows)), dtype=int)

if sep_for_plot_mosaic > 0:
    sep = sep_for_plot_mosaic  # select the first separation for the mosaic
    # create a mosaic figure (with squared plots) of the same slice in different data-cubes for each window and offset
    fig_mosaic, ax_mosaic = plt.subplots(nrows, ncols, figsize=(18, 12), sharex=True, sharey=True)
    fig_mosaic.suptitle(f'Mosaic of Detected Events Cube Slices by separation of the 2 pulses (config: {xifu_config=}, {th=}, {sUp=}, {sDown=}, {sep=})', fontsize=10)
    
for io in range(len(offsets)):
    for iw in range(len(windows)):    
        # get offset in inverse order for plotting reasons: mosaic plots would otherwise show the first offset at the top
        io_plot = len(offsets) - 1 - io
        off = offsets[io_plot]  # use the current offset for plotting
        #off = offsets[io]  # use the current offset
        win = windows[iw]
        #print(f"Window: {win}, Offset: {off}")
        if win == 0: 
            off = 0
        pickle_file = f'analysis_pairs/detectedFakes_win{win}_off{off}.pkl'
        if win == 0 and off > 0:
            #Skipping window {win} with offset {off} as it is not applicable.
            continue
        # read the data from the pickle file
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
            if win == 0 and off == 0:
                data_table = table.Table(rows=data, names=('separation', 'energy1', 'energy2', 'samplesDown', 'samplesUp', 'threshold', 'ndetected', 'nfake'))
                data_filtered = data_table[(data_table['threshold'] == th) & (data_table['samplesUp'] == sUp) & (data_table['samplesDown'] == sDown)]
            else:
                data_table = table.Table(rows=data, names=('separation', 'energy1', 'energy2', 'window', 'offset', 'ndetected', 'nfake')) 
                data_filtered = data_table.copy()
        #print(f"Filtered data: {data_filtered}")
        if first_read:
            separations = np.unique(data_filtered['separation'])
            energies1 = np.unique(data_filtered['energy1'])
            energies2 = np.unique(data_filtered['energy2'])
            nsimulated = 200 * len(energies1) * len(energies2)  # total number of simulated events
            e1_labs = [f'{e:.0f}' for e in energies1]
            e1_labs[0] = f"{energies1[0]:.1f}"  # first label with one decimal
            e1_labs[1] = f"{energies1[1]:.1f}"  # second label with one decimal
            e2_labs = [f'{e:.0f}' for e in energies2]
            e2_labs[0] = f"{energies2[0]:.1f}"  # first label with one decimal
            e2_labs[1] = f"{energies2[1]:.1f}"  # second label with one decimal
            first_read = False

        #look for rows where nfake > 0
        data_filtered_nfake = data_filtered[data_filtered['nfake'] > 0]
        if len(data_filtered_nfake) > 0:
            print(f"Filtered data with nfake > 0: {data_filtered_nfake}") 

        # create a data cube for each pickle file (and optionally save it to a FITS file)
        data_cube_file=""
        if create_cubes:
            data_cube_file = f"analysis_pairs/detected_events_cube_win{win}_off{off}.fits"
        data_cube = create_fits_cube(data_filtered, data_cube_file=data_cube_file)
        
        # store nlost pulses in the 3D array
        for isep, sep in enumerate(relevant_separations):
            data_slice = data_cube[isep, :, :]
            # calculate the number of detected events in the slice
            ndetected_slice = np.sum(data_slice)
            nlost_slice = ndetected_slice - nsimulated
            # use io reversed index for FITS cube so that the first offset is at the bottom of the plot
            nlost_pulses[isep, io_plot, iw] = nlost_slice
                

        if sep_for_plot_mosaic > 0:
            # plot mosaic of slices of the data cube
            # --------------------------------------
            
            # create a normalization for the color map
            norm = mcolors.Normalize(vmin=min_detected, vmax=max_detected)
            cmap = plt.get_cmap('viridis')
            # create a new figure for each window and offset using the slice for the separation_index
            separation_index = np.where(separations == sep_for_plot_mosaic)[0][0]
            data_slice = data_cube[separation_index, :, :]
            # calculate the number of detected events in the slice
            ndetected_slice = np.sum(data_slice)
            nlost_slice = ndetected_slice - nsimulated
            
            im = ax_mosaic[io, iw].imshow(data_slice, aspect='auto', origin='lower', cmap=cmap, norm=norm, interpolation='nearest')
            ax_mosaic[io, iw].set_title(f'Window: {windows[iw]}, Offset: {off}\n N. lost={nlost_slice}', fontsize=8)
            ax_mosaic[io, iw].set_xlabel('Energy primary (keV)', fontsize=8)
            ax_mosaic[io, iw].set_ylabel('Energy secondary (keV)', fontsize=8)
            ax_mosaic[io, iw].set_xticks(np.arange(len(energies1)))
            ax_mosaic[io, iw].set_xticklabels(e1_labs, rotation=45, fontsize=8)
            ax_mosaic[io, iw].set_yticks(np.arange(len(energies2)))
            ax_mosaic[io, iw].set_yticklabels(e2_labs, fontsize=8)    
            ax_mosaic[io, iw].set_aspect('equal')
            # add color bar to each subplot
            cbar = fig_mosaic.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax_mosaic[io, iw], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)  # adjust color bar tick label size    
if sep_for_plot_mosaic > 0:
    # adjust layout
    plt.tight_layout()
    plt.show()
    # save the mosaic figure (png and PDF)
    fig_mosaic.savefig(f'analysis_pairs/mosaic_detected_events_cube_slices_windows_offsets.png', dpi=300, bbox_inches='tight')
    fig_mosaic.savefig(f'analysis_pairs/mosaic_detected_events_cube_slices_windows_offsets.pdf', bbox_inches='tight')
    


# %% [markdown]
# ## Create cube of nlost_pulses
# 
#  | NAXIS3(sep)   
#  |        
#  |____ NAXIS1(window)     
#  /         
# NAXIS2(offset)
# 

# %%
# save a FITS cube with the nlost_pulses[sep, offset, window]
hdu = fits.PrimaryHDU(nlost_pulses)
hdu.header['SEPS'] = ', '.join(map(str, relevant_separations)) # axis3
hdu.header['OFFSETS'] = ', '.join(map(str, offsets)) #axis2
hdu.header['WINDOWS'] = ', '.join(map(str, windows)) # axis1
# set units for the axes: samples
hdu.header['CUNIT3'] = 'samples'
hdu.header['CUNIT2'] = 'samples'
hdu.header['CUNIT1'] = 'samples'
# save the FITS file
hdu.writeto('analysis_pairs/nlost_pulses_cube.fits', overwrite=True)
print("nlost_pulses cube saved to nlost_pulses_cube.fits")


# %% [markdown]
# ## Collapse nlost_pulses cube
# 
# 1. Take mean value along separations axis   
# 2. Plot collapsed image   

# %%
# collase cube in axis 0 (separation) to get the mean of lost photons and plot image 
nlost_pulses_collapsed = np.mean(nlost_pulses, axis=0)  # collapse the cube in axis 0 (separation)
print(nlost_pulses_collapsed.shape)  # should be (offsets, windows)
# create a new figure for the collapsed cube
fig_collapsed, ax_collapsed = plt.subplots(figsize=(8, 6))
# create a normalization for the color map
norm_collapsed = mcolors.Normalize()
cmap_collapsed = plt.get_cmap('viridis')
# plot the collapsed cube
im_collapsed = ax_collapsed.imshow(nlost_pulses_collapsed, origin='lower', cmap=cmap_collapsed, norm=norm_collapsed, interpolation='nearest')
ax_collapsed.set_title(f'Collapsed N. lost pulses (config: {xifu_config=}, {th=}, {sUp=}, {sDown=})', fontsize=10)
ax_collapsed.set_ylabel('Offset (samples)')
ax_collapsed.set_xlabel('Window (samples)')
ax_collapsed.set_yticks(np.arange(len(offsets)))
ax_collapsed.set_yticklabels(offsets, rotation=45, fontsize=8)
ax_collapsed.set_xticks(np.arange(len(windows)))
ax_collapsed.set_xticklabels(windows, fontsize=8)
# add color bar to the collapsed plot
cbar_collapsed = fig_collapsed.colorbar(plt.cm.ScalarMappable(norm=norm_collapsed, cmap=cmap_collapsed), ax=ax_collapsed, fraction=0.032, pad=0.04)
cbar_collapsed.ax.tick_params(labelsize=8)
plt.tight_layout()
plt.show()
# save the collapsed figure (png and PDF)
fig_collapsed.savefig(f'analysis_pairs/collapsed_nlost_pulses_cube_windows_offsets.png', dpi=300, bbox_inches='tight')
fig_collapsed.savefig(f'analysis_pairs/collapsed_nlost_pulses_cube_windows_offsets.pdf', bbox_inches='tight')

# %% [markdown]
# #### Prepare for a future "main" in the python script

# %%
"""
if __name__ == "__main__" and not is_notebook():
    main(**get_parameters())
"""

# %% [markdown]
# ### Some DUMB test

# %%
"""
########### TESTING PART ###########
# This part is for testing purposes only, to read the data from a pickle file and print
# the filtered data and an example row with specific values.
# It should not be part of the main code execution.

#read the data for window=20 and offset=0
win = 1
off = 0
e1 = 0.5
e2 = 12
pickle_file = f'detectedFakes_win{win}_off{off}.pkl'
# read the data from the pickle file
with open(pickle_file, 'rb') as f:
    data = pickle.load(f)
    data_table = table.Table(rows=data, names=('separation', 'energy1', 'energy2', 'window', 'offset', 'ndetected', 'nfake')) 
    data_filtered = data_table.copy()
#print(f"Filtered data: {data_filtered}")
# print row with seaparation=20, energy1=0.2, energy2=0.5
example_row = data_filtered[(data_filtered['separation'] == sep_for_plot_mosaic) & (data_filtered['energy1'] == e1) & (data_filtered['energy2'] == e2)]
#print(f"Example row: {example_row}")
# print separations, energies1, energies2
print(f"Separation: {separations}, Energies1: {energies1}, Energies2: {energies2}")
# print the number of detected events in the example
"""

# %%



