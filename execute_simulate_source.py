# %% [markdown]
# # XIFUSIM Simulation of a real source (model spectra)   
# 
# It simulates a real source in all the pixels of X-IFU with `xifusim.`   
# The parameters that define the simulation are (* for mandatory):   
#     - Source flux in mCrab*   
#     - Source model*   
#     - filter* applied to defocused case:  `thinOpt` // `thickOpt` // `nofilt` // `thinBe` // `thickBe`   
#     - focus: `infoc` or `defoc`. If not provided `defoc` is applied to flux > 0.5 flux_mcrab and `infoc` otherwise       
#     - Lower energy in flux band (`Emin`): 2. keV   
#     - Upper energy in flux band (`Emax`): 10. keV   
# 
# Other parameters (calculated/derived) are:   
#     - RA of source: 0.   
#     - Dec of source: 0.   
#     - Exposure time: x times the interval required to have 2 close pulses (assuming Poissonian)
# 

# %% [markdown]
# Possible models are:   
# 1. `CRAB`    
#     Power law spectrum: $\Gamma=2.05$      
#     Unabsorbed Flux(2-10keV): $21.6 \times 10^{-12} \rm{erg\,cm^{-2}\,s^{-1}}$     
#     Foregroung absorption: $N_H=2\times 10^{21} \rm{cm^{-2}}$    
#     XSPEC model: phabs*pegpwrlw   
# 2. `EXTEND` 

# %% [markdown]
# Simulation steps   
# 1. Read simulation parameters and derived parameters   
# 2. HEASOFT `xspec`: create xspec model file    
# 3. SIXTE `simputfile`: Create simput file with photons distribution    
# 4. SIXTE `sixtesim`: Run simulation to get   
#     4.1 ImpactList - piximpact file for ALL photons   
#     4.2 EventList - which photons (PH_ID) impact in each pixel of the detector (including background)     
#     4.3 PixImpactList: piximpact file for each pixel with impacts (PH_ID) (needed by xifusim)   
# 5. Get list of pixels w/ impacts. For each pixel with >1 impact:   
#     5.1. Check if there are "close" photons (otherwise skip xifusim simulation)   
#     5.2. Create a sub-piximpact file from the total piximpact file with the close photons   
#     5.3. `xifusim`: Do single-pixel (NOMUX) xifusim simulation     
#     5.3. `sirena`: reconstruct xifusim simulation   
#     5.4. Analyse reconstruction to check for missing photons   
#    

# %% [markdown]
# 
# ***
# > **NOTE**:   
# > to convert this notebook into a Python script (for Slurm), just "*Export as*" -> Python and comment the line: `%matplotlib widget`
# 
# ***

# %% [markdown]
# ## Import routines and read parameters

# %%
import os
from subprocess import run
import sys
import tempfile
import glob

import auxiliary as aux
from astropy.io import fits
from astropy.table import Table
#import heasoftpy as hsp
import numpy as np
import  pandas as pd
from xspec import Xset, Model, AllModels


# %% [markdown]
# ## Get parameters (check if running Jupyter notebook or Python script (Slurm))

# %% [markdown]
# ###  parameters   
# ```
# sim_number: simulation run number   
# flux_mcrab: erg/cm^2/s (1 mcrab=2.4E-11 erg/cm^2/s)   
# Emin: (keV) to define the energy range of the flux   
# Emax: (keV) to define the energy range of the flux   
# model: "crab"//TBD   
# filter:  thinOpt // thickOpt // nofilt // thinBe // thickBe    
# focus: '' (TBD from flux)   
# recons: 0 for no_reconstruction, 1 for do_reconstruction   
# verbose: 0 (silent) or 1 (chatty)
# ```

# %%
# parameter handling
def get_parameters():
    """
    Get parameters for pairs detection analysis.
    If running in a Jupyter Notebook, use default parameters.
    If running as a script (e.g., SLURM), parse command line arguments.
    """
    if aux.is_notebook():
        # Default parameters for interactive use
        print("Running in notebook mode for source simulation")
        return {
            "sim_number": 1,
            "flux_mcrab": 0.32,
            "Emin": 2.,
            "Emax": 10.,
            "model": "crab",
            "filter": "nofilt",
            "focus": '',
            "recons": 1,
            "verbose": 1,
            "config_version": "v5_20250621" #v3_20240917"
        }
    else:
        # Parse command line arguments for script execution
        print("Running in script mode for source simulation")
        import argparse
        parser = argparse.ArgumentParser(description="Source simulation parameters")
        parser.add_argument("--sim_number", type=int, default=1, help="Simulation number")
        parser.add_argument("--flux_mcrab", type=float, required=True, help="Flux in mCrab")
        parser.add_argument("--Emin", type=float, default=2.0, help="Minimum of the flux energy range in keV")
        parser.add_argument("--Emax", type=float, default=10.0, help="Maximum of the flux energy range in keV")
        parser.add_argument("--model", type=str, required=True, help="Model name for the simulation")
        parser.add_argument("--filter", type=str, default="nofilt", help="Filter name for the simulation",
                            choices=['thinOpt', 'thickOpt', 'nofilt', 'thinBe', 'thickBe'])
        parser.add_argument("--focus", type=str, default='', help="Focus to be used for the simulation. If not provided, option is selected automatically from the input flux",
                            choices=['infoc', 'defoc',''])
        parser.add_argument("--recons", type=int, default=0, help="Do Reconstruction? (1-Yes or 0-No)", choices=[0, 1])
        parser.add_argument("--verbose", type=int, default=0, help="Verbose level (0-1)", choices=[0, 1])
        parser.add_argument("--config_version", type=str, default="v5_20250621", help="XIFU Configuration version for xifusim",
                            choices=["v5_20250621", "v3_20240917"])
        
        args = parser.parse_args()
        return vars(args)

# %% [markdown]
# ### Get input parameters

# %%
params = get_parameters()
sim_number = params['sim_number']
flux_mcrab = params['flux_mcrab']
Emin = params['Emin']
Emax = params['Emax']
model = params['model']
filter = params['filter']
focus = params['focus']
recons = params['recons']
verbose = params['verbose']
config_version = params['config_version']

# %% [markdown]
# ### Read derived/extra parameters

# %%
tmpDir = tempfile.mkdtemp()
os.environ["PFILES"] = f"{tmpDir}:{os.environ['PFILES']}"
os.environ["HEADASNOQUERY"] = ""
os.environ["HEADASPROMPT"] = "/dev/null/"
SIXTE = os.environ["SIXTE"]
#if config_version == "v5_20250621":
#    xmldir = f"{SIXTE}/share/sixte/instruments/athena-xifu/internal_design_goal"
#elif config_version == "v3_20240917":
#    xmldir = f"{SIXTE}/share/sixte/instruments/athena-xifu/baseline"
xmldir = f"{SIXTE}/share/sixte/instruments/athena-xifu/baseline"
xml = f"{xmldir}/xifu_nofilt_infoc.xml"

# %%
SD=2 # SamplesDown for SIRENA detection
tH=6. # threshold for SIRENA detection
RA=0.
Dec=0.
sampling_rate=130210 #Hz
prebuff_xifusim=1500  #prebuffer samples for xifusim
pileup_dist=30 #samples for pileup
if config_version == "v3_20240917":
    close_dist_toxifusim = 100 #samples for close events to decide if xifusim simulation will be done
elif config_version == "v5_20250621":
    close_dist_toxifusim = 200 #samples for close events to decide if xifusim simulation will be done
else:
    raise ValueError(f"Unknown config_version: {config_version}")
secondary_samples = 1563
HR_samples = 8192
# 1 mCrab = 90 counts/s in the 2-10 keV band
flux = flux_mcrab * 2.4E-11
rate = flux_mcrab * 90 #counts/s
time_30samps = pileup_dist/sampling_rate #s
npile = 2 #counts 
npairs = 100 #pairs (close photons) to analyse pile-up
aux.vprint(f"rate={rate} ct/s, time_interval(30 samples)={time_30samps:.3e}s, npairs={npairs} pairs")
# get MAX number of PH_IDs in xifusim:
# open file $SIXTE/../xifusim/libxifusim/WriteFile.h ang get numeric value in PH_IDs
with open(f"{SIXTE}/../xifusim/libxifusim/WriteFile.h") as f:
    for line in f:
        if "const unsigned int max_phids" in line:
            MAX_PHIDS_xifusim = line.split()[5]
            # remove the last character (comma)
            MAX_PHIDS_xifusim = int(MAX_PHIDS_xifusim[:-1])
            break

aux.vprint(f"MAX_PHIDS_xifusim = {MAX_PHIDS_xifusim}")

# %% [markdown]
# ### Create folder structure for output

# %%
# create a new folder (if it does not exist) for the output using the flux as the name
if flux_mcrab < 0.01:
    flux_mcrab_str = f"{flux_mcrab:.2e}"
else:
    flux_mcrab_str = f"{flux_mcrab:.2f}"

fluxDir = f"{config_version}/flux{flux_mcrab_str}mcrab"

outDir = f"{fluxDir}/sim_{sim_number}"
outDirPath = f"{os.getcwd()}/{outDir}"
if not os.path.exists(outDirPath):
    os.makedirs(outDirPath)
log_file = f"{fluxDir}/sim_{sim_number}.log"
# if log_file exits, remove it
if os.path.exists(log_file):
    os.remove(log_file)


# %% [markdown]
# ### Set `sixtesim` XML file based on parameters (focus and filter)

# %%
## Set XML file based on parameters
# if 'focus' is not provided: get it automatically according to the FLUX
# Filter will only be applied to the defocussed case
if focus == '':
    if flux_mcrab <= 0.5:
        focus="infoc"
        xml_sixtesim = f"{xmldir}/xifu_nofilt_{focus}.xml"
    else:
        focus="defoc"
        xml_sixtesim = f"{xmldir}/xifu_{filter}_{focus}.xml"
elif focus == "defoc":
    xml_sixtesim = f"{xmldir}/xifu_{filter}_defoc.xml"
elif focus == "infoc":
    xml_sixtesim = f"{xmldir}/xifu_nofilt_infoc.xml"

if focus == "defoc":
    npairs = npairs*50 # get more close pulses to have better statistics
aux.vprint(f"Using XML file: {xml_sixtesim}")

# %% [markdown]
# ### get exposure time   
# required to get a good number of close photons (possible missing/bad-reconstructed)

# %%
#calculate exposure to get counts 
time_npairs = aux.time_to_observe_n_pairs(count_rate=rate, pairs_separation=time_30samps, npairs=npairs)
exposure = time_npairs*20.1
exposure_label = int(exposure)
aux.vprint(f"Using exposure time: {exposure:.2e}s")
aux.vprint(f"Using exposure time label: {exposure_label}")

# %%
# set string to name files based on input parameters
filestring_simput = f"./{fluxDir}/{model}_flux{flux_mcrab_str}_Emin{Emin:.0f}_Emax{Emax:.0f}_RA{RA}_Dec{Dec}"
filestring = f"./{outDir}/{model}_flux{flux_mcrab_str}_Emin{Emin:.0f}_Emax{Emax:.0f}_exp{exposure_label}_RA{RA}_Dec{Dec}_{filter}_{focus}"
print(filestring)

# %% [markdown]
# ## Create XSPEC model file   

# %%
# is spectral model file does not exist, create it
if model == "crab":
    xcm = f"{model}.xcm"
    if not os.path.exists(xcm):
        # Clear all models
        AllModels.clear()
        # define XSPEC parameters
        Xset.abund = "wilm"
        Xset.cosmo = "70 0. 0.73"
        Xset.xsect = "bcmc"
        mcmod = Model("phabs*pegpwrlw")
        mcmod.phabs.nH = 0.2
        mcmod.pegpwrlw.PhoIndex = 2.05
        mcmod.pegpwrlw.eMin = 2.
        mcmod.pegpwrlw.eMax = 10.
        mcmod.pegpwrlw.norm = 1.
        #retrieve the flux value
        AllModels.calcFlux(f"{Emin} {Emax}")
        model_flux = AllModels(1).flux[0]
        # calculate the new norm value
        new_norm = flux/model_flux
        mcmod.pegpwrlw.norm = new_norm
        # Save the model to the specified .xcm file path
        Xset.save(xcm)
        aux.vprint(f"Model saved to {xcm}")
else:
    aux.vprint("Model not implemented yet")
    sys.exit(1)
#mcmod.show()

# %% [markdown]
# ## Create simput file

# %%
# run simputfile to create the simput file
simputfile = f"{filestring_simput}_simput.fits"
if not os.path.exists(simputfile):
        comm = (f'simputfile Simput={simputfile} RA={RA} Dec={Dec} '
                f'srcFlux={flux} Emin={Emin} Emax={Emax} '
                f'XSPECFile={xcm} clobber=yes')
        aux.vprint(f"Running {comm}")
        # Run the command through the subprocess module
        output_simputfile = run(comm, shell=True, capture_output=True)
        #print(output_simputfile.stdout.decode())
        assert output_simputfile.returncode == 0, f"simputfile failed to run: {comm}"
        assert os.path.exists(simputfile), f"simputfile did not produce an output file"

# %% [markdown]
# ## Run sixtesim simulation: Create PIXIMPACT file 

# %%
evtfile = f"{filestring}_evt.fits"
photfile = f"{filestring}_photon.fits"
impfile = f"{filestring}_impact.fits"
if not os.path.exists(evtfile):
        aux.vprint(f"evtfile does not exist: {evtfile}")
if not os.path.exists(photfile):
        aux.vprint(f"photfile does not exist: {photfile}")
if not os.path.exists(impfile):
        aux.vprint(f"impfile does not exist: {impfile}")
        
if not os.path.exists(evtfile) or not os.path.exists(photfile) or not os.path.exists(impfile):    
        comm = (f'sixtesim PhotonList={photfile} Simput={simputfile} '
                f'ImpactList={impfile} EvtFile={evtfile} '
                f'XMLFile={xml_sixtesim} Background=yes RA={RA} Dec={Dec} ' 
                f'Exposure={exposure} clobber=yes')
        aux.vprint(comm)
        output_sixtesim = run(comm, shell=True, capture_output=True)
        assert output_sixtesim.returncode == 0, f"sixtesim failed to run"
        assert os.path.exists(evtfile), f"sixtesim did not produce an output file"
        assert os.path.exists(photfile), f"sixtesim did not produce an output file"
        assert os.path.exists(impfile), f"sixtesim did not produce an output file"

# %% [markdown]
# ## Do xifusim simulation  

# %% [markdown]
# ### Get list of pixels with counts produced by sixtesim 

# %%
#verbose=1
#read column PIXID from evtfile and save to a list of unique pixels
hdulist = fits.open(evtfile, mode='update')
evtdata = hdulist[1].data
pixels_with_impacts = np.unique(evtdata['PIXID']) #photons coming from sources (PH_ID>-1) and background (PH_ID=-1)
aux.vprint(f"Number of pixels with impacts: {len(pixels_with_impacts)}")
# all bkg impacts have same PH_ID identifier (-1)
# if more than one event with PH_ID=-1, change PH_ID of bkg impacts: if PH_ID==-1, change to consecutive negative number
if len(evtdata['PH_ID'][evtdata['PH_ID'] == -1]) > 1:
    phid_bkg = -1
    for i in range(len(evtdata)):
        if evtdata['PH_ID'][i] == -1:
            evtdata['PH_ID'][i] = phid_bkg
            phid_bkg -= 1
    #save changes to evtfile
hdulist.close()

# get pixels used and the events in each pixel
nimpacts_inpix = dict()
phid_impacts_inpix = dict()
for pixel in pixels_with_impacts:
    phid_impacts_inpix[pixel] = evtdata['PH_ID'][evtdata['PIXID'] == pixel]
    nimpacts_inpix[pixel] = len(phid_impacts_inpix[pixel])

#print number of impacts per pixel sorted by number of impacts
for key, value in sorted(nimpacts_inpix.items(), key=lambda item: item[1], reverse=True):
    aux.vprint(f"Pixel {key}: {value} impacts")


#print the PH_ID of impacts in pixels
for key, value in phid_impacts_inpix.items():
    aux.vprint(f"Pixel {key}: ")
    aux.vprint(f"      PH_ID:{value}")

# %% [markdown]
# ### Read sixtesim output data

# %%
# open sixtesim ImpactList and read data 
hdulist = fits.open(impfile)
impdata = hdulist[1].data
hdulist.close()
# open sixtesim EvtList and read data (has been modified to include different PH_ID for background photons)
hdulist = fits.open(evtfile)
evtdata = hdulist[1].data
hdulist.close()

# %% [markdown]
# ### For each pixel with impacts, do a xifusim simulation

# %%
aux.vprint(pixels_with_impacts)

# %% [markdown]
# #### Extract a piximpact file for each interesting pixel

# %%
# get list of already existing piximpactfiles
existing_piximpactfiles = glob.glob(f"{filestring}_pixel*_piximpact.fits")

for ipixel in pixels_with_impacts:
    aux.vprint(f"Checking existence of piximpact file for pixel {ipixel}")
    # create a subsample of the piximpact file selecting only those rows where PH_ID is in 
    # the list of the impacts in the pixel
    piximpactfile = f"{filestring}_pixel{ipixel}_piximpact.fits"
    
    # if file does not exist, create it
    #if not os.path.exists(piximpactfile):
    if piximpactfile not in existing_piximpactfiles:
        # copy src impacts from impact file and bkgs from event file
        # background are not in the impact list (only in the event file)
        phid_impacts_inpix_srcs = phid_impacts_inpix[ipixel][phid_impacts_inpix[ipixel] > 0]
        
        # if there are no source impacts in the pixel, create a new table with only the background
        if len(phid_impacts_inpix_srcs) > 0:  # src impacts
            # create a mask to select only the rows with the impacts in this pixel
            mask = np.isin(impdata['PH_ID'], phid_impacts_inpix_srcs)
            # create a new table with the selected rows
            newtable = Table(impdata[mask])
        else: # create table to include only background impacts
            aux.vprint(f"No source impacts in pixel {ipixel}")
            newtable = Table()
            # add columns TIME, SIGNAL, PH_ID and SRCID
            newtable['TIME'] = []
            newtable['ENERGY'] = []
            newtable['PH_ID'] = []
            newtable['SRC_ID'] = []
            
        # add the background impacts to the new table based on data in the event file
        # get indices of rows from event file where PH_ID < 0 (background)
        #   if 0.15keV < ENERGY < 0.2 keV => in EVT file: ENERGY = 0  => 
        # #    in pixelimpact will inherit ENERGY = 0 AND THEY WILL NOT BE SIMULATED BY XIFUSIM
        #   ** ARF/RMF threshold = 0.15keV but X-IFU/XML readout threshold=0.2keV **
        bkg_indices = np.where(evtdata['PH_ID'] < 0)[0]
        for ibkg in bkg_indices:
            if evtdata['PIXID'][ibkg] != ipixel:
                continue
            # create a new row in the new table
            newtable.add_row()
            # copy the bkg values from the event file (columns TIME, SIGNAL, PH_ID and SRCID) 
            # to the new table (to columns TIME, ENERGY, PH_ID and SRCID)
            # Check first if newtable is empty (no source impacts)
            if len(newtable) == 0:
                newtable['TIME'] = evtdata['TIME'][ibkg]
                newtable['ENERGY'] = evtdata['SIGNAL'][ibkg] # if 0.15 < ENERGY <0.2 => ENERGY=0 in piximapct => not xifusim
                newtable['PH_ID'] = evtdata['PH_ID'][ibkg]
                newtable['SRC_ID'] = evtdata['SRC_ID'][ibkg]
            else:
                newtable['TIME'][-1] = evtdata['TIME'][ibkg]
                newtable['ENERGY'][-1] = evtdata['SIGNAL'][ibkg] # if 0.15 < ENERGY <0.2 => ENERGY=0 in piximapct => not xifusim
                newtable['PH_ID'][-1] = evtdata['PH_ID'][ibkg]
                newtable['SRC_ID'][-1] = evtdata['SRC_ID'][ibkg]
        # sort newtable according to TIME
        newtable.sort('TIME')
            
        # add new columns X,Y,U,V, GRADE1, GRADE2, TOTALEN with the value 0 and PIXID with the value of ipixel
        newtable['X'] = 0.
        newtable['Y'] = 0.
        newtable['U'] = 0.
        newtable['V'] = 0.
        newtable['GRADE1'] = 0
        newtable['GRADE2'] = 0
        newtable['TOTALEN'] = 0
        newtable['PIXID'] = ipixel

        # name the new table 'PIXELIMPACT'
        newtable.meta['EXTNAME'] = 'PIXELIMPACT'
    
        # write the new table to a new FITS file
        newtable.write(piximpactfile, format='fits', overwrite=True)

        # print the name of the new file rewriting the output line
        aux.vprint(f"Created {piximpactfile} for pixel {ipixel}")


# %% [markdown]
# ### Get XML for xifusim simulations

# %%
# Find name of (unique) xml file in indir directory
#xml_xifusim = glob.glob(f"./config*.xml")
#if len(xml_xifusim) != 1:
#    raise FileNotFoundError(f"Error: expected 1 XML file but found {len(xml_xifusim)}")
#xml_xifusim = xml_xifusim[0]
if config_version == "v5_20250621":
    xml_xifusim = "config_xifu_50x30_v5_v20250621.xml"
elif config_version == "v3_20240917":
    xml_xifusim = "config_xifu_50x30_v3_20240917.xml"
aux.vprint(f"Using XIFUSIM XML file: {xml_xifusim}")

# %% [markdown]
# ### Run the xifusim simulation (with XML for single pixel)   
#     - simulate time between min and max TIME in piximpact   
#     - set PIXID to '1' for simulation   
#     - xifusim simulation    
#     - re-establish correct PIXID    
#     - get the number of phsims in the pixel: check different values in PH_ID column

# %%
#for each piximpact file, run xifusim
prebuffer = 1500
phsims_inpix = dict()
nphsims_inpix = dict()
skipped_photons_inpix = dict()
skipped_xifusim = []   

existing_xifusimfiles = glob.glob(f"{filestring}_pixel*_xifusim.fits")
existing_piximpactfiles_toxifusim = glob.glob(f"{filestring}_pixel*_piximpact_toxifusim.fits")
for ipixel in pixels_with_impacts:
    #if not ipixel == 21:  
    #    continue
    aux.vprint("====================================")
    aux.vprint(f"Selecting photons for pixel {ipixel}")
    aux.vprint("====================================")
    # get total number of simulated photons in the pixel
    piximpactfile = f"{filestring}_pixel{ipixel}_piximpact.fits"
    with fits.open(piximpactfile) as hdulist_piximpact:
        piximpactdata = hdulist_piximpact[1].data.copy()
    phsims_inpix[ipixel] =  piximpactdata['PH_ID']
    nphsims_inpix[ipixel] = len(phsims_inpix[ipixel])

    # if no more than 1 impact in the pixel, skip the simulation
    if nimpacts_inpix[ipixel] <= 1:
        aux.vprint(f"  Skipping simulation for pixel {ipixel} with {nimpacts_inpix[ipixel]} impact")
        skipped_xifusim.append(ipixel)
        continue
    
    # if there is no xifusim file for the pixel, create it
    xifusimfile = f"{filestring}_pixel{ipixel}_xifusim.fits"
    if xifusimfile not in existing_xifusimfiles:
    # if not os.path.exists(xifusimfile):
        # check if there are close events in the pixel
        aux.vprint(f"  Checking if needed simulation for pixel {ipixel} with {nimpacts_inpix[ipixel]} impact")
        PH_ID_toxifusim = []
        PH_ID_skipped = []
        time_diff = np.diff(piximpactdata['TIME'])
        close_photons = time_diff <= close_dist_toxifusim / sampling_rate # couples
        last_i = len(piximpactdata) - 1
        for i in range(len(piximpactdata)):
            keep_i = False
            # check proximity of photons
            # First photon: close partner? 
            if (i == 0 and close_photons[i]):
                keep_i = True
                aux.vprint(f"  First Photon {piximpactdata['PH_ID'][i]} is close to another photon next in time")
            # Last photon: close partner?
            elif (i == last_i and close_photons[i - 1]):
                keep_i = True
                aux.vprint(f"  Last Photon {piximpactdata['PH_ID'][i]} is close to another photon previous in time")
            # Middle photons: 
            # a) close partner to the previous or next photon? or
            elif (0 < i < last_i and (close_photons[i] or close_photons[i - 1])):
                 keep_i = True
                 aux.vprint(f"  Intermediate Photon {piximpactdata['PH_ID'][i]} is close to another photon previous or next in time")
                 aux.vprint(f"close_photons[i]: {close_photons[i]}, close_photons[i-1]: {close_photons[i-1]}")
            # b) previous photon (i-1) will be simulated and "i" is closer than HR_samples?
            elif (0 < i < last_i and piximpactdata['PH_ID'][i-1] in PH_ID_toxifusim and
                (piximpactdata['TIME'][i] - piximpactdata['TIME'][i - 1])*sampling_rate <= HR_samples):
                keep_i = True
                aux.vprint(f"  Intermediate Photon {piximpactdata['PH_ID'][i]} is closer to the previous photon in time than HR_samples")
            # c) next photon (i+1) closer than secondary_samples and is part of a couple (i+1, i+2)?
            elif (0 < i < last_i-1 and 
                  (piximpactdata['TIME'][i + 1] - piximpactdata['TIME'][i])*sampling_rate <= secondary_samples and
                  (piximpactdata['TIME'][i + 2] - piximpactdata['TIME'][i])*sampling_rate <= secondary_samples+close_dist_toxifusim):
                keep_i = True
                aux.vprint(f"  Intermediate Photon {piximpactdata['PH_ID'][i]} is closer to the next photon in time than secondary_samples")
            # if keep_i is True, add PH_ID to the list of PH_IDs to be used in xifusim
            if keep_i:
                # be sure that if "i" is to be simulated, the previous photon (i-1) is also in the list if closer than secondary_samples
                if i > 0 and (piximpactdata['TIME'][i] - piximpactdata['TIME'][i - 1])*sampling_rate <= secondary_samples:
                    if piximpactdata['PH_ID'][i - 1] not in PH_ID_toxifusim:
                        PH_ID_toxifusim.append(piximpactdata['PH_ID'][i - 1])
                        aux.vprint(f"  Photon {piximpactdata['PH_ID'][i - 1]} will be simulated in xifusim")
                PH_ID_toxifusim.append(piximpactdata['PH_ID'][i])
                aux.vprint(f"  Photon {piximpactdata['PH_ID'][i]} will be simulated in xifusim")
            else:
                PH_ID_skipped.append(piximpactdata['PH_ID'][i])

        skipped_photons_inpix[ipixel] = PH_ID_skipped
        #aux.vprint(f"{len(PH_ID_toxifusim)} Photons for xifusim: {PH_ID_toxifusim}")
        #aux.vprint(f"{len(PH_ID_skipped)} Skipped photons in pixel {ipixel}: {skipped_photons_inpix[ipixel]}")
        aux.vprint(f"    {len(PH_ID_skipped)} Skipped photons in pixel {ipixel}")
        aux.vprint(f"    {len(PH_ID_toxifusim)} Photons for xifusim")

        # if no photons closer than close_dist_toxifusim/sampling_rate (secs) in the pixel, skip the simulation 
        if len(PH_ID_toxifusim) == 0:
            aux.vprint(f"    No photons closer than {close_dist_toxifusim} samples in pixel {ipixel}: skipping simulation")
            skipped_xifusim.append(ipixel)
            continue
        
        # create a new (reduced) FITS piximpact file keeping only those PH_ID ...
        # ... where TIME of photons is closer than close_dist_toxifusim (samples)
        piximpactfile_toxifusim = f"{filestring}_pixel{ipixel}_piximpact_toxifusim.fits"
        # if piximpactfile_toxifusim does not exist, create it
        #if not os.path.exists(piximpactfile_toxifusim):
        if piximpactfile_toxifusim not in existing_piximpactfiles_toxifusim:
            # create a new table with the selected rows
            #mask = np.isin(piximpactdata['PH_ID'], PH_ID_toxifusim[PH_ID_toxifusim != 0])
            mask = np.isin(piximpactdata['PH_ID'], PH_ID_toxifusim)
            newtable = Table(piximpactdata[mask])
            # replace PIXID column with value '1' (xifusim requirement)
            newtable['PIXID'] = 1
            # name the new table 'PIXELIMPACT'
            newtable.meta['EXTNAME'] = 'PIXELIMPACT'
            # write the new table to a new FITS file
            newtable.write(piximpactfile_toxifusim, format='fits', overwrite=True)
            aux.vprint(f"  Created {piximpactfile_toxifusim} for pixel {ipixel}")

        # use reduced piximpact file to run xifusim
        with fits.open(piximpactfile_toxifusim) as hdulist_toxifusim:
            piximpactdata_toxifusim = hdulist_toxifusim[1].data.copy()
        #calculate minimum and maximum time for impacts in the pixel
        mintime = np.min(piximpactdata_toxifusim['TIME'])
        maxtime = np.max(piximpactdata_toxifusim['TIME'])
        expos_init = mintime - 2.*prebuff_xifusim/sampling_rate
        expos_fin = maxtime + 0.1
        
        #create xifusim name based on input parameters    
        comm = (f'xifusim PixImpList={piximpactfile_toxifusim} Streamfile={xifusimfile} '
                f'tstart={expos_init} tstop={expos_fin} '
                f'trig_reclength=12700 '
                f'trig_n_pre={prebuff_xifusim} '
                f'trig_n_suppress=8192 '
                f'trig_maxreclength=100000 '
                f'XMLfilename={xml_xifusim} clobber=yes ')
        
        aux.vprint(f"  Doing simulation for pixel {ipixel} with {len(piximpactdata_toxifusim)} impacts ({nimpacts_inpix[ipixel]} TOTAL impacts)")
        aux.vprint(f"Running {comm}")
        
        output_xifusim = run(comm, shell=True, capture_output=True)
        assert output_xifusim.returncode == 0, f"xifusim failed to run: {comm}"
        #assert os.path.exists(xifusimfile), f"xifusim did not produce an output file"

        # re-write correct PIXID in the xifusim file
        with fits.open(xifusimfile, mode='update') as hdulist:
            xifusimdata = hdulist["TESRECORDS"].data
            xifusimdata['PIXID'] = ipixel
            hdulist.flush()

# %%
# if recons==0 stop notebok here
if recons == 0:
    aux.vprint("Reconstruction not requested, stopping here")
    sys.exit(0)

# %% [markdown]
# ## DO reconstruction with SIRENA

# %% [markdown]
# ### get LIBRARY adequate to XML file

# %%
# Find name of (unique) library file in indir directory compatible with xifusim XML file
lib_sirena = glob.glob(f"./*library*{config_version}*fits")
if len(lib_sirena) != 1:
    raise FileNotFoundError(f"Expected 1 LIBRARY file but found {len(lib_sirena)}")
lib_sirena = lib_sirena[0]
aux.vprint(f"Using LIBRARY file: {lib_sirena}")

# %% [markdown]
# ### run `tesrecons`

# %%
existing_reconsfiles = glob.glob(f"{filestring}_pixel*_sirena.fits")
for ipixel in pixels_with_impacts:
    # if pixel was skipped in xifusim, skip reconstruction
    if ipixel in skipped_xifusim:
        continue
    xifusimfile = f"{filestring}_pixel{ipixel}_xifusim.fits"
    reconsfile = f"{filestring}_pixel{ipixel}_sirena.fits"
    if reconsfile in existing_reconsfiles:
    #if os.path.exists(reconsfile):
        # check that all PH_ID rows have at least one zero element 
        # (previos versions of SIRENA had a limitation of 3 values even if there were more detections)
        # if not, run again
        with fits.open(reconsfile) as hdulist_recons:
            reconsdata = hdulist_recons[1].data.copy()
        # check if all PH_ID rows have at least one zero element
        # get PH_ID values from the reconsdata
        phid_recons = reconsdata['PH_ID']
        # get the number of unique PH_ID values
        unique_phid_recons = np.unique(phid_recons)
        # check if a '0' is present in the PH_ID values
        # if not, run again
        if 0 not in unique_phid_recons:
            aux.vprint(f"Reconstruction file {reconsfile} does not have a zero PH_ID: possibly not listing all detections")
            aux.vprint(f"Reconstruction file {reconsfile} will be removed")
            # remove the file
            os.remove(reconsfile)
        else:
            aux.vprint(f"Reconstruction file {reconsfile} already exists: skipping reconstruction")
            continue
    
    #if not os.path.exists(reconsfile):
    if reconsfile not in existing_reconsfiles:
        comm = (f"tesrecons Recordfile={xifusimfile} "
            f" TesEventFile={reconsfile}"
            f" LibraryFile={lib_sirena}"
            f" XMLFile={xml_xifusim}"
            f" clobber=yes"
            f" EnergyMethod=OPTFILT"
            f" OFStrategy=BYGRADE"
            f" filtEeV=6000"
            f" OFNoise=NSD"
            f" samplesDown={SD}"   #changed for new smoothed derivative (4 samples)
            f" samplesUp=3"
            f" threshold={tH}"
            #f" nSgms=3.5"
        )
        aux.vprint(f"Doing reconstruction for pixel {ipixel}", end='\r')
        #aux.vprint(f"Running {comm}")
        output_tesrecons = run(comm, shell=True, capture_output=True)
        assert output_tesrecons.returncode == 0, f"tesrecons failed to run:{comm}"
        assert os.path.exists(reconsfile), f"tesrecons did not produce an output file"
        

# %% [markdown]
# ### Try to IDENTIFY detected photons in SIRENA based on PIXIMPACT time

# %%
# Add a new column to SIRENA file with a possible ID of the photons (based on time in the piximact file)   
# open the SIRENA file and read the data
for ipixel in pixels_with_impacts:
    # if pixel was skipped in xifusim, skip reconstruction
    if ipixel in skipped_xifusim:
        continue
    reconsfile = f"{filestring}_pixel{ipixel}_sirena.fits"
    # open the piximpact file and read the data
    piximpactfile = f"{filestring}_pixel{ipixel}_piximpact.fits"
    with fits.open(piximpactfile) as hdulist_piximpact:
        piximpactdata = hdulist_piximpact[1].data
        piximpact_phids = piximpactdata['PH_ID'].copy()

    # if PROBPHID does not exist add a new column to the SIRENA file with the name 'PROBPHID'
    with fits.open(reconsfile, mode='update') as hdulist_recons:
        # read the data from the SIRENA file
        EVENTS_hdu = hdulist_recons[1]
        reconsdata = EVENTS_hdu.data
        cols = EVENTS_hdu.columns
        # check if the column already exists
        if not 'PROBPHID' in cols.names:
            aux.vprint(f"Adding PROBPHID column to {reconsfile}")
            col_PROBPHID = fits.Column(name='PROBPHID', format='J', unit='', array=np.zeros(len(reconsdata), dtype=int))
            new_cols = fits.ColDefs(cols + col_PROBPHID)
            # create a new BinTableHDU with the new columns
            new_hdu = fits.BinTableHDU.from_columns(new_cols)
            # name the new HDU 'EVENTS'
            new_hdu.name = 'EVENTS'
            # replace the old table with the new one
            hdulist_recons[1] = new_hdu
            hdulist_recons.flush()

    # Check if column PROBPHID does not have any '0' values (already populated): then skip to new pixel
    with fits.open(reconsfile) as hdulist_recons:
        reconsdata = hdulist_recons[1].data
        cols = hdulist_recons[1].columns
        probphid = reconsdata['PROBPHID']
        # check if all values are '0' (initialized but not updated)
        if np.all(probphid == 0):
            aux.vprint(f"PROBPHID column in {reconsfile} has '0' values: will be updated")
        else:
            aux.vprint(f"PROBPHID column in {reconsfile} already exists and it is populated: skipping update")
            continue


    # if '0' values are present: update the PROBPHID column with the best guess of the PH_ID
    with fits.open(reconsfile, mode='update') as hdulist_recons:
        reconsdata = hdulist_recons[1].data
        PH_ID = reconsdata['PH_ID']
        TIME = reconsdata['TIME']
        
        # for each event in the SIRENA file, check if it is close to an event in the piximpact file
        for irow in range(len(reconsdata)):
            ph_nonzero_sequence = PH_ID[irow][np.nonzero(PH_ID[irow])]
            number_of_ph_zeros = len(PH_ID[irow]) - len(ph_nonzero_sequence)
            # if number of values == 0 in PH_ID[irow] is 0, then max number of detections reached: some photons may be not registered in PH_ID
            # !! that's not correct anymore as SIRENA USES SAME PH_IDs than xifusim
            #if number_of_ph_zeros == 0:
            #    message = f"Maximum number of photons reached in row {irow} of {reconsfile}: some photons may have not been registered in PH_ID"
            #    print(f"*** ERROR: {message}")
            #    raise ValueError(f"{message}")
            
            time_sirena = TIME[irow]
            
            # if number of values != 0 in PH_ID[irow] is 1, then it is a single photon
            # initialize ph_id_i to -1 (not found)
            ph_id_i = -1
            if len(ph_nonzero_sequence) == 1:
                # single photon
                ph_id_i = PH_ID[irow][0]
            else:
                # more than one photon in the record: check corresponding time in impact file
                min_time_diff = float('inf')
                for ph_id in ph_nonzero_sequence:
                    # get the time of the photon in the impact file: same PH_ID 
                    index_match = np.where((piximpactdata['PH_ID'] == ph_id))
                    # check if there is a match
                    if len(index_match[0]) == 0:
                        raise ValueError(f"PH_ID {ph_id} not found in impact file ({piximpactfile})")
                    time_ph_piximpact = piximpactdata['TIME'][index_match]
                    time_diff = abs(time_ph_piximpact - time_sirena)
                    # check if the time difference is smaller than the minimum time difference
                    # if so, update the minimum time difference and the PH_ID
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        ph_id_i = ph_id
            #if ph_id_i == -1:
            #    aux.vprint(f"  No matching PH_ID found for row {irow} in SIRENA file {reconsfile}")
            #    raise ValueError(f"No matching PH_ID found for row {irow} in SIRENA file {reconsfile}")
            # Update PROBPHID with the PH_ID of the photon
            reconsdata['PROBPHID'][irow] = ph_id_i
        # save changes to the SIRENA file
        hdulist_recons.flush()
        #aux.vprint(f"Updated PROBPHID and GRADE2 column in {reconsfile}")
        aux.vprint(f"Updated PROBPHID a")

# %% [markdown]
# ### check missing or misreconstructed photons

# %%
# Detailed information about missing photons for the simulation (for each pixel)

# save info to pandas table
# Column 1: pixel ID (integer)
# Column 2: bad-reconstructed photons (integer)
# Column 3: non-reconstructed photons (integer)
# Column 4: grade1 of bad-reconstructed photons (integer)
# Column 5: grade2 of bad-reconstructed photons (integer)
info_table = pd.DataFrame(columns=['Pixel','Bad-reconstructed photons', 'Bad-reconstructed photons energy', 
                                    'Non-reconstructed photons', 'Non-reconstructed photons energies',
                                    'Non-reconstructed photons distances',
                                    'GRADE1 Bad-recons', 'GRADE2 Bad-recons'], dtype=object)
    

# %%

# reset skipped photons (just in case, photon separation criteria changed in xifusim simulations)
skipped_photons_inpix = dict()

# Check how many photons were reconstructed: compare the number of impacts in the pixel with the number of reconstructed photons
ph_non_recons_inpix = dict()
ph_bad_recons_inpix = dict()
nextra_recons_inpix = dict()
nbad_recons_inpix = dict()
nnon_recons_inpix = dict()
nrecons_inpix = dict()

nrecons_total = 0
nimpacts_total = 0
nbad_recons_total = 0
nnon_recons_total = 0

# open summary log file


for ipixel in pixels_with_impacts:
    #if not ipixel == 776:
    #    continue
    # if pixel was skipped in xifusim (1 impact or separated impacts)
    if ipixel in skipped_xifusim:
        # get the number of reconstructed photons
        nrecons_inpix[ipixel] = nphsims_inpix[ipixel] 
        nextra_recons_inpix[ipixel] = 0  
        ph_non_recons_inpix[ipixel] = np.array([])
        ph_bad_recons_inpix[ipixel] = np.array([])
        nnon_recons_inpix[ipixel] = 0
        nbad_recons_inpix[ipixel] = 0
    else:
        nextra_recons_inpix[ipixel] = 0
        missing_for_badrecons = dict()
        energies_missing_for_badrecons = dict()
        distances_missing_for_badrecons = dict()
        #read SIRENA file
        reconsfile = f"{filestring}_pixel{ipixel}_sirena.fits"
        hdulist = fits.open(reconsfile)
        reconsdata = hdulist[1].data
        PH_ID = reconsdata['PH_ID'].copy()
        GRADE1 = reconsdata['GRADE1'].copy()
        GRADE2 = reconsdata['GRADE2'].copy()
        PROBPHID = reconsdata['PROBPHID'].copy()
        hdulist.close()

        # read PIXIMPACT file (for TIME column)
        hdulist = fits.open(f"{filestring}_pixel{ipixel}_piximpact.fits")
        piximpactdata = hdulist[1].data.copy()
        piximpact_phids = piximpactdata['PH_ID'].copy()
        hdulist.close()
        # read PIXIMPACT_toxifusim file 
        hdulist = fits.open(f"{filestring}_pixel{ipixel}_piximpact_toxifusim.fits")
        piximpact_toxifusim_data = hdulist[1].data.copy()
        piximpact_toxifusim_phids = piximpact_toxifusim_data['PH_ID'].copy()
        hdulist.close()

        # set the skipped photons as the difference between PH_IDs in piximapct and in piximpact_toxifusim
        skipped_photons_inpix[ipixel] = np.setdiff1d(piximpact_phids, piximpact_toxifusim_phids)
        # get the number of reconstructed photons
        nrecons_inpix[ipixel] = len(reconsdata) + len(skipped_photons_inpix[ipixel])

        # inititalize the lists of photons 
        ph_non_recons_inpix[ipixel] = phsims_inpix.get(ipixel, [])
        ph_bad_recons_inpix[ipixel] = np.array([])
        nnon_recons_inpix[ipixel] = 0
        nbad_recons_inpix[ipixel] = 0

        aux.vprint(f"Pixel {ipixel}: ")
        aux.vprint(f"      {nimpacts_inpix[ipixel]} impacts")
        aux.vprint(f"      {nphsims_inpix[ipixel]} SIXTE simulated photons")
        aux.vprint(f"      {nrecons_inpix[ipixel]} (inititally)reconstructed photons")

        irows_checked = []
        for irow in range(len(PH_ID)):
            if irow in irows_checked:
                aux.vprint(f"    SIRENA: row {irow+1} in pixel {ipixel} already checked")
                continue
            
            aux.vprint(f"SIRENA: Checking row {irow+1} in pixel {ipixel}")
            # get number of values in the array PH_ID[irow] that are /= 0
            nphotons_in_sirena_record = np.count_nonzero(PH_ID[irow])
            nzeros_in_sirena_record = len(PH_ID[irow]) - nphotons_in_sirena_record
            aux.vprint(f"    SIRENA: PH_ID[row={irow+1}]: {PH_ID[irow]}")
            
            if nphotons_in_sirena_record == 1:
                # remove PH_ID[irow] value from the list of non-reconstructed photons
                aux.vprint(f"    SIRENA: Removing photon {PH_ID[irow][0]} from the list of non-reconstructed photons")
                ph_non_recons_inpix[ipixel] = np.delete(ph_non_recons_inpix[ipixel], np.where(ph_non_recons_inpix[ipixel] == PH_ID[irow][0]))
            else:
                # check which photons in record are not reconstructed
                ph_full_sequence_sirena = PH_ID[irow][np.nonzero(PH_ID[irow])]
                #if nzeros_in_sirena_record == 0:
                #    # save message in log_file
                #    with open(log_file, "a") as f:
                #        message = f"Error: MAX_PHIDs in xifusim/sirena reached in pixel {ipixel} of {reconsfile}: execution stopped\n"
                #        f.write(message)
                #        # stop execution
                #        raise ValueError(message)
                # get all the indices of the SIRENA rows with the same values as PH_ID[irow] (all the detections)
                sirena_row_indices_same_photons = np.where((PH_ID == PH_ID[irow]).all(axis=1))[0]
                aux.vprint(f"    SIRENA: rows with same photons: {sirena_row_indices_same_photons+1}")
                # add sirena_row_indices_same_photons values to the list of checked rows
                irows_checked.extend(sirena_row_indices_same_photons)
                
                if len(sirena_row_indices_same_photons) == nphotons_in_sirena_record:
                    # remove ph_full_sequence_sirena values from the list of non-reconstructed photons
                    for ph in ph_full_sequence_sirena:
                        aux.vprint(f"    SIRENA: Removing photon {ph} from the list of non-reconstructed photons")
                        ph_non_recons_inpix[ipixel] = np.delete(ph_non_recons_inpix[ipixel], np.where(ph_non_recons_inpix[ipixel] == ph))
                else:
                    if len(sirena_row_indices_same_photons) < nphotons_in_sirena_record:
                        aux.vprint(f"    SIRENA: Warning: missed {nphotons_in_sirena_record - len(sirena_row_indices_same_photons)} photons in the list {PH_ID[irow]}")
                    else:
                        aux.vprint(f"    SIRENA: Warning: more detections ({len(sirena_row_indices_same_photons)}) than photons ({nphotons_in_sirena_record}) in the list {ph_full_sequence_sirena}")                        
                        nextra_recons_inpix[ipixel] += (len(sirena_row_indices_same_photons) - nphotons_in_sirena_record)

                    # try to identify missed photon(s) looking at the TIME column
                    closest_sirena_row_for_photon = dict()
                    timediff_sirena_row_for_photon = dict()
                    
                    for ph in ph_full_sequence_sirena: # photons simulated by xifusim
                        # look for the photon in the piximpact file and get TIME value
                        idx_ph = np.where(piximpactdata['PH_ID'] == ph)[0]
                        time_ph_piximpact = piximpactdata['TIME'][idx_ph]
                        # look for the photon in the SIRENA file and get TIME value: compare with TIME in piximpact file
                        # assignate the closest SIRENA photon to the XIFUSIM photon
                        min_time_diff = float('inf')
                        for idx in sirena_row_indices_same_photons:
                            time_ph_sirena = reconsdata['TIME'][idx]
                            time_diff = abs(time_ph_piximpact-time_ph_sirena)
                            if time_diff < min_time_diff:
                                min_time_diff = time_diff
                                closest_sirena_row_for_photon[ph] = idx
                                timediff_sirena_row_for_photon[ph] = time_diff                        
                    aux.vprint(f"    SIRENA: closest sirena row for photons[photon_PH_ID:SIRENA_rowIndex]: {closest_sirena_row_for_photon}")
                    # check that number of unique xifusim_sirena_photons values is consistent with SIRENA detected photons: all xifusim
                    # photons should be assigned to a sirena photon. Only if there are invented pulses, number could be different
                    if len(sirena_row_indices_same_photons) != len(set(closest_sirena_row_for_photon.values())):
                        if len(sirena_row_indices_same_photons) < nphotons_in_sirena_record:                      
                            raise ValueError(f"Error: Incorrect assignation of SIRENA {reconsfile} rows to XIFUSIM photons (some sirena photons do not correspond to a xifusim photon)- check assignation time interval")
                    
                    # look if the xifusim_sirena_photons dictionary has duplicated values: identify keys with the same value
                    sirena_detections_rows = list(closest_sirena_row_for_photon.values())
                    xifusim_phs = list(closest_sirena_row_for_photon.keys())
                    unique_sirena_detections = list(set(sirena_detections_rows))
                    # identify the xifusim photons with same sirena identification
                    # loop over the indices of unique sirena_phs:
                    for uni_detection_row in unique_sirena_detections:                            
                        i = sirena_detections_rows.index(uni_detection_row)
                        irow_detection = sirena_detections_rows[i]
                        if sirena_detections_rows.count(irow_detection) > 1: 
                            # get xifusim photons with the same sirena identification
                            indices_mixed_photons = [j for j, x in enumerate(sirena_detections_rows) if x == irow_detection]  
                            aux.vprint(f"    SIRENA: Warning - indices_mixed_photons: {indices_mixed_photons}")
                            aux.vprint(f"    SIRENA: Warning XIFUSIM photons with the same SIRENA row: {[xifusim_phs[j] for j in indices_mixed_photons]}")
                            # remove the closest xifusim photon to the sirena photon from the list of non-reconstructed photons
                            photon_to_remove = PROBPHID[irow_detection]
                            # get energy of the photon to remove (bad-recons) from piximpact file
                            energy_badrecons = piximpactdata['ENERGY'][np.where(piximpactdata['PH_ID'] == photon_to_remove)][0]
                            # and add the photon to the list of compromised photons
                            aux.vprint(f"    SIRENA: adding photon {photon_to_remove} to the list of compromised photons")
                            ph_bad_recons_inpix[ipixel] = np.append(ph_bad_recons_inpix[ipixel], int(photon_to_remove))                                
                            # leave other mixed photons in the list of non-reconstructed photons
                            missing_for_badrecons[photon_to_remove] = []
                            energies_missing_for_badrecons[photon_to_remove] = []
                            distances_missing_for_badrecons[photon_to_remove] = []
                            for j in range(len(indices_mixed_photons)):
                                photon_to_check = xifusim_phs[indices_mixed_photons[j]]
                                if photon_to_check != photon_to_remove:
                                    aux.vprint(f"    SIRENA: leaving photon {photon_to_check} in the list of non-reconstructed photons")
                                    # append photon_to_check to the list of missing photons for badrecons                                        
                                    missing_for_badrecons[photon_to_remove].append(photon_to_check)
                                    # append the energy of the photon to the list of energies
                                    en_miss = piximpactdata['ENERGY'][np.where(piximpactdata['PH_ID'] == photon_to_check)]                            
                                    energies_missing_for_badrecons[photon_to_remove].append(en_miss[0])
                                    # get distance between badrecons and missing photon
                                    dist_miss = (piximpactdata['TIME'][np.where(piximpactdata['PH_ID'] == photon_to_remove)] - 
                                                        piximpactdata['TIME'][np.where(piximpactdata['PH_ID'] == photon_to_check)])
                                    distances_missing_for_badrecons[photon_to_remove].append(dist_miss[0])
                            message = (f"Adding info to table:"
                                        f" Pixel: {ipixel}, badrecons: {photon_to_remove}, energy_badrecons:{energy_badrecons}"
                                        f" Missing phs: {missing_for_badrecons[photon_to_remove]}"
                                        f" Energies_missing:{energies_missing_for_badrecons[photon_to_remove]}"
                                        f" Distances_missing: {distances_missing_for_badrecons[photon_to_remove]}"
                                        f" Distances_missing: {distances_missing_for_badrecons[photon_to_remove]}"
                                        f" GRADE1_badrecons: {GRADE1[closest_sirena_row_for_photon[photon_to_remove]]}"
                                        f" GRADE2_badrecons: {GRADE2[closest_sirena_row_for_photon[photon_to_remove]]}")
                            aux.vprint(message)
                            # add info to the table
                            info_table.loc[len(info_table)] = [ipixel,photon_to_remove, energy_badrecons,
                                                                missing_for_badrecons[photon_to_remove],
                                                                energies_missing_for_badrecons[photon_to_remove],
                                                                distances_missing_for_badrecons[photon_to_remove],
                                                                GRADE1[closest_sirena_row_for_photon[photon_to_remove]],
                                                                GRADE2[closest_sirena_row_for_photon[photon_to_remove]]]
                        else: # only one photon in the list
                            indices_mixed_photons = []
                            photon_to_remove = xifusim_phs[i]
                        # remove the photons found in the SIRENA file from the list of non-reconstructed photons
                        aux.vprint(f"    SIRENA: Removing photon {photon_to_remove} from the list of non-reconstructed photons")
                        ph_non_recons_inpix[ipixel] = np.delete(ph_non_recons_inpix[ipixel], np.where(ph_non_recons_inpix[ipixel] == photon_to_remove))
            # end sirena row with more than 1 photon
            # remove skipped photons in this pixel from the list of non-reconstructed photons
            ph_non_recons_inpix[ipixel] = np.setdiff1d(ph_non_recons_inpix[ipixel], skipped_photons_inpix[ipixel])
            
        # end loop over rows in SIRENA file
        nnon_recons_inpix[ipixel] = len(ph_non_recons_inpix[ipixel])
        nbad_recons_inpix[ipixel] = len(ph_bad_recons_inpix[ipixel])
    # end if pixel was skipped in xifusim (sirena file exists or not)
    nbad_recons_total += nbad_recons_inpix[ipixel]
    #nrecons_total += nrecons_inpix[ipixel] - nextra_recons_inpix[ipixel] 
    nrecons_total += nrecons_inpix[ipixel]
    nnon_recons_total += nnon_recons_inpix[ipixel]       
    nimpacts_total += nimpacts_inpix[ipixel]

    if verbose > 0:
        print(f"Summary for Pixel {ipixel}: ")
        print(f"=====================================")
        print(f"      {nimpacts_inpix[ipixel]} impacts")
        print(f"      {nphsims_inpix[ipixel]} SIXTE simulated photons")
        print(f"      {nextra_recons_inpix[ipixel]} (extra)reconstructed photons")
        print(f"      {nbad_recons_inpix[ipixel]} compromised-recons photons: {ph_bad_recons_inpix[ipixel]}")
        print(f"      {nrecons_inpix[ipixel]} (final)reconstructed photons")
        print(f"      {nnon_recons_inpix[ipixel]} missed photons: {ph_non_recons_inpix[ipixel]}")
        print(f"      {nrecons_total} Accumulated reconstructed photons")

        # print missing photons in the pixel
        if nrecons_inpix[ipixel] < nphsims_inpix[ipixel]:
            # identify in which row of PH_ID_xifusim the missing photons are
            hdulist = fits.open(f"{filestring}_pixel{ipixel}_xifusim.fits")
            xifusimdata = hdulist["TESRECORDS"].data
            PH_ID_xifusim = xifusimdata['PH_ID']
            hdulist.close()
            idx_phs = []
            for ph in ph_non_recons_inpix[ipixel]:
                idx_ph = np.where(PH_ID_xifusim == ph)[0]
                idx_phs.append(idx_ph)
                print(f"      Missed photons: {ph} in xifusim rows {np.array(idx_ph)+1}")
    # save summary information in the log file
    with open(log_file, "a") as f:
        f.write(f"Pixel {ipixel}: \n")
        f.write(f"      {nimpacts_inpix[ipixel]} impacts\n")
        f.write(f"      {nphsims_inpix[ipixel]} SIXTE simulated photons\n")
        f.write(f"      {nextra_recons_inpix[ipixel]} (extra)reconstructed photons\n")
        f.write(f"      {nbad_recons_inpix[ipixel]} compromised-recons photons: {ph_bad_recons_inpix[ipixel]}\n")
        f.write(f"      {nrecons_inpix[ipixel]} (final)reconstructed photons\n")
        f.write(f"      {nnon_recons_inpix[ipixel]} missed photons: {ph_non_recons_inpix[ipixel]}\n")
        f.write(f"      {nrecons_total} Accumulated reconstructed photons\n")
# end loop over pixels


# calculate fraction of lost photons: one is lost and the other one is piledup
#fraction_lost = 1. - nrecons_total/nimpacts_total 
fraction_lost = nnon_recons_total/nimpacts_total
fraction_badrecons = nbad_recons_total/nimpacts_total


# %% [markdown]
# ## Save results   

# %%
# Detailed information about missing photons for the simulation (for each pixel)

aux.vprint(info_table)
# save table to a csv file
infofile = f"{outDir}/00_info_{filter}_{focus}_sim{sim_number}_missing.csv"
info_table.to_csv(infofile, index=False)
aux.vprint(f"Information saved to {infofile}")


# %%

# simulation, flux_mcrab, exposure, filter, focus, number of pixels, number of impacts, number of reconstructed photons, fraction_lost
aux.vprint(f"Simulation {sim_number}:")
aux.vprint(f"      Flux: {flux_mcrab:.3f} mCrab")
aux.vprint(f"      Exposure: {exposure:.2e} s")
aux.vprint(f"      Filter: {filter}")
aux.vprint(f"      Focus: {focus}")
aux.vprint(f"      Number of pixels: {len(pixels_with_impacts)}")
aux.vprint(f"      Number of impacts: {nimpacts_total}")
aux.vprint(f"      Number of reconstructed photons: {nrecons_total}")
aux.vprint(f"      Number of lost photons: {nnon_recons_total}")
aux.vprint(f"      Number of bad reconstructed photons: {nbad_recons_total}")
aux.vprint(f"      Fraction of lost photons: {fraction_lost:.2e}")
aux.vprint(f"      Fraction of bad reconstructed photons: {fraction_badrecons:.2e}")


# %%
#global info
infofile = f"info_{filter}_{focus}_global_{flux_mcrab:.3f}mCrab.csv"

if not os.path.exists(infofile):
    with open(infofile, 'w') as f:
        f.write(f"simulation,flux[mcrab],exposure[s],filter,focus,Npixels,Nimpacts,Nrecons,Missing,Nbadrecons,fraction_lost[%],fraction_badreconstructed[%]\n")
with open(infofile, 'a') as f:
    f.write(f"{sim_number},{flux_mcrab:.3f},{exposure:.2e},{filter},{focus},{len(pixels_with_impacts)},"
            f"{nimpacts_total},{nrecons_total},{nnon_recons_total},{nbad_recons_total},"
            f"{100*fraction_lost:.2e},{100*fraction_badrecons:.2e}\n")

# %%



