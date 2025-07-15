# %% [markdown]
# # FLAG multipulses

# %% [markdown]
# It checks that SIRENA reconstructed photons are correctly labelled based on ELOWRES vs. SIGNAL and AVG4SD vs. SIGNAL confidence contours.   
# 
# For each SIRENA photon, it answers these questions:   
# 1. Is it a single PRIMARY (dist_to_prev > 1563) BUT it would be flagged as PILED-UP (outside confidence intervals)?   
# 2. Is is a PILED-UP pulse (in the table of missing or bad-reconstructed photons) BUT it is flagged as SINGLE?   
# 
# Photons not considered (not analysed):
# * Solitary Pulses that do not have a close previous or posterior partner (< 100 samples). These pulses have not been even simulated in `xifusim` to save resources.
# * Pulses reconstructed with OF-length = 8 samples (same as ELOWRES)   
# * Pulses classified as SECONDARIES: previous pulse closer than 1563 samples -> diagnostic confidence intervals do not work for them   
# 
# PROCEDURE:   
# 1. Import modules   
# 2. Read parameters:   
#     - location of simulated monochromatic files for diagnostic confidence intervals   
#     - width of conf. interval (n_sigmas)   
#     - order of the polynomial for the confidence region fit   
#     - secondaries definition
# 3. Confidence areas definition:   
#     3.1 Read/Reconstruct monochromatic HR pulses with SIRENA using all the pre-defined filter lengths: get ELOWRES and AVG4SD columns   
#     3.2 For each filter length: ELOWRES vs Reconstructed_Calibration_Energy (SIGNAL) -> polynomial fit to confidence region   
#     3.3 For each filter length: AVG4SD vs Reconstructed_Calibration_Energy (SIGNAL) -> polynomial fit to confidence region   
# 4. Run over all the photons in SIRENA files (all simulations for a given flux) and check if there are miss-flagged photons. Add these miss-classifications to a list (for plotting afterwards)   
# 5. Plot:   
#     5.1 confidence intervals for monochromatic simulated pulses (all filter lengths) and residuals plots (data points outside confidence regions)   
#     5.2 miss-flagged photons identified in step 4   
# 
#    

# %% [markdown]
# ### Import modules

# %%
import ipywidgets as widgets 
#%matplotlib widget

import os
import glob
from subprocess import run, PIPE, STDOUT
import tempfile
from astropy.io import fits
import ast
import pandas as pd
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import xml.etree.ElementTree as ET
import auxiliary as aux


# %% [markdown]
# ## Get parameters (check if running Jupyter notebook or Python script (Slurm))

# %% [markdown]
# ###  parameters   
# ```
# secondary_samples: number of samples to the previous pulse to consider a pulse as secondary
# flux_mcrab: flux in mcrab (1 mcrab=2.4E-11 erg/cm^2/s)   
# simEnergies: Energies (keV) of the single simulated pulses
# model: "crab"//TBD   
# filter:  thinOpt // thickOpt // nofilt // thinBe // thickBe    
# focus: '' # 'infoc', 'defoc'  
# verbose: 0 (silent) or 1 (chatty)
# config_version: XIFU configuration file version (for xifusim)
# nsigmas: for confidence interval polyfit
# poly_order: order of polynomial fit to the singles distribution in the ELOWRES vs SIGNAL locus
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
            "secondary_samples": 1563,
            "verbose": 1,
            "simEnergies": [0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            "nsigmas": 5,
            "poly_order": 8,
            "model": "crab",
            "flux_mcrab": 0.32,
            "filter": "nofilt", 
            "focus": "infoc",
            "config_version": "v5_20250621"
        }
    else:
        # Parse command line arguments for script execution
        import argparse
        parser = argparse.ArgumentParser(description="Pairs detection analysis parameters")
        parser.add_argument("--secondary_samples", type=int, default=1563, help="Number of secondary samples")
        parser.add_argument("--verbose", type=int, default=0, help="Verbosity? 0 for silent, 1 for chatty", choices=[0,1])
        parser.add_argument("--simEnergies", nargs='*', type=float, default=[0.2,0.5,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0], 
                            help="Simulated energies of single pulses (in keV)")
        parser.add_argument("--nsigmas", type=int, default=5, help="Number of sigmas for confidence interval polyfit")
        parser.add_argument("--poly_order", type=int, default=8, help="Polynomial order for fit to singles distribution in ELOWRES vs SIGNAL locus")
        parser.add_argument("--model", type=str, default="crab", help="Source Model to use for simulation")
        parser.add_argument("--flux_mcrab", type=float, required=True, help="Flux in mCrab")
        parser.add_argument("--filter", type=str, default="nofilt",choices=['thinOpt','thickOpt','nofilt','thinBe','thickBe'], help="Filter to use")
        parser.add_argument("--focus", type=str, default="infoc", choices=['infoc','defoc'], help="Focus mode")
        parser.add_argument("--config_version", type=str, default="v5_20250621", choices=['v3_20240917','v5_20250621'], help="Configuration version")

        args = parser.parse_args()
        
        return vars(args)

# %% [markdown]
# ### Get parameters

# %%
params = get_parameters()
secondary_samples = params["secondary_samples"]
verbose = params["verbose"]
simEnergies = params["simEnergies"]
nsigmas = params["nsigmas"]
poly_order = params["poly_order"]
model = params["model"]
flux_mcrab = params["flux_mcrab"]
filter = params["filter"]
focus = params["focus"]
config_version = params["config_version"]

# %% [markdown]
# ### Derived parameters

# %%
tmpDir = tempfile.mkdtemp()
os.environ["PFILES"] = f"{tmpDir}:{os.environ['PFILES']}"
os.environ["HEADASNOQUERY"] = ""
os.environ["HEADASPROMPT"] = "/dev/null/"
SIXTE = os.environ["SIXTE"]

# %%
workdir = "/dataj6/ceballos/INSTRUMEN/EURECA/TN350_detection/2024_revision/"
datadir = f"{workdir}/{config_version}/singles"
# read xifusim simulated files
nsims=100
simfiles = []
for ener in simEnergies:
    simfile = f"{datadir}/mono{ener}keV_5000p_50x30.fits"
    simfiles.append(simfile)
#simfiles = glob.glob(f"{datadir}/mono*keV_5000p_50x30.fits")
if len(simfiles) > len(simEnergies):
    raise ValueError("More files than energies")

flux_mcrab_str = f"{flux_mcrab:.2f}"
fluxdir = f"{workdir}/{config_version}/flux{flux_mcrab_str}mcrab"
if flux_mcrab > 999:
    nsims=200

# %% [markdown]
# ## Create confidence contours in diagnostic plots

# %% [markdown]
# ### Read/create SIRENA monochromatic files   
# 
# Read/Reconstruct monochromatic `xifusim` simulated files using all filter lengths in XML file

# %%
lib_sirena = f"{datadir}/optimal_filters_6keV_50x30.fits"
assert os.path.exists(lib_sirena), f"{lib_sirena} does not exist"
if config_version == "v3_20240917":
    xml_xifusim = f"{workdir}/config_xifu_50x30_v3_20240917.xml"
elif config_version == "v5_20250621":
    xml_xifusim = f"{workdir}/config_xifu_50x30_v5_v20250621.xml"
else:
    raise ValueError(f"Undefined config_version: {config_version}")
assert os.path.exists(xml_xifusim), f"{xml_xifusim} does not exist"
# read filter lengths from XML file: under <reconstruction> tag in filtlen attribute
# Load and parse the XML file
tree = ET.parse(xml_xifusim)
root = tree.getroot()
# Extract all values of "filtlen" attributes
filtlen_values = [elem.attrib["filtlen"] for elem in root.iter() if "filtlen" in elem.attrib]
# convert it to a list of integers
filtlen_values = [int(x) for x in filtlen_values]

# %%
# reconstruct simulated files with SIRENA
reconsfiles = []
for oflen in filtlen_values:
    for xifusimfile in simfiles:
        assert os.path.exists(xifusimfile), f"{xifusimfile} does not exist"
        # remove path from filename:
        filename = os.path.basename(xifusimfile)
        # get energy value from xifusim file name
        energy = float(filename.split("_")[0].replace("keV", "").replace("mono", ""))
        reconsfile = f"{datadir}/events_mono{energy}keV_5000p_50x30_of{oflen}.fits"
        if not os.path.exists(reconsfile):
            comm = (f"tesrecons"
                    f" Recordfile={xifusimfile}"
                    f" TesEventFile={reconsfile}"
                    f" LibraryFile={lib_sirena}"
                    f" XMLFile={xml_xifusim}"
                    f" clobber=yes"
                    f" EnergyMethod=OPTFILT"
                    f" OFStrategy=FIXED"
                    f" OFLength={oflen}"
                    f" filtEeV=6000"
                    f" OFNoise=NSD"
                    f" threshold=6"
                    f" samplesUp=3"
                    f" samplesDown=2"
            )
            aux.vprint(f"Doing reconstruction for xifusim file {xifusimfile}")
            aux.vprint(f"Running {comm}")
            
            output_tesrecons = run(comm, shell=True, capture_output=True)
            assert output_tesrecons.returncode == 0, f"tesrecons failed to run => {output_tesrecons.stderr.decode()}"
            assert os.path.exists(reconsfile), f"tesrecons did not produce an output file"
        else:
            aux.vprint(f"Reconstructed file {reconsfile} already exists")
        # add reconsfile to the list of reconsfiles
        reconsfiles.append(reconsfile)
#reconsfiles = glob.glob(f"singles/events_mono*keV_5000p_50x30_*.fits")
# get the maximum number of photons in the reconstructed files
max_photons = aux.get_max_photons(reconsfiles)        

# %% [markdown]
# ### Store information of SIRENA reconstruction of monochromatic files   
# 
# Save info of columns SIGNAL, ELOWRES and AVG4SD and take also median and stdev values     

# %%

# create numpy arrays to store the data
# the first dimension is the energy, the second dimension is the number of photons  
# inititalize the arrays with NaN
SIGNAL_mono = np.full((len(simEnergies),len(filtlen_values),max_photons), np.nan)
ELOWRES_mono = np.full((len(simEnergies), len(filtlen_values),max_photons), np.nan)
DIFFELOWRES_mono = np.full((len(simEnergies), len(filtlen_values),max_photons), np.nan)
AVG4SD_mono = np.full((len(simEnergies), len(filtlen_values),max_photons), np.nan)
medianSIGNAL = np.full((len(simEnergies), len(filtlen_values)), np.nan)
medianELOWRES = np.full((len(simEnergies), len(filtlen_values)), np.nan)
medianDIFFELOWRES = np.full((len(simEnergies), len(filtlen_values)), np.nan)
stdELOWRES = np.full((len(simEnergies), len(filtlen_values)), np.nan)
stdDIFFELOWRES = np.full((len(simEnergies), len(filtlen_values)), np.nan)
medianAVG4SD = np.full((len(simEnergies), len(filtlen_values)), np.nan)
stdAVG4SD = np.full((len(simEnergies), len(filtlen_values)), np.nan)

#initialize array to store simulated energies in integer format
simEnergies_lab = np.array(simEnergies, dtype='str')

for ifl in range(len(filtlen_values)):
    ofl = filtlen_values[ifl]
    for ie in range(len(simEnergies)):
        simE = simEnergies[ie]
        if simE >= 1:
            simEnergies_lab[ie] = int(simEnergies[ie])
        sirena_file = f"{datadir}/events_mono{simE}keV_5000p_50x30_of{ofl}.fits"
        #print(f"Reading {sirena_file}")
        f = fits.open(sirena_file)
        # read the data and store 
        # store the array in the SIGNAL column in the second dimension of the SIGNAL array
        signal_data = f[1].data['SIGNAL']
        elow_data = f[1].data['ELOWRES']
        avg4sd_data = f[1].data['AVG4SD']
        if len(signal_data) == 0:
            print(f"Warning: No data in {sirena_file} for energy {simE} keV and filter length {ofl}")
            raise ValueError(f"No data in {sirena_file} for energy {simE} keV and filter length {ofl}")
        SIGNAL_mono[ie, ifl, :len(signal_data)] = signal_data
        ELOWRES_mono[ie, ifl, :len(elow_data)] = elow_data
        DIFFELOWRES_mono[ie, ifl, :len(elow_data)] = elow_data - signal_data
        AVG4SD_mono[ie, ifl, :len(avg4sd_data)] = avg4sd_data
        medianSIGNAL[ie, ifl] = np.nanmedian(signal_data)
        medianELOWRES[ie, ifl] = np.nanmedian(elow_data)
        medianDIFFELOWRES[ie, ifl] = np.nanmedian(DIFFELOWRES_mono[ie, ifl])
        stdDIFFELOWRES[ie, ifl] = np.nanstd(DIFFELOWRES_mono[ie, ifl])
        stdELOWRES[ie, ifl] = np.nanstd(elow_data)
        medianAVG4SD[ie, ifl] = np.nanmedian(avg4sd_data)
        stdAVG4SD[ie, ifl] = np.nanstd(avg4sd_data)
        f.close()
    

# %% [markdown]
# ### Polynomial fit to median+n_sigmas*stdev    
# 
# Create these topt & bottom polynomials for all the filter lengths

# %%
# foreach filter length fit polynomials to create confidence intervals
# initialize arrays to store the polynomial coefficients for the confidence intervals
poly_top_coeffs_ELOWRES = np.full((len(filtlen_values), poly_order+1), np.nan)
poly_bottom_coeffs_ELOWRES = np.full((len(filtlen_values), poly_order+1), np.nan)
poly_top_coeffs_DIFFELOWRES = np.full((len(filtlen_values), poly_order+1), np.nan)
poly_bottom_coeffs_DIFFELOWRES = np.full((len(filtlen_values), poly_order+1), np.nan)
poly_top_coeffs_AVG4SD = np.full((len(filtlen_values), poly_order+1), np.nan)
poly_bottom_coeffs_AVG4SD = np.full((len(filtlen_values), poly_order+1), np.nan)

for ifl in range(len(filtlen_values)):
    poly_coeffs_dict = aux.get_polyfit_intervals_columns(columnX=SIGNAL_mono[:,ifl,:], columnY=ELOWRES_mono[:,ifl,:], nsigmas=nsigmas, order=poly_order)
    poly_top_coeffs_ELOWRES[ifl] = poly_coeffs_dict['top']
    poly_bottom_coeffs_ELOWRES[ifl] = poly_coeffs_dict['bottom']

    poly_coeffs_dict = aux.get_polyfit_intervals_columns(columnX=SIGNAL_mono[:,ifl,:], columnY=DIFFELOWRES_mono[:,ifl,:], nsigmas=nsigmas, order=poly_order)
    poly_top_coeffs_DIFFELOWRES[ifl] = poly_coeffs_dict['top']
    poly_bottom_coeffs_DIFFELOWRES[ifl] = poly_coeffs_dict['bottom']

    poly_coeffs_dict = aux.get_polyfit_intervals_columns(columnX=SIGNAL_mono[:,ifl,:], columnY=AVG4SD_mono[:,ifl,:], nsigmas=nsigmas, order=poly_order)
    poly_top_coeffs_AVG4SD[ifl] = poly_coeffs_dict['top']
    poly_bottom_coeffs_AVG4SD[ifl] = poly_coeffs_dict['bottom']

# %% [markdown]
# ## Check flagging of SIRENA photons

# %% [markdown]
# For each SIRENA reconstructed photons (only close photons have been simulated and reconstructed), check if:   
# * Single photons are flagged as 'singles'   
# * Piled-up photons ('bad-reconstructed' or 'Non-reconstructed') are flagged as piled-up       

# %%
# create a structure to save SIGNAL, ELOWRES and AVG4SD of incorrectly flagged photons
bad_flagged = {"SIGNAL": [], "ELOWRES": [], "AVG4SD": [], "GRADE1": [], "BADREC_ENERGY": [], "MISS0_ENERGY": [], "DIST_BADREC_MISS0": []}
# create a structure to save SIGNAL, ELOWRES and AVG4SD of all badrecons photons
all_badrecons = {"SIGNAL": [], "ELOWRES": [], "AVG4SD": [], "GRADE1": [], "BADREC_ENERGY": [], "MISS0_ENERGY": [], "DIST_BADREC_MISS0": []}

for isim in range(1,nsims+1):
    #if not isim == 1:
    #    continue
    aux.vprint(f"Processing sim_{isim} and flux{flux_mcrab_str}mcrab")
    print(f"Looking for {fluxdir}/sim_{isim}/{model}_flux{flux_mcrab_str}_Emin2_Emax10_exp*_RA0.0_Dec0.0_{filter}_{focus}_pixel*")
    filestring = glob.glob(f"{fluxdir}/sim_{isim}/{model}_flux{flux_mcrab_str}_Emin2_Emax10_exp*_RA0.0_Dec0.0_{filter}_{focus}_pixel*")
    if len(filestring) == 0:
        aux.vprint(f"*** WARNING: No files found for sim_{isim} and flux{flux_mcrab_str}mcrab")
        raise ValueError(f"*** WARNING: No files found for sim_{isim} and flux{flux_mcrab_str}mcrab")
    
    # remove the last part of filestring after the 'focus' tag
    filestring = filestring[0].split(f"_pixel")[0]

    # read CSV file
    csv_file = f"{fluxdir}/sim_{isim}/00_info_{filter}_{focus}_sim{isim}_missing.csv"
    missing_table = pd.read_csv(csv_file, comment="#", converters={"Non-reconstructed photons": ast.literal_eval,
                                                    "Non-reconstructed photons energies": ast.literal_eval,
                                                    "Non-reconstructed photons distances": ast.literal_eval})
    
    # list of SIRENA files
    reconsfiles = glob.glob(f"{filestring}_pixel*_sirena.fits")
    if len(reconsfiles) == 0:
        aux.vprint(f"*** WARNING: No SIRENA files found for sim_{isim} and flux{flux_mcrab_str}mcrab")
        raise ValueError(f"*** WARNING: No SIRENA files found for sim_{isim} and flux{flux_mcrab_str}mcrab")
    for reconsfile in reconsfiles:
        #aux.vprint(f"Checking {reconsfile}")
        ipixel = int(reconsfile.split("_")[-2].replace("pixel", ""))
        piximpact_file = f"{filestring}_pixel{ipixel}_piximpact.fits"
        
        # read the data from the SIRENA file
        sirena_data = aux.get_sirena_info(sirena_file=reconsfile, impact_file=piximpact_file)
        
        for iph in range(len(sirena_data["SIGNAL"])):
            time_iph = sirena_data['TIME'][iph]
            signal_iph = sirena_data['SIGNAL'][iph]
            elowres_iph = sirena_data['ELOWRES'][iph]
            avg4sd_iph  = sirena_data['AVG4SD'][iph]
            ph_id_iph = sirena_data['PROBPHID'][iph]
            grade1_iph = sirena_data['GRADE1'][iph]
            grade2_iph = sirena_data['GRADE2'][iph]

            is_photon_in_missing = any(ph_id_iph in tabrow for tabrow in missing_table["Non-reconstructed photons"])
            if is_photon_in_missing:
                message = (f"*** ERROR: In sim_{isim} pixel {ipixel}, reconstructed photon is in list of missing photons (it should not!)\n"
                           f"           TIME={time_iph:.3e}; PROBPHID={ph_id_iph}")
                raise ValueError(message)
                
            if grade1_iph == 8:
                #aux.vprint(f"*** WARNING: Photon {ph_id_iph} for sim_{isim} pixel {ipixel} is ELOWRES: skipping")
                continue
            # secondary photons cannot be compared with the confidence intervals (created for primary photons only)
            if grade2_iph <= secondary_samples:
            #    aux.vprint(f"*** WARNING: Photon {ph_id_iph} for sim_{isim} pixel {ipixel} is SECONDARY: skipping")
                continue               
                         
            # check if ph_id_iph is in the list of bad-reconstructed (or missing) photons
            is_photon_bad_reconstructed = (ph_id_iph in missing_table["Bad-reconstructed photons"].values)
            
            # get the index value in filtlen_values that is closest to grade1_irow
            ifl = np.argmin(np.abs(filtlen_values-grade1_iph))
            
            # get the polynomial values for the SIGNAL value: choose the filter length that is closest to grade1_irow
            top_ELOWRES = poly.polyval(signal_iph, poly_top_coeffs_ELOWRES[ifl])
            bottom_ELOWRES = poly.polyval(signal_iph, poly_bottom_coeffs_ELOWRES[ifl])
            top_AVG4SD = poly.polyval(signal_iph, poly_top_coeffs_AVG4SD[ifl])
            bottom_AVG4SD = poly.polyval(signal_iph, poly_bottom_coeffs_AVG4SD[ifl])
            if (elowres_iph < top_ELOWRES and elowres_iph > bottom_ELOWRES and 
                avg4sd_iph < top_AVG4SD and avg4sd_iph > bottom_AVG4SD): # single photon
                flag_piledup = False
            else:
                flag_piledup = True
            
            wrong_flag = False
            if flag_piledup and not is_photon_bad_reconstructed:
                wrong_flag = True
                aux.vprint(f"*** WARNING: Single photon {ph_id_iph} for sim_{isim} pixel {ipixel} would be flagged as a piled-up")
            elif not flag_piledup and is_photon_bad_reconstructed:
                wrong_flag = True
                aux.vprint(f"*** WARNING: Bad-recons photon {ph_id_iph} for sim_{isim} pixel {ipixel} would be flagged as single")
            if wrong_flag:
                aux.vprint(f"             elowres_irow = {elowres_iph}, top_ELOWRES = {top_ELOWRES}, bottom_ELOWRES = {bottom_ELOWRES}")
                aux.vprint(f"             avg4sd_irow = {avg4sd_iph}, top_AVG4SD = {top_AVG4SD}, bottom_AVG4SD = {bottom_AVG4SD}")
                # store the photon in the bad_flagged dictionary
                # save signal, elowres, avg4sd to be added to the plot
                bad_flagged["SIGNAL"].append(signal_iph)
                bad_flagged["ELOWRES"].append(elowres_iph)
                bad_flagged["AVG4SD"].append(avg4sd_iph)
                bad_flagged["GRADE1"].append(grade1_iph)
                # initialize the energies to NaN until we get the values from the missing_table
                bad_flagged["BADREC_ENERGY"].append(np.nan)
                bad_flagged["MISS0_ENERGY"].append(np.nan)
                bad_flagged["DIST_BADREC_MISS0"].append(np.nan)
            # save signal, elowres, avg4sd of ALL BADRECONS/MISSING photons to be added to the plot
            if is_photon_bad_reconstructed:
                # get the index value in missing_table that corresponds to the bad-reconstructed photon
                badr_index_table = missing_table[missing_table["Bad-reconstructed photons"] == ph_id_iph].index
                # get the first value of missing energies in table at index badr_index_table
                miss0_energy = missing_table["Non-reconstructed photons energies"][badr_index_table[0]][0]
                miss0_dist = missing_table["Non-reconstructed photons distances"][badr_index_table[0]][0]
                # get the badrecons energy value
                badrec_energy = missing_table["Bad-reconstructed photons energy"][badr_index_table[0]]
                # do not consider BGD photons with "0.0" energy
                if miss0_energy == 0.0:
                    continue
                all_badrecons["SIGNAL"].append(signal_iph)
                all_badrecons["ELOWRES"].append(elowres_iph)
                all_badrecons["AVG4SD"].append(avg4sd_iph)
                all_badrecons["GRADE1"].append(grade1_iph)
                
                all_badrecons["MISS0_ENERGY"].append(miss0_energy)
                all_badrecons["BADREC_ENERGY"].append(badrec_energy)
                all_badrecons["DIST_BADREC_MISS0"].append(miss0_dist)

                if wrong_flag:
                    bad_flagged["BADREC_ENERGY"].append(badrec_energy)
                    bad_flagged["MISS0_ENERGY"].append(miss0_energy)
                    bad_flagged["DIST_BADREC_MISS0"].append(miss0_dist)


# %% [markdown]
# ## Plot confidence contours and miss-flagged photons   
# 
# Plot the data points from the simulated monochromatic pulses reconstructed with different filter lengths.   
# Overplot the miss-flagged photons.   
# Figure:   
# 
# |   ELOWRES vs SIGNAL                |          AVGD4SD vs SIGNAL     |
# |----------------------------------- | ------------------------------ |
# | Residuals (Points out of interval) | Residuals (Points out of interval) |

# %% [markdown]
# ### Change Y axis scale

# %%
bad_flagged_SIGNAL = np.array(bad_flagged["SIGNAL"])
bad_flagged_ELOWRES = np.array(bad_flagged["ELOWRES"])
bad_flagged_GRADE1 = np.array(bad_flagged["GRADE1"])
bad_flagged_AVG4SD = np.array(bad_flagged["AVG4SD"])
bad_flagged_BADREC_ENERGY = np.array(bad_flagged["BADREC_ENERGY"])
bad_flagged_MISS0_ENERGY = np.array(bad_flagged["MISS0_ENERGY"])

# remove negative values from bad_flagged_ELOWRES and bad_flagged_SIGNAL (set to NaN - problem ocurred during reconstruction)
# identify indices of negative values in bad_flagged_ELOWRES or bad_flagged_SIGNAL
bad_flagged_ELOWRES[bad_flagged_ELOWRES < 0] = np.nan
indices_nan_elowres = np.where(np.isnan(bad_flagged_ELOWRES))
bad_flagged_SIGNAL[bad_flagged_SIGNAL < 0] = np.nan
indices_nan_signal = np.where(np.isnan(bad_flagged_SIGNAL))
bad_flagged_DIFFELOWRES = bad_flagged_ELOWRES- bad_flagged_SIGNAL

# remove negative values from all_bad_recons_ELOWRES and all_badrecons_SIGNAL (set to NaN - problem ocurred during reconstruction)
# identify indices of negative values in all_badrecons_ELOWRES or all_badrecons_SIGNAL
all_badrecons_SIGNAL = np.array(all_badrecons["SIGNAL"])
all_badrecons_SIGNAL[all_badrecons_SIGNAL < 0] = np.nan
all_badrecons_MISS0_ENERGY = np.array(all_badrecons["MISS0_ENERGY"])
all_badrecons_BADREC_ENERGY = np.array(all_badrecons["BADREC_ENERGY"])
all_badrecons_ELOWRES = np.array(all_badrecons["ELOWRES"])
all_badrecons_ELOWRES[all_badrecons_ELOWRES < 0] = np.nan
all_badrecons_GRADE1 = np.array(all_badrecons["GRADE1"])
all_badrecons_DIFFELOWRES = all_badrecons_ELOWRES - all_badrecons_SIGNAL


# %%
plot_bad_flagged = True #if True, plot the flagged photons

fig, axes = plt.subplots(2, 2, figsize=(12, 14))
(ax1,ax2),(ax3,ax4) = axes
# =================================================
# PLOT 1-TOPL: ELOWRES-SIGNAL vs SIGNAL FOR SINGLES
# =================================================
ax1.set_xlabel("[SIGNAL] (uncalibrated ~keV)")
ax1.set_ylabel("[ELOWRES]-[SIGNAL] (uncalibrated ~keV)")
ax1.set_xlim(0, 12.5)
ax1.set_ylim(-1.5, 0.25)
ytop = ax1.get_ylim()[1]
xtoplot = np.linspace(0.1,13, 100)
# add a top title for the energy values
ax1.text(0.5*ax1.get_xlim()[1], 1.35*ytop, "Simulated Energy (keV)", ha='center', va='center', fontsize='small', color="darkgray")

for ifl in range(len(filtlen_values)):
    ofl = filtlen_values[ifl]
    # skip OFLEN=8 as it is equal to ELOWRES (not useful as diagnostic parameter)
    if ofl == 8:
        continue
    points_color = f"C{ifl}"
    for ie in range(len(simEnergies)):
        simE = simEnergies[ie]
        ax1.plot(SIGNAL_mono[ie, ifl], DIFFELOWRES_mono[ie, ifl], marker='.', linestyle='None',markersize=3, color=points_color)
        # plot simulated energy vertical lines (only for first filter length)
        if ifl == 0:
            ax1.axvline(medianSIGNAL[ie, 0], color='gray', linestyle='--', alpha=0.1)
            # add labels where the vertical lines cross the top axis
            if not simEnergies[ie] == 0.5:
                ax1.text(medianSIGNAL[ie, 0], 0.3, simEnergies_lab[ie], ha='center', va='center', fontsize='small', color="darkgray")
            # add also small ticks at the top axis
            ax1.plot(medianSIGNAL[ie, 0], 0.25, alpha=0.5, marker='|', markersize=5,color="black")
        # plot median and the error bars
        ax1.errorbar(medianSIGNAL[ie, ifl], medianDIFFELOWRES[ie,ifl], yerr=nsigmas*stdDIFFELOWRES[ie, ifl], fmt='x', color='black', markersize=1)

    poly_top_plot = poly.polyval(xtoplot, poly_top_coeffs_DIFFELOWRES[ifl])
    poly_bottom_plot = poly.polyval(xtoplot, poly_bottom_coeffs_DIFFELOWRES[ifl])
    ax1.plot(xtoplot, poly_top_plot, linestyle='--', color=points_color, label=f'OFL: {ofl}')
    ax1.plot(xtoplot, poly_bottom_plot, linestyle='--', color=points_color)

# Write text in the ploting area (normalized coordinates) with information about the flux
text1 = f"Singles location"
text2 = f"Flux: {flux_mcrab_str} mCrab"
ax1.text(0.25, 0.6, text1, transform=ax1.transAxes, fontsize=10, verticalalignment='top')
ax1.text(0.25, 0.45, text2, transform=ax1.transAxes, fontsize=10, verticalalignment='top')

ax1.legend(fontsize='small')

# ============================================================
# PLOT 2-TOPR: ELOWRES-SIGNAL vs SIGNAL (SINGLES) & BADRECONS
# ============================================================

ax2.set_xlabel("[SIGNAL] (uncalibrated ~keV)")
ax2.set_ylabel("[ELOWRES]-[SIGNAL] (uncalibrated ~keV)")
ax2.set_xlim(0, 12.5)
ax2.set_ylim(-10.5, 0.55)
ytop = ax2.get_ylim()[1]
xtoplot = np.linspace(0.1,13, 100)
# add a top title for the energy values
ax2.text(0.5*ax1.get_xlim()[1], 1.75*ytop, "Simulated Energy (keV)", ha='center', va='center', fontsize='small', color="darkgray")

# initialize dictionary for point colors
points_color = {}
for ifl in range(len(filtlen_values)):
    ofl = filtlen_values[ifl]
    # skip OFLEN=8 as it is equal to ELOWRES (not useful as diagnostic parameter)
    if ofl == 8:
        continue
    points_color[ifl] = f"C{ifl}"
    for ie in range(len(simEnergies)):
        simE = simEnergies[ie]
        ax2.plot(SIGNAL_mono[ie, ifl], DIFFELOWRES_mono[ie, ifl], marker='.', linestyle='None',markersize=3, color=points_color[ifl])
        # plot simulated energy vertical lines (only for first filter length)
        if ifl == 0:
            ax2.axvline(medianSIGNAL[ie, 0], color='gray', linestyle='--', alpha=0.1)
            # add labels where the vertical lines cross the top axis
            if not simEnergies[ie] == 0.5:
                ax2.text(medianSIGNAL[ie, 0], 0.77, simEnergies_lab[ie], ha='center', va='center', fontsize='small', color="darkgray")
            # add also small ticks at the top axis
            ax2.plot(medianSIGNAL[ie, 0], 0.55, alpha=0.5, marker='|', markersize=5,color="black")
        # plot median and the error bars
        ax2.errorbar(medianSIGNAL[ie, ifl], medianDIFFELOWRES[ie,ifl], yerr=nsigmas*stdDIFFELOWRES[ie, ifl], fmt='x', color='black', markersize=1)

    poly_top_plot = poly.polyval(xtoplot, poly_top_coeffs_DIFFELOWRES[ifl])
    poly_bottom_plot = poly.polyval(xtoplot, poly_bottom_coeffs_DIFFELOWRES[ifl])
    ax2.plot(xtoplot, poly_top_plot, linestyle='--', color=points_color[ifl], label=f'OFL: {ofl}')
    ax2.plot(xtoplot, poly_bottom_plot, linestyle='--', color=points_color[ifl])

    # plot all badrecons photons (x marks colored by value of GRADE1)
    indices_to_plot = np.where(all_badrecons_GRADE1 == ofl)
    ax2.plot(all_badrecons_SIGNAL[indices_to_plot], all_badrecons_DIFFELOWRES[indices_to_plot], marker='x', linestyle='None', markersize=3, 
         color=points_color[ifl], label=f'{len(all_badrecons_SIGNAL[indices_to_plot])} prim-badrecons phs (OFL: {ofl})')

# plot bad_flagged photons (red x)
ax2.plot(bad_flagged_SIGNAL, bad_flagged_DIFFELOWRES, marker='x', linestyle='None', markersize=3, 
             color='red', label=f'{len(bad_flagged_SIGNAL)} wrong flagged primary phs')
# set legend for the 5-sigma confidence interval
ax2.legend(loc='lower left', fontsize='small')

# ==========================
# PLOT 3-BOTTL: ELOWRES-SIGNAL vs SIGNAL & BADRECONS (BY ENERGY RATIO)
# ==========================
ax3.set_xlabel("[SIGNAL] (uncalibrated ~keV)")
ax3.set_ylabel("[ELOWRES]-[SIGNAL] (uncalibrated ~keV)")
ax3.set_xlim(0, 12.5)
ax3.set_ylim(-10.5, 0.55)
ytop = ax3.get_ylim()[1]
xtoplot = np.linspace(0.1,13, 100)
# add a top title for the energy values
ax3.text(0.5*ax1.get_xlim()[1], 1.75*ytop, "Simulated Energy (keV)", ha='center', va='center', fontsize='small', color="darkgray")

# initialize dictionary for (singles) point colors
points_color = {}
for ifl in range(len(filtlen_values)):
    ofl = filtlen_values[ifl]
    # skip OFLEN=8 as it is equal to ELOWRES (not useful as diagnostic parameter)
    if ofl == 8:
        continue
    points_color[ifl] = f"C{ifl}"
    for ie in range(len(simEnergies)):
        simE = simEnergies[ie]
        ax3.plot(SIGNAL_mono[ie, ifl], DIFFELOWRES_mono[ie, ifl], marker='.', linestyle='None',markersize=3, color=points_color[ifl])
        # plot simulated energy vertical lines (only for first filter length)
        if ifl == 0:
            ax3.axvline(medianSIGNAL[ie, 0], color='gray', linestyle='--', alpha=0.1)
            # add labels where the vertical lines cross the top axis
            if not simEnergies[ie] == 0.5:
                ax3.text(medianSIGNAL[ie, 0], 0.77, simEnergies_lab[ie], ha='center', va='center', fontsize='small', color="darkgray")
            # add also small ticks at the top axis
            ax3.plot(medianSIGNAL[ie, 0], 0.55, alpha=0.5, marker='|', markersize=5,color="black")
        # plot median and the error bars
        ax3.errorbar(medianSIGNAL[ie, ifl], medianDIFFELOWRES[ie,ifl], yerr=nsigmas*stdDIFFELOWRES[ie, ifl], fmt='x', color='black', markersize=1)

    poly_top_plot = poly.polyval(xtoplot, poly_top_coeffs_DIFFELOWRES[ifl])
    poly_bottom_plot = poly.polyval(xtoplot, poly_bottom_coeffs_DIFFELOWRES[ifl])
    ax3.plot(xtoplot, poly_top_plot, linestyle='--', color=points_color[ifl], label=f'OFL: {ofl}')
    ax3.plot(xtoplot, poly_bottom_plot, linestyle='--', color=points_color[ifl])

# plot all badrecons photons (x marks colored by ratio of ENERGY with its MISSING partner)
#ratio_badrecons = all_badrecons_SIGNAL / all_badrecons_MISS0_ENERGY
ratio_badrecons = all_badrecons_BADREC_ENERGY / all_badrecons_MISS0_ENERGY
ratio_badrecons_log = np.log10(ratio_badrecons)
# create a colormap in log scale
norm = mcolors.Normalize(vmin=min(ratio_badrecons), vmax=max(ratio_badrecons))
norm_log = mcolors.Normalize(vmin=min(ratio_badrecons_log), vmax=max(ratio_badrecons_log))
sc = ax3.scatter(all_badrecons_SIGNAL, all_badrecons_DIFFELOWRES, marker='x', s=3,
            c=ratio_badrecons_log, cmap='cividis', norm=norm_log, alpha=0.5)
# plot colorbar
cbar = fig.colorbar(sc, ax=ax3)
#cbar.set_label('log(SIGNAL(badrecons()/ENERGY(missing_partner))', fontsize='small')
cbar.set_label('log(ENERGY(badrecons()/ENERGY(missing_partner))', fontsize='small')


# plot bad_flagged photons (red x)
ax3.plot(bad_flagged_SIGNAL, bad_flagged_DIFFELOWRES, marker='x', linestyle='None', markersize=3, 
             color='red', label=f'{len(bad_flagged_SIGNAL)} wrong flagged primary phs')

# set legend for the 5-sigma confidence interval
ax3.legend(loc='lower left', fontsize='small')

# ==========================
# PLOT 4-BOTTOMR: Histogram of ENERGY ratios of badrecons photons
# =========================
#ax4.set_xlabel("SIGNAL(badrecons()/ENERGY(missing_partner)")
ax4.set_xlabel("ENERGY(badrecons()/ENERGY(missing_partner)")
ax4.set_ylabel("Counts")
#ax4.set_xlim(0, 2.5)
#ax4.set_ylim(0, 100)
ax4.hist(ratio_badrecons, bins=50, color='blue', alpha=0.5)
text1 = f"Piled-up photons distribution\n"
text2 = f"Flux: {flux_mcrab_str} mCrab"
ax4.text(0.5, 0.6, text1, transform=ax4.transAxes, fontsize=10, verticalalignment='top')
ax4.text(0.5, 0.45, text2, transform=ax4.transAxes, fontsize=10, verticalalignment='top')

fig.tight_layout()


# %%
print(len(bad_flagged_SIGNAL))
print(len(bad_flagged_ELOWRES))
print(len(bad_flagged_GRADE1))
print(len(bad_flagged_BADREC_ENERGY))
print(len(bad_flagged_MISS0_ENERGY))
print(bad_flagged.keys())

# %%
# save a csv file with all the information about the badrecons photons
badrecons_df = pd.DataFrame({"SIGNAL": all_badrecons_SIGNAL, "ELOWRES": all_badrecons_ELOWRES,
                             "AVG4SD": all_badrecons["AVG4SD"], "GRADE1": all_badrecons_GRADE1,
                             "MISS0_ENERGY": all_badrecons_MISS0_ENERGY, "BADREC_ENERGY": all_badrecons_BADREC_ENERGY,
                             "DIST_BADREC_MISS0": all_badrecons["DIST_BADREC_MISS0"],
                             "diffELOWRES": all_badrecons_DIFFELOWRES})
badrecons_df.to_csv(f"Figures/flagging/badrecons_photons_{filter}_{focus}_{flux_mcrab_str}mCrab.csv", index=False)

# save a csv file with all the information about the bad_flagged photons
# if empty list, create a dataframe only with column names
if len(bad_flagged["SIGNAL"]) == 0:
    bad_flagged_df = pd.DataFrame(columns=["SIGNAL", "ELOWRES", "AVG4SD", "GRADE1", "BADREC_ENERGY", "MISS0_ENERGY", "diffELOWRES"])
else:
    bad_flagged_df = pd.DataFrame({"SIGNAL": bad_flagged_SIGNAL, "ELOWRES": bad_flagged_ELOWRES,
                                   "AVG4SD": bad_flagged_AVG4SD, "GRADE1": bad_flagged_GRADE1,
                                   "BADREC_ENERGY": bad_flagged_BADREC_ENERGY, "MISS0_ENERGY": bad_flagged_MISS0_ENERGY,
                                   "DIST_BADREC_MISS0": bad_flagged["DIST_BADREC_MISS0"],
                                   "diffELOWRES": bad_flagged_DIFFELOWRES})
bad_flagged_df.to_csv(f"Figures/flagging/bad_flagged_photons_{filter}_{focus}_{flux_mcrab_str}mCrab.csv", index=False)

# %%
# save the figure to PDF in "Figures" directory
figdir = "Figures/flagging"
fig.savefig(f"{figdir}/flagging_{filter}_{focus}_{flux_mcrab_str}mCrab.png", dpi=300)
fig.savefig(f"{figdir}/flagging_{filter}_{focus}_{flux_mcrab_str}mCrab.pdf", bbox_inches='tight')

# %%
print(ratio_badrecons)
print(ratio_badrecons_log)

# get index where ratio_badrecons is maximum
max_ratio_index = np.argmax(ratio_badrecons)
# get badrecons SIGNAL and missing energy
max_ratio_SIGNAL = all_badrecons_SIGNAL[max_ratio_index]
max_ratio_MISS0_ENERGY = all_badrecons_MISS0_ENERGY[max_ratio_index]
print(f"Maximum ratio of badrecons/missing energy: {max_ratio_SIGNAL:.3f} / {max_ratio_MISS0_ENERGY:.3f} = {ratio_badrecons[max_ratio_index]:.3f}")

# get index where ratio_badrecons is minimum
min_ratio_index = np.argmin(ratio_badrecons)
# get badrecons SIGNAL and missing energy
min_ratio_SIGNAL = all_badrecons_SIGNAL[min_ratio_index]
min_ratio_MISS0_ENERGY = all_badrecons_MISS0_ENERGY[min_ratio_index]
print(f"Minimum ratio of badrecons/missing energy: {min_ratio_SIGNAL:.3f} / {min_ratio_MISS0_ENERGY:.3f} = {ratio_badrecons[min_ratio_index]:.3f}")

# %%



