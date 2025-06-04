# xpileup
Analysis of TES piled up events from simulations

# Notebooks:

- simulate_source.ipynb:
========================
Given a flux, model and some additional simulation parameters, it simulates the photons impacting in the X_IFU pixels and creates two CSV files:
  * 00_info_<filt>_sim<number>_missing.csv with info about pixels, bad reconstructed photon detected by SIRENA (energy, PH_ID, GRADE1, GRADE2) in multiple arrivals, missing photons in multiple arrival (PH_IDs, energies).
  * an entry in info_<filt>_<focus>_global_flux.csv for each of the above simulations

- plot_summary.ipynb:
=====================
It makes histograms of mean fractions of lost photons using all the simulations. It performs normality tests to check whether the exposure time selected produces enough pairs so that taking a mean and a stddev has sense

- missing_analysis.ipynb:
========================
Using CSV info analyses the missing and badreconstructed photons information and makes two plots:
  * histogram of separations distribution + histogram of energies distribution
  * 2D histogram (image) of energies in the problematic pairs (badrecons//missing)

flag_multipulse.ipynb
======================
Creates contours (5sigma) around the location of single pulses in the ELOWRES-ERENCONS space.
Then checks location of badrecons pulses in this map and checks whether they could be flagged.
It also checks if single pulses could be miss-identified.

plot_fractions.ipynb
====================
Reads global summary files and plots mean fractions of missing (and badrecons) photons.


## 12/05/2025

SIRENA files are redone using  sirena branch "nomodelsubtractio_smoothDerivative": 4 samples for the derivative and no model subtraction (during detection)
Previous SIRENA files are saved in "_2" dirs
Previous CSV (global) files saved in sirena_2deriv_4SD (derivative w/ 2 points + 4 SamplesDown)
Added distance between BADRECONS and first missing (MISS0) in simulate_source.ipynb


# 14/05/2025
Using SamplesDown=4 SIRENA loses more photons than with the previous version of the derivative.
We try now with SamplesDown=1
Save results in individual dirs sirena_4deriv_1SD (also the global CSV)

# 21/05/2025
Run sirena w/ 4 points for the derivative but 2 samples down (with 1 SD it invented some events)

#### 26/05/2025
I discovered that if single pixels are not simulated by xifusim SIRENA reconstructions could be using larger filters that those really possible AND secondary pulses may not be classified as such.
In addition, post-modification of GRADE2 based on piximpact was causing incorrect GRADE2 (very small) when the bad-recons is the second and missing is the first one (sirena detection closer to the second pulse). Reassignation of GRADE2 was giving a value of 1 or 2 samples (distance between both members of the couple).
I correct simulate_source.ipynb: add single close pulses that coudl affect GRADE1 or GRADe2 of reconstruction + do not change GRADE2
Move all files to folder "previous_singles_simulation"

# 30/05/2025
Edoardo has produced a new configuration file for xifusim simulations (v5_20250621).
He confirms that the previous one (with fatser pulses) can be wrong (v3_20240917).
Simulations done so far have been moved to folder: v3_20240917 (including previous_singles_simulations)
A new folder has been created to store new simulations. As pulses are now slower, we simulate now couples separated by 200 samples (instead of 100 samples as before). Other parameters (apart from the configuration file) are kept the same.
