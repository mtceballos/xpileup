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
