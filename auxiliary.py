import math
import os
import glob
from subprocess import run, PIPE, STDOUT

from astropy.io import fits
from astropy.table import Table

import numpy as np
import numpy.polynomial.polynomial as poly


verbose = 1
def vprint(*args, **kwargs):
    """
    Print function that can be turned on/off with the verbose variable.
    """
    if verbose > 0:
        print(*args, **kwargs)


def time_to_observe_n_pairs(count_rate, pairs_separation, npairs):
    """
    Calculate the time needed to observe a given number of pairs in a Poisson process.

    For this:

    d = pairs_separation
    ctr = count_rate

    1. calculate how many pairs of events you have in a Poisson process where the distance 
    between two events is less than a quantity (d)
        1.1 Determine the probability of a pair: the probability that the distance between 
        two consecutive events is less than (d) is:
             P(T < d) = 1 - exp(-ctr * d)

        1.2 Calculate the expected number of pairs: Having a time interval (T) and an event rate (\\lambda), 
        the expected number of events in that interval is (\\lambda T). 
        Each pair of consecutive events can be considered a potential pair. 
        Therefore, the expected number of pairs is approximately (\\lambda T - 1) 
        (since the first event has no previous event with which to form a pair).

        1.3 Multiply by the probability of a pair: The expected number of pairs where the distance between 
        events is less than (d) is the expected number of pairs multiplied by the probability that the 
        distance between two events is less than (d):

        E[pairs] = npairs = (ctr * T - 1) * (1 - exp(-ctr * d))
    2. Get the time needed to observe a given number of pairs:
        2.1 Solve for T in the equation above:
            ctr * T - 1 = npairs / (1 - exp(-ctr * d))
            T = ((npairs / (1 - exp(-ctr * d)) + 1)/ ctr
    """
    
    # calculate the time needed to observe npairs pairs
    time = ((npairs / (1 - math.exp(-count_rate * pairs_separation))) + 1) / count_rate
    return time

def get_max_photons(simfiles):
    """
    Get the maximum number of photons in the files by reading the NAXIS2 keyword.

    Parameters:
    simfiles (list of str): List of file paths to the FITS files to be read.
    
    Returns:
    int: The maximum number of photons found in the FITS files.
    """
    
    max_photons = 0
    for file in simfiles:
        with fits.open(file) as hdul:
            max_photons = max(max_photons, hdul[1].header['NAXIS2'])
    return max_photons

#Function to get polynomial fit to confidence intervals in the data (starting from coulumX and columnY arrays)
def get_polyfit_intervals_columns(columnX, columnY, nsigmas, order=2):
    """
    Get the polynomial fit to the confidence intervals in the data (reading from arrays).

    Parameters:
    columnX (array-like): 2D Array containing the X values for the diagnostic plot
            First index runs through simulated energies, second index runs through the number of photons
    columnY (array-like): 2D Array containing the Y values for the diagnostic plot
            First index runs through simulated energies, second index runs through the number of photons
    nsigmas (float): Number of sigmas to be used to define the confidence interval.
    order (int): Order of the polynomial fit to be applied to the data. Default is 2.
    
    Returns:
    dict: Dictionary containing the polynomial coefficients of numpy.polynomial.polynomial fit to the 
        confidence intervals in the data. The polynomial coefficients are stored in the 'top' and 'bottom' keys.
        To be used as: 
            import numpy.polynomial.polynomial as poly
            poly.polyval(x, poly_top_coeffs) and poly.polyval(x, poly_bottom_coeffs)
    """

    # get the maximum number of simulated energies
    nenergies = columnX.shape[0]
    assert columnY.shape[0] == nenergies, "columnX and columnY must have the same number of simulated energies"

    medianX = np.full((nenergies), np.nan)
    medianY = np.full((nenergies), np.nan)
    stdY = np.full((nenergies), np.nan)
    # calculate the median and standard deviation of the Y values for each energy
    for ie in range(nenergies):
        medianX[ie] = np.nanmedian(columnX[ie, :])
        medianY[ie] = np.nanmedian(columnY[ie, :])
        stdY[ie] = np.nanstd(columnY[ie, :])

    # fit polynomials to the boundaries of the 5-sigma confidence interval
    poly_top_coeffs = poly.polyfit(medianX, medianY+nsigmas*stdY, order)
    poly_bottom_coeffs = poly.polyfit(medianX, medianY-nsigmas*stdY, order)
    return {'top': poly_top_coeffs, 'bottom': poly_bottom_coeffs}

#Function to get polynomial fit to confidence intervals in the data (starting from simfiles)
def get_polyfit_intervals_simfiles(simfiles, simEnergies, columnX, columnY, nsigmas, order=2):
    """
    Get the polynomial fit to the confidence intervals in the data.

    Parameters:
    simfiles (list of str): List of file paths to the simulated monochromatic FITS files to be read.
    simEnergies (list of float): List of the energies of the simulated monochromatic files.
    columnX (str): Name of the column containing the X values for the diagnostic plot
    columnY (str): Name of the column containing the Y values for the diagnostic plot
    nsigmas (float): Number of sigmas to be used to define the confidence interval.
    order (int): Order of the polynomial fit to be applied to the data. Default is 2.
    
    Returns:
    dict: Dictionary containing the polynomial coefficients of numpy.polynomial.polynomial fit to the 
        confidence intervals in the data. The polynomial coefficients are stored in the 'top' and 'bottom' keys.
        To be used as: 
            import numpy.polynomial.polynomial as poly
            poly.polyval(x, poly_top_coeffs) and poly.polyval(x, poly_bottom_coeffs)
    """
    
    # check that the filenames in simfiles do exist
    for file in simfiles:
        if not os.path.exists(file):
            raise FileNotFoundError(f"File {file} not found.")
    
    # get the maximum number of photons in the files
    max_photons = get_max_photons(simfiles)

    # create numpy arrays to store the data
    # the first dimension is the energy, the second dimension is the number of photons  
    # inititalize the arrays with NaN
    columnX = np.full((len(simEnergies), max_photons), np.nan)
    columnY = np.full((len(simEnergies), max_photons), np.nan)
    medianX = np.full((len(simEnergies)), np.nan)
    medianY = np.full((len(simEnergies)), np.nan)
    stdY = np.full((len(simEnergies)), np.nan)

    for ie in range(len(simEnergies)):
        simE = simEnergies[ie]
        simfile = simfiles[ie]
        #print(f"Reading {simfile}")
        f = fits.open(simfile)
        # read the data and store 
        # store the array in the SIGNAL column in the second dimension of the SIGNAL array
        x_data = f[1].data[columnX]
        y_data = f[1].data[columnY]
        columnX[ie, :len(x_data)] = x_data
        columnY[ie, :len(y_data)] = y_data
        medianX[ie] = np.nanmedian(x_data)
        medianY[ie] = np.nanmedian(y_data)
        stdY[ie] = np.nanstd(y_data)
        f.close()

    # fit polynomials to the boundaries of confidence interval using get_polyfit_intervals_columns
    poly_dict = get_polyfit_intervals_columns(columnX, columnY, nsigmas, order)
    return poly_dict

def is_inside_conf_interval(xvalue, yvalue, poly_top_coeffs, poly_bottom_coeffs):
    """
    Check if a point (xvalue, yvalue) is inside a polynomial confidence interval defined 
    by the top and bottom polynomial coefficients.
    Parameters:
        xvalue (float): The x-coordinate of the point to check.
        yvalue (float): The y-coordinate of the point to check.
        poly_top_coeffs (list or array-like): Coefficients of the polynomial (numpy.polynomial.polynomial) 
                        defining the top boundary of the confidence interval.
                        To be used as: poly.polyval(x, poly_top_coeffs)
        poly_bottom_coeffs (list or array-like): Coefficients of the polynomial (numpy.polynomial.polynomial)
                        defining the bottom boundary of the confidence interval.
                        To be used as: poly.polyval(x, poly_bottom_coeffs)
    Returns:
        bool: True if the point is inside the confidence interval, False otherwise.
    """
    
    poly_top_xvalue = poly.polyval(xvalue, poly_top_coeffs)
    poly_bottom_xvalue = poly.polyval(xvalue, poly_bottom_coeffs)
    dist_top = yvalue - poly_top_xvalue
    dist_bottom = yvalue - poly_bottom_xvalue
    if dist_top < 0 and dist_bottom > 0:
        return True
    else:
        return False

def get_sirena_info(sirena_file, impact_file):
    """
    Get the SIRENA information for all the reconstructed photons (including PROBABLE PH_ID)
          
    Parameters:
        sirena_file (str): The path to the SIRENA file to be read.
        impact_file (srt): The path to the impact file to be read
        
    Returns:
        dict of dict: A list of dictionaries containing the SIRENA information (TIME, SIGNAL, ELOWRES, AVG4SD_PROBPHID) 
        for all the reconstructed photons.
    """

    # open the SIRENA file
    with fits.open(sirena_file) as hdul:
        nrecons = hdul[1].header['NAXIS2']
        # create a structure to store the SIRENA information
        sirena_info = {"TIME": [], "SIGNAL": [], "ELOWRES": [], "AVG4SD": [], "GRADE1": [], "GRADE2": [], "PROBPHID": []}
        
        # read the data and store
        data = hdul[1].data
        PH_ID = data['PH_ID']
        TIME = data['TIME']
        SIGNAL = data['SIGNAL']
        ELOWRES = data['ELOWRES']
        AVG4SD = data['AVG4SD']
        GRADE1 = data['GRADE1']
        GRADE2 = data['GRADE2']
        # check if the column PROBPHID exists in the SIRENA file
        colPROBPHID = 'PROBPHID' in data.columns.names
        if not colPROBPHID:
            # open the impact file
            with fits.open(impact_file) as hdul_impact:
                data_impact = hdul_impact[1].data
                # get the PROBABLE PH_ID and arrival time
                ph_id_imp = data_impact['PH_ID'].copy()
                time_imp = data_impact['TIME'].copy()
                
        for irow in range(len(SIGNAL)):
            time_irow = TIME[irow]
            signal_irow = SIGNAL[irow]
            elowres_irow = ELOWRES[irow]
            avg4sd_irow  = AVG4SD[irow]
            grade1_irow = GRADE1[irow]
            grade2_irow = GRADE2[irow]

            # get possible IDs of the photons in the record if PROBPHID is not present
            # check if column PROBPHID exists in the SIRENA file
            # check if the column exists
            if not colPROBPHID:
                ph_nonzero_sequence = PH_ID[irow][np.nonzero(PH_ID[irow])]
                number_of_ph_zeros = len(PH_ID[irow]) - len(ph_nonzero_sequence)
                # if number of values == 0 in PH_ID[irow] is 0, then max number of detections reached: some photons may be not registered in PH_ID
                if number_of_ph_zeros == 0:
                    print(f"*** WARNING: maximum number of photons reached in row {irow}: some photons may have not been registered in PH_ID")

                # if number of values != 0 in PH_ID[irow] is 1, then it is a single photon
                if len(ph_nonzero_sequence) == 1:
                    # single photon
                    ph_id_irow = PH_ID[irow][0]
                else:
                    # more than one photon in the record: check corresponding time in impact file
                    min_time_diff = float('inf')
                    for ph_id in ph_nonzero_sequence:
                        # get the time of the photon in the impact file: same PH_ID 
                        index_match = np.where((ph_id_imp == ph_id))
                        # check if there is a match
                        if len(index_match[0]) == 0:
                            raise ValueError(f"PH_ID {ph_id} not found in impact file")
                        time_ph_piximpact = time_imp[index_match]
                        time_diff = abs(time_ph_piximpact-time_irow)
                        if time_diff < min_time_diff:
                            min_time_diff = time_diff
                            ph_id_irow = ph_id
            else:
                ph_id_irow = data['PROBPHID'][irow]
                # check if the PROBPHID is not zero
                if ph_id_irow == 0:
                   raise ValueError(f"PROBPHID is zero in row {irow}: no photon found") 
            # save the SIRENA information for the event
            sirena_info['TIME'].append(time_irow)
            sirena_info['SIGNAL'].append(signal_irow)
            sirena_info['ELOWRES'].append(elowres_irow)
            sirena_info['AVG4SD'].append(avg4sd_irow)
            sirena_info['GRADE1'].append(grade1_irow)
            sirena_info['GRADE2'].append(grade2_irow)
            sirena_info['PROBPHID'].append(ph_id_irow)

    return sirena_info

def get_missing_for_bad_recons(phid_bad_recons, possible_phids_missing, impact_file):
    """
    Get the missing photons for the bad reconstruction photon.

    Parameters:
        phid_bad_recons (int): PH_ID of the bad reconstructed photon.
        possible_phids_missing (list of int): List of possible IDs of the missing photons.
        impact_file (str): The path to the impact file to be read.
        
    Returns:
        int: PH_ID of the missing photon
    """
    
    # open the impact file
    with fits.open(impact_file) as hdul_impact:
        data_impact = hdul_impact[1].data
        # get the PROBABLE PH_ID and arrival time
        ph_id_imp = data_impact['PH_ID'].copy()
        time_imp = data_impact['TIME'].copy()

    # get the time of the bad reconstruction photon in the impact file: same PH_ID
    index_match = np.where((ph_id_imp == phid_bad_recons))
    # check if there is a match
    if len(index_match[0]) == 0:
        raise ValueError(f"PH_ID {phid_bad_recons} not found in impact file")
    time_imp_id_bad_recons = time_imp[index_match]
    
    min_time_diff = float('inf')
    ph_id_missing = 0
    # get the missing photon for the bad reconstruction photon (closest in time)
    for id_missing in possible_phids_missing:
        # get the time of the missing photon in the impact file: same PH_ID
        index_match = np.where((ph_id_imp == id_missing))
        # check if there is a match
        if len(index_match[0]) == 0:
            raise ValueError(f"PH_ID {id_missing} not found in impact file")
        time_imp_id_missing = time_imp[index_match]
        # get the time difference
        time_diff = abs(time_imp_id_missing - time_imp_id_bad_recons)
        # check if it is the minimum time difference
        if time_diff < min_time_diff:
            min_time_diff = time_diff
            ph_id_missing = id_missing
    return ph_id_missing
        
def get_sirena_fake_events(sirena_file=""):
    """
    Get the SIRENA information for all the fake events
          
    Parameters:
        sirena_file (str): The path to the SIRENA file to be read.
        
    Returns:
        Astropy Table: table containing the SIRENA information (TIME, SIGNAL, SIRENA_ROW, CLOSE_PHID)
        for all the fake events.
        The Table contains the following columns:
        - TIME: Time of the event
        - SIGNAL: Signal of the event
        - SIRENA_ROW: Row number of the event in the SIRENA file
        - CLOSE_PHID: PH_ID of the closest simulated photon (from the PH_ID column)
    """
    import pandas as pd
    from astropy.table import Table
    # open the SIRENA file
    with fits.open(sirena_file) as hdul:
        nrecons = hdul[1].header['NAXIS2']
        # get the data from the SIRENA file
        data = hdul[1].data 

        # check if there is a PROBPHID column
        if 'PROBPHID' in hdul[1].data.columns.names:
            PROBPHID = data['PROBPHID']
            # get number of unique PH_IDs in PROBPHID
            unique_phids = np.unique(PROBPHID)
            if nrecons > len(unique_phids): #fake pulses
                # get the TIME and SIGNAL columns
                TIME = data['TIME']
                SIGNAL = data['SIGNAL']
                # identify the duplicated values in PROBPHID column
                duplicated_phids = [phid for phid in unique_phids if np.sum(PROBPHID == phid) > 1]
                # identify the PROBPHID rows where the PH_ID is duplicated
                duplicated_rows = [irow for irow in range(nrecons) if PROBPHID[irow] in duplicated_phids]
                # create an astropy table to store the SIRENA information
                sirena_info = Table(names=('SIRENA_ROW', 'PHID', 'TIME', 'SIGNAL'), 
                                    dtype=('i4', 'i4', 'f8', 'f8'))
                for irow in duplicated_rows: 
                    time_irow = TIME[irow]
                    signal_irow = SIGNAL[irow]
                    sirena_info.add_row([irow+1, PROBPHID[irow], time_irow, signal_irow])
        else:
            # create an astropy table to store the SIRENA information
            sirena_info = Table(names=('SIRENA_ROW', 'PHID', 'TIME', 'SIGNAL'), 
                                dtype=('i4', 'i4', 'f8', 'f8'))
            # get the PH_ID column
            PH_ID = data['PH_ID']
            # get the non-zero values in PH_ID
            ph_nonzero_sequence = PH_ID[np.nonzero(PH_ID)]
            if len(ph_nonzero_sequence) == len(PH_ID): # all values are non-zero
                raise ValueError("All values in PH_ID are non-zero: maximum number of photons stored - check xifusim file")
            # get the unique values in ph_nonzero_sequence
            unique_phids = np.unique(ph_nonzero_sequence)
            nunique_phids = len(unique_phids)

            if nrecons > nunique_phids: #fake pulses
                # get the TIME and SIGNAL columns
                TIME = data['TIME']
                SIGNAL = data['SIGNAL']
                # get the rows where PH_ID is identical
                _, idx, counts = np.unique(PH_ID, axis=0, return_index=True, return_counts=True)
                same_PHID_rows = [np.where((PH_ID == PH_ID[i]).all(axis=1))[0] for i in idx[counts > 1]]
                for group in same_PHID_rows:
                    ph_nonzero_sequence = PH_ID[group[0]][np.nonzero(PH_ID[group[0]])]
                    if len(ph_nonzero_sequence) < len(group): # fake pulses in the group
                        # add the rows to the table
                        for irow in group:
                            time_irow = TIME[irow]
                            signal_irow = SIGNAL[irow]
                            phid = PH_ID[irow]
                            # add the row to the table
                            sirena_info.add_row([irow+1, ph_nonzero_sequence, time_irow, signal_irow])
    #print(sirena_info)
    return sirena_info
    