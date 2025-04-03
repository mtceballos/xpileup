import math
import os
import glob
from subprocess import run, PIPE, STDOUT

from astropy.io import fits

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

        1.2 Calculate the expected number of pairs: Having a time interval (T) and an event rate (\lambda), 
        the expected number of events in that interval is (\lambda T). 
        Each pair of consecutive events can be considered a potential pair. 
        Therefore, the expected number of pairs is approximately (\lambda T - 1) 
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
        impact_file (srt): The path to the impact file to be read.
        
    Returns:
        dict of dict: A list of dictionaries containing the SIRENA information (TIME, SIGNAL, ELOWRES, AVG4SD_PROBPHID) 
        for all the reconstructed photons.
    """

    # open the impact file
    with fits.open(impact_file) as hdul_impact:
        data_impact = hdul_impact[1].data
        # get the PROBABLE PH_ID and arrival time
        ph_id_imp = data_impact['PH_ID'].copy()
        time_imp = data_impact['TIME'].copy()
        pixel_imp = data_impact['PIXEL'].copy()

    # open the SIRENA file
    with fits.open(sirena_file) as hdul:
        nrecons = hdul[1].header['NAXIS2']
        #initialize a list of dictionaries to store the SIRENA information
        sirena_info = [{} for _ in range(nrecons)]
        # read the data and store
        data = hdul[1].data
        PH_ID = data['PH_ID'].copy()
        TIME = data['TIME'].copy()
        SIGNAL = data['SIGNAL'].copy()
        ELOWRES = data['ELOWRES'].copy()
        AVG4SD = data['AVG4SD'].copy()
        GRADE1 = data['GRADE1'].copy()
        GRADE2 = data['GRADE2'].copy()
        PIXEL = data['PIXEL'].copy()
    
        for irow in range(len(SIGNAL)):
            time_irow = TIME[irow]
            signal_irow = SIGNAL[irow]
            elowres_irow = ELOWRES[irow]
            avg4sd_irow  = AVG4SD[irow]
            grade1_irow = GRADE1[irow]
            grade2_irow = GRADE2[irow]
            pixel_irow = PIXEL[irow]

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
                    # get the time of the photon in the impact file: same PH_ID and same PIXEL
                    time_ph_piximpact = time_imp[ph_id_imp == ph_id and pixel_imp == pixel_irow]
                    time_diff = abs(time_ph_piximpact-time_irow)
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        ph_id_irow = ph_id
            
            # save the SIRENA information for the event
            sirena_info[irow] = {
                'SIGNAL': signal_irow, 'TIME': time_irow, 
                'ELOWRES': elowres_irow, 'AVG4SD': avg4sd_irow,
                'GRADE1': grade1_irow, 'GRADE2': grade2_irow, 
                'PH_ID': ph_id_irow, 'PIXEL': pixel_irow}
    return sirena_info
