import math

verbose = 1
def vprint(*args, **kwargs):
    """
    Print function that can be turned on/off with the verbose variable.
    """
    if verbose > 0:
        print(*args, **kwargs)


def time_to_observe_n_counts(rate, time_interval, n):
    """
    Calculate the expected time to observe n counts in a given time interval under Poisson statistics.

    Parameters:
        rate (float): The count rate of the source (counts per second).
        time_interval (float): The time interval (in seconds) during which we want n counts.
        n (int): The number of counts to observe.

    Returns:
        float: The expected observation time (in seconds) to see n counts in the given time interval.
    """
    if n <= 0:
        raise ValueError("The number of counts (n) must be a positive integer.")

    # Calculate the expected number of counts in the time interval
    mu = rate * time_interval

    # Calculate the probability of observing n counts using the Poisson formula
    p_n = (mu ** n * math.exp(-mu)) / math.factorial(n)

    # Calculate the rate of observing n counts
    rate_of_n_counts = p_n / time_interval

    # Return the expected observation time (reciprocal of the rate)
    return 1 / rate_of_n_counts

import os
import glob
from subprocess import run, PIPE, STDOUT

from astropy.io import fits

import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt


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

def get_sirena_info(ph_id, arrival_time, sirena_file):
    """
    Get the SIRENA information from the SIRENA file for a given PH_ID and TIME.
          HACER QUE PUEDA LEER UNA LISTA DE FOTONES Y DEVOLVER UNA LISTA DE DICCIONARIOS CON LA INFO DE SIRENA!!!!!!!!!!!!!!!!
    Parameters:
        ph_id (int): The PH_ID of the event to search for in the SIRENA file.
        arrival_time (float): The arrival time of the event (from piximpact file for example) to search for in the SIRENA file.
        sirena_file (str): The path to the SIRENA file to be read.
        
    Returns:
        dict: A dictionary containing the SIRENA information (TIME, SIGNAL, ELOWRES, AVG4SD) for the event with the specified PH_ID and TIME.
    """
    
    with fits.open(sirena_file) as hdul:
        data = hdul[1].data
        PH_ID = data['PH_ID']
        TIME = data['TIME']
        SIGNAL = data['SIGNAL']
        ELOWRES = data['ELOWRES']
        AVG4SD = data['AVG4SD']
        GRADE1 = data['GRADE1']
        GRADE2 = data['GRADE2']
        for irow in range(len(PH_ID)):
            # find the rows where the PH_ID column matches the ph_id
            if not ph_id in PH_ID[irow]:
                continue
            irows_same_PH_ID = np.where((PH_ID == PH_ID[irow]).all(axis=1))[0]
        # get the closest arrival time to the specified arrival_time among the rows with the same PH_ID
        time_diff = np.abs(TIME[irows_same_PH_ID] - arrival_time)
        closest_time_index = np.argmin(time_diff)
        closest_time_row = irows_same_PH_ID[closest_time_index]
        # get the SIRENA information for the event
        sirena_info = {'SIGNAL': SIGNAL[closest_time_row], 'TIME': TIME[closest_time_row], 
                       'ELOWRES': ELOWRES[closest_time_row], 'AVG4SD': AVG4SD[closest_time_row],
                       'GRADE1': GRADE1[closest_time_row], 'GRADE2': GRADE2[closest_time_row]}
    return sirena_info

        