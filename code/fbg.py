import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# test

def psg_to_numpy(filepath, verbose=True):

    """
    Takes a psg_trn.txt file generated from the NASA-GSF Planetary Spectrum Gemerator (PSG, Villanueva et al. 2018, 2022, https://psg.gsfc.nasa.gov/) and converts the data into a numpy array. The first column is the wavelength, the second column is the total fractional transmittence and the following columns correspond to each gas species. 
    

        **filepath: string**
            Path to the psg file.

        **verbose: boolean, (default=True)**
            If True, displays the metadata information in the file.

        .. seealso::
            :ref:`Using NASA PSG <psg_walkthrough>`

    """
    with open(filepath, "r") as f:
        file = f.read().strip()
        file = file.split('\n')

    skip = 0
    for line in file:
        if line[0] == '#':
            if verbose:
                print(line)
            skip += 1

    data = pd.read_csv(filepath, sep=' ', skiprows=skip, header=None).to_numpy()

    return data


def locate_peaks(frac_transmittence, threshold=0.99):
    """
    Takes an array of fractional transmittence and returns the index of the downard peaks. Each downward peak is lower than the threshold value.

        **frac_transmittence: 1D numpy array**
            Stores the fractional transmittence corresponding to a given wavelength. These values should be between 0 and 1.

        **threshold: float, (default=0.99)**
            A downward peak will only register if it dips below the specified threshold.
        
        **return: 1D numpy array**
            Containing the index of the indentified peaks.
    """

    peaks, properties = find_peaks(-frac_transmittence, height=-threshold)

    return peaks

def gaussian(x, sigma, mu, amp):
    """
    Generates a Gaussian function on the given domain at mu with standard deviation sigma and relative amplitude amp.

    **x: 1D numpy array**
        Domain of the function.
    **sigma: float**
        Standard deviation of the function.
    **mu: float**
        Mean of the function.
    **amp: float**
        Relative height of the function.
    """
    return amp*1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x - mu)**2/(2*sigma**2))

def lorentzian(x, sigma, mu, amp):
    """
    Generates a Lorentzian function on the given domain at mu with standard deviation sigma and relative amplitude amp.

    **x: 1D numpy array**
        Domain of the function.
    **sigma: float**
        Standard deviation of the function.
    **mu: float**
        Mean of the function.
    **amp: float**
        Relative height of the function.
    """
    return (amp/np.pi) * (sigma/((x-mu)**2 + sigma**2))

def generate_spectrum(data, peaks, n=None, sigma=1e-5, type='Gaussian'):
    """
    Models a transimittence spectrum based on the given peaks.

    **data: ND numpy array**
        This should match the format of the output of :func:`psg_to_numpy`.

    **peaks: 1D numpy array**
        This should match the format of the output of :func:`locate_peaks`.

    **n: integer**
        The number of samples along the spectrum.

    **sigma: float (default=1e-5)**
        Width of each peak.
    
    **type: string (default='Gaussian')**
        The type of function used to model each peak. Options are 'Gaussian' or 'Lorentzian'.
    """
    locs = data[peaks, 0]
    heights = 1 - data[peaks, 1]

    if n is None:
        n = len(data)

    wavelengths = np.linspace(data[:, 0].min(), data[:, 0].max(), n)
    transmittence = np.ones(n)

    if type == 'Gaussian':
        for i in range(len(peaks)):
            transmittence += gaussian(wavelengths, sigma=sigma, mu=locs[i], amp=heights[i])
    elif type == 'Lorentzian':
        for i in range(len(peaks)):
            transmittence += gaussian(wavelengths, sigma=sigma, mu=locs[i], amp=heights[i])
    else:
        print('Error: you need to specify a type of function to model the peak')

    transmittence *= heights.max()/transmittence.max()
    transmittence = 1 - transmittence

    return wavelengths, transmittence

