import pandas as pd
import numpy as np
from scipy.signal import find_peaks

from scipy.interpolate import interp1d
from hapi import absorptionCoefficient_Lorentz, fetch

from scipy.signal import find_peaks, peak_prominences, butter, filtfilt

from scipy import signal

#FIBER BRAGG GRATING TRANSMISSION SPECTRUM GENERATOR
def preprocess(filename, c):

    """
            File preprocessing to remove comments in the header. Returns a 2-dimensional numpy array, wavelength first column and absorbance or transmittance as second column.

            :param str filename: File path

            :param str c: Can be either 'absorbance' or 'transmittance'
    """

    lines = [line.strip() for line in open(filename)]
    skip = sum([1 for line in lines if line[0] == '#'])

    if c == 'absorbance':
        lines = [line[1:] for line in lines]
        separator = '    '
    elif c == 'transmittance':
        separator = ' '
    else:
        print('Invalid data type. Must be "absorbance" or "transmittance"')

    data = pd.read_csv(filename, skiprows = skip, header = 0, sep = separator).to_numpy()
    return data


def detect_peaks(fractional_transmittance, height_peaks = -0.95, threshold_peaks = -0.95, distance_peaks = 50):

    """
            Detect peaks in transmittance spectrum.

            :param numpy.ndarray fractional_transmittance: fractional transmittance

            :param numpy.ndarray height_peaks: Required height of peaks

            :param float threshold_peaks: Required threshold of peaks, the vertical distance to its neighboring samples.

            :param float distance_peaks: Required minimal horizontal distance (>= 1) in samples between neighbouring peaks.
    """

    peaks, properties = find_peaks(-fractional_transmittance, height = -height_peaks, threshold = -threshold_peaks, distance = distance_peaks)
    return peaks, properties


def generate_gaussian(x, mu, sigma, amp):

    """
            Generates a gaussian profile.

            :param numpy.ndarray x: wavelengths in nm

            :param numpy.ndarray mu: wavelength of FBG transmission peak in nm

            :param float sigma: standard deviation

            :param numpy.ndarray amp: FBG transmission peak height
    """

    return amp * (1 / (sigma*np.sqrt(2 * np.pi))) * np.exp(-(x - mu)**2 / (2*sigma**2) )


def simulate_transmittance(wavelength, height_peaks, peaks, sigma, k = 0, strain = 0):
    
    """
            Simulates the transmission spectrum of FBG depending on its strain.

            :param numpy.ndarray wavelength: wavelengths in nm

            :param numpy.ndarray fractional_transmittance: fractional transmittance

            :param numpy.ndarray peaks: index of transmission notches

            :param float sigma: standard deviation

            :param float k: electro optic parameter

            :param numpy.ndarray strain: FBG strain in microstrain

            :param int n: number of samples between minimum and maximum wavelength
    """
    n = int((wavelength.max()-wavelength.min())/0.0025)
    reflectance = np.zeros(n)

    #Wavelength of peaks with strain
    fbg_peaks = wavelength[peaks] * ( 1 + k * strain)

    fbg_wavelength = np.linspace(wavelength.min(), wavelength.max(), n)

    for i in range(len(peaks)):
        reflectance += generate_gaussian(fbg_wavelength, fbg_peaks[i], sigma, height_peaks[i])
        
    reflectance *= height_peaks.max()/reflectance.max()
    fbg_transmittance = 1 - reflectance

    return fbg_wavelength, fbg_transmittance, fbg_peaks


def shift_peaks(wavelengths, peaks, peak_center, k, strain_step):

    #There are 400 line per nm
    for i in range(len(peaks)):
        delta_lambda = k * wavelengths[peaks[i]] * strain_step * (peak_center - i)
        peaks[i] += int(delta_lambda * 400)
           
    return peaks





#GAS MIXTURE TRANSMISSION SPECTRUM GENERATOR
dict_mol = {'Methane': [6, 3*1e-6]}
dict_env = {'T':293, 'p':1}


def fetch_molecules_spectrum(molecules, wavenumber_min, wavenumber_max):

    """
            Fetches the spectrum of molecules on the HITRAN database.

            :param list molecules: list of molecules

            :param float wavenumber_min: minimum wavenumber

            :param float wavenumber_max: maximum wavenumber
    """

    for i in range(len(molecules)):
        fetch(molecules[i], dict_mol[molecules[i]][0], 1, wavenumber_min, wavenumber_max)
    return


def set_concentration_dictionary(molecules):

    """
            Creates a dictionary {'molecule' : concentration, ...}, and the remaining volume is filled by air.

            :param list molecules: list of molecules

    """

    dict_molC = { molecules[i]:dict_mol[molecules[i]][1] for i in range(len(molecules))}
    dict_molC['Air'] = 1 - sum([dict_mol[molecules[i]][1] for i in range(len(molecules))] )
    return


def get_spectrum(molecules, path_length, wavenumber_min, wavenumber_max):

    """
            Generate transmission spectrum for given gas mixture.

            :param list molecules: list of molecules

            :param float wavenumber_min: minimum wavenumber in cm^-1

            :param float wavenumber_max: maximum wavenumber in cm^-1
    """
    
    #Fethces spectrum of molecules 
    fetch_molecules_spectrum(molecules, wavenumber_min, wavenumber_max)

    #Creates molecule : concentration dictionary
    dict_molC = set_concentration_dictionary(molecules)

    #Generates absorption cross section in cm^-1
    nu, abs_coef = absorptionCoefficient_Lorentz(SourceTables=molecules, Diluent=dict_molC, Environment=dict_env, HITRAN_units = False)

    #Convert wavenumber (cm^-1) to wavelength (nm)
    wavelengths = 10000000/nu

    #Compute fractional transmittance of gas mixture
    transmittance = np.exp(- abs_coef * path_length * 3 * 1e-6)

    #Creates an array of equally spaced wavelengths 
    f=interp1d(wavelengths, transmittance)
    wavelengths_lin = np.linspace(wavelengths.min(), wavelengths.max(), int((wavelengths.max()-wavelengths.min())/0.0025))
    transmittance_lin = f(wavelengths_lin)

    return wavelengths_lin, transmittance_lin






#LOCK IN AMPLIFICATION
def lock_in(t, frequency_sine, frequency_cutoff, frequency_sampling, filter_order, V_out_photodetector):

    """
            2f lock-in amplification of signal.

            :param numpy.ndarray t: time array

            :param float frequency_sine: frequency of the sine wave used for modulation in Hz

            :param float frequency_cutoff: Cut-off frequency of the low-pass filter

            :param float frequency_sampling: sampling frequency of the digital lock-in in Hz

            :param int filter_order: order of of the low-pass filter

            :param numpy.ndarray V_out_photodetector: Output voltage of photodetector
    """

    #reference signals
    sin_2f_ref = np.sin(2 * np.pi * 2 * frequency_sine * t)
    cos_2f_ref = np.cos(2 * np.pi * 2 * frequency_sine * t)

    #sin_1f_ref = np.sin(2 * np.pi * frequency_sine * t)
    #cos_1f_ref = np.cos(2 * np.pi * frequency_sine * t)

    X_2f = V_out_photodetector * cos_2f_ref
    Y_2f = V_out_photodetector * sin_2f_ref

    #X_1f = V_out_photodetector * cos_1f_ref
    #Y_1f = V_out_photodetector * sin_1f_ref

    # Design the low-pass filter
    nyq = 0.5 * frequency_sampling
    normal_cutoff = frequency_cutoff / nyq
    b, a = butter(filter_order, normal_cutoff, btype='lowpass')

    # Apply the filter
    X_2f_filtered = filtfilt(b, a, X_2f)
    Y_2f_filtered = filtfilt(b, a, Y_2f)

    #X_1f_filtered = filtfilt(b, a, X_1f)
    #Y_1f_filtered = filtfilt(b, a, Y_1f)

    S_2f_filtered = np.sqrt(X_2f_filtered ** 2 + Y_2f_filtered **2)*100

    # Design the low-pass filter
    nyq = 0.5 * frequency_sampling
    normal_cutoff1 = 13 / nyq
    normal_cutoff2 = 17 / nyq
    c, d = butter(filter_order, [normal_cutoff1, normal_cutoff2], btype='bandstop')
    S_2f_filtered_filtered = filtfilt(c, d, S_2f_filtered)

    return S_2f_filtered_filtered






#WAVEFORM GENERATOR
def triangle(t, f, Vpp, Offset):
    return Offset +  (Vpp/2) * signal.sawtooth(2 * np.pi * f * t, 0.5)
    
def sine(t, f, Vpp, Offset):
    return Offset +  (Vpp/2) * np.sin(2 * np.pi * f * t)




