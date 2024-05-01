# Funcs.py

# This python script contains the functions used to simulate the straining of a Fibre Bragg Grating. 
  
# This script goes hand in hand with scratch_SimulatingCode_FBG.ipynb and newSimulatingCode_FBG.ipynb and 
# those files cannot be used without this script


# Written by: Samhita S Sodhi


################################################ import

import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
from matplotlib.pyplot import figure # type: ignore
import numpy as np
from astropy import units as u
from astropy import constants as const
from astropy.modeling.models import BlackBody
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

################################################ test

def special_sum(a, b=4):
    return a + b

################################################
 
def input_spectra(filepath, wavelengthcol, datacol, separation, isheader = None, wavelengthunits = "um", dataunits = "fractional"):
    """
    This function loads in the datafile for the spectral information and assigns values to the wavelength and spectral data.
    
    Parameters include:
        - filepath                  : requires the path of the datafile contained in "", the file must contain at least one column 
                                      for wavelength given in nanometers (nm) or micrometres (um) and one column for spectral line 
                                      intensity in decibels or fractional values.  
        - isheader                  : parameter takes in whether there is a header in the file. None means there is no header present.
        - wavelengthcol             : user inputs which column consists of the wavelength values.
        - datacol                   : user inputs which column consists of the spectra data values.
        - separation                : parameter takes in what the separation between the variables within the file
                                      e.g. "\t" means values separated by a tab.
        - wavelengthunits           : function uses data either micrometres "um" or nanometres "nm", default assumes wavelength is 
                                      in "um" else user must input the units, i.e., "nm".
        - dataunits                 : function uses data either as a fractional value or dB, default assumes units of spectral data 
                                      is fractional else user must input the units, "dB".
    """

    # ~~~~~ Inputs the data file
    gas_raw_data = pd.read_csv(filepath, header = isheader, sep = separation).values #stores the file into a variable 
    gas_wavelength = (gas_raw_data[:,wavelengthcol]).astype(float) #variable stores the wavelength range
    gas_line_data = (gas_raw_data[:,datacol]).astype(float) #variable stores the absorption line data for each line 
        
    if wavelengthunits == "um":
        gas_wavelength = gas_wavelength #leaves the wavelength units in micrometres (um)
    elif wavelengthunits == "nm": 
        gas_wavelength = (gas_wavelength/1000.0) #converts the wavelength units into micrometres (um)
    else:
        print("Please input the wavelength as um or nm")  

    if dataunits == "fractional": 
        gas_line_data = gas_line_data #leaves the data in terms of percentage
    elif dataunits == "dB":
        gas_line_data = (10**(gas_line_data/10))  #converts from decibels to a fractional value between 0 and 1
    else: 
        print("Please input dB or percentage")
            
    return gas_wavelength, gas_line_data


################################################ Plotting


def plot_spectra(xaxis, yaxis, title = 'Transmittance spectrum', xaxislabel = 'Wavelength (um)', yaxislabel = 'Transmittance', xlimits = None, ylimits = None):
   """
   This function provides a plotting facility as a simple line graph for any of the data. 

   The parameters include: 
      - xaxis        : loads in any variable as the x-axis.  
      - yaxis        : loads in any variable as the y-axis.  
      - title        : an input is required for the title of the plot, default is 'Transmittance Spectrum'
      - xaxislabel   : an input for the x-axis is required, default is 'Wavelength (um)'
      - yaxislabel   : an input for the y-axis is required, default is 'Transmittance' 
      - xlimits      : user has a choice to input a range of x values (wavelength) i.e.,  [ , ] for the plot else default is complete 
                       range of values 
      - ylimits      : user has a choice to input a range of y values (Transmitted Intensity) i.e.,  [ , ] for the plot else default 
                       is complete range of values
   """

   if xlimits is not None or ylimits is not None: 
      # ~~~~~ Plot with custom limits
      plt.figure(figsize=(15, 4))
      plt.plot(xaxis, yaxis)
      plt.xlim(xlimits)
      plt.ylim(ylimits)
      plt.title(title)
      plt.xlabel(xaxislabel)
      plt.ylabel(yaxislabel)
      plt.show()
   else: 
      # ~~~~~ Plot without custom limits
      plt.figure(figsize=(15, 4))
      plt.plot(xaxis, yaxis)
      plt.title(title)
      plt.xlabel(xaxislabel)
      plt.ylabel(yaxislabel)
      plt.show()

################################################ Fit models


def fit_curves(spectra_wav, spectra_data, detection_height, model = 'lorentzian', xlimits = None, ylimits = None):
    """
    This function first detects peaks for the spectral data by reversing the values, and then plots a model - either Lorentzian 
    or a Gaussian - for each peak of the input gas data whilst converting it back to the default spectrum appearance. 
    
    The parameters include: 
      - spectra_wav     : loads in the wavelengths 
      - spectra_data    : loads in the spectral data  
      - detection_height: input a value for the height of a spectral line to classify as a peak i.e., a value of 0.01 means 
                          any spectral line smaller than this value will not be classified as a peak. Please note: overfitting 
                          or underfitting may occur if the value inputted is too large or too small
      - model           : choose to either fit a 'gaussian' or a 'lorentzian' curve to the data, default is a lorentzian model
      - xlimits         : user has a choice to input a range of x values (wavelength) i.e.,  [ , ] for the plot else default is complete range of values
      - ylimits         : user has a choice to input a range of y values (Transmitted Intensity) i.e.,  [ , ] for the plot else default is complete range of values
   """
    
    # ~~~~~~~~ DETECTING PEAKS ~~~~~~~~
    x_values = np.array(spectra_wav)
    y_values = np.array(1.0 - spectra_data) # flipping the graph so peak detection can occur

    peaks, properties = find_peaks(y_values, height= detection_height, width = 0) #properties include, peak heights, peak widths, etc. 
     
    # PLOTS THE SPECTRA AS WELL AS THE DETECTED PEAKS SO USER CAN VISUALISE THE COMPUTATION
    if xlimits is not None or ylimits is not None: 
        # Plot with custom limits
        figure(figsize=(15, 4))
        plt.plot(x_values, y_values)
        plt.plot(x_values[peaks], y_values[peaks], "x")
        plt.xlim(xlimits)
        plt.ylim(ylimits)
        plt.xlabel('Wavelength (um)')
        plt.ylabel('Absorption')
        plt.title('Visualise detected peaks')
        plt.show()
    else:
        # Plot without custom limits
        figure(figsize=(15, 4))
        plt.plot(x_values, y_values)
        plt.plot(x_values[peaks], y_values[peaks], "x")
        plt.xlabel('Wavelength (um)')
        plt.ylabel('Absorption')
        plt.title('Visualise detected peaks')
        plt.show()

    # ~~~~~~~~ FITS GAUSSIAN OR LORENTZIAN MODELS TO THE DATA AND PLOTS THEM ~~~~~~~~
    # ~~~~~~~~ GAUSSIAN MODEL ~~~~~~~~
    if model == 'gaussian': 
        def gaussian(x, sigma, mu, amp):
            return amp*1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x - mu)**2/(2*sigma**2))     # equation for gaussian model

        peak = np.array(x_values[peaks]) #since peaks is an array of indexes, associating each index to it's wavelength value
        #widths = np.array(properties['widths']) #array of all the widths found by find_peaks doesn't appear to work (!!)
        widths = 0.00001 #using a hardcoded/random value for the width instead (problematic !!)
        amps = np.array(properties['peak_heights']) #array of all the heights found by find_peaks

        x = np.array(spectra_wav)
        y_g = np.zeros(x.shape) 

        for i in range(len(peaks)): #for each peak in the range of the length of the number of peaks... 
            g = gaussian(x, sigma=widths, mu=peak[i], amp=amps[i]) #...fit this gaussian
            y_g += g

        # fitting a gaussian changes the y value due to a normalisation process, thus to change it back to it's original value
        # a scale variable is created 
        scale = max(y_g)/max(properties['peak_heights'])
        amps_scaled = np.array(properties['peak_heights']) / scale #array of all the heights scaled down to the actual values

        y_gscaled = np.zeros(x.shape) 

        for i in range(len(peaks)): #for each peak in the range of the length of the number of peaks... 
            g = gaussian(x, sigma=widths, mu=peak[i], amp=amps_scaled[i]) #...fit this gaussian
            y_gscaled += g

        transmittance = 1 - y_gscaled    #this variable will allow a plot of the transmittance with the y-axis scaled down

        # PLOTTING 
        if xlimits is not None or ylimits is not None:
            # Plot with custom limits
            figure(figsize=(15, 4))
            plt.plot(x, transmittance)
            plt.xlim(xlimits)
            plt.ylim(ylimits)
            plt.xlabel('Wavelength (um)')
            plt.ylabel('Transmittance')
            plt.title('Fitting a Gaussian to the spectrum')
            plt.legend(['Gaussian fitted spectrum'])
            plt.show()
        else: 
            # Plot without custom limits
            figure(figsize=(15, 4))
            plt.plot(x, transmittance)
            plt.xlabel('Wavelength (um)')
            plt.ylabel('Transmittance')
            plt.title('Fitting a Gaussian to the spectrum')
            plt.legend(['Gaussian fitted spectrum'])
            plt.show()

    # ~~~~~~~~ LORENTZIAN MODEL ~~~~~~~~
    elif model == 'lorentzian':
        def lorentzian(x, sigma, mu, amp):
            return (amp/np.pi) * (sigma/((x-mu)**2 + sigma**2))     # equation for lorentzian model

        peak = np.array(x_values[peaks])  #since peaks is an array of indexes, associating each index to it's wavelength value
        #widths = np.array(properties['widths'])  #array of all the widths found by find_peaks doesn't appear to work (!!)
        widths = 0.00001 #using a hardcoded/random value for the width instead (problematic !!)
        amps = np.array(properties['peak_heights']) #array of all the heights found by find_peaks
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
        x = np.array(spectra_wav)
        y_l = np.zeros(x.shape) 

        for i in range(len(peaks)): #for each peak in the range of the length of the number of peaks...
            lor = lorentzian(x, sigma=widths, mu=peak[i], amp=amps[i]) #...fit this lorentzian
            y_l += lor

        # fitting a gaussian changes the y value due to a normalisation process, thus to change it back to it's original value
        # a scale variable is created 

        scale = max(y_l)/max(properties['peak_heights'])
        amps_scaled = np.array(properties['peak_heights']) / scale #array of all the heights

        y_lscaled = np.zeros(x.shape) 

        for i in range(len(peaks)): #for each peak in the range of the length of the number of peaks...
            lor = lorentzian(x, sigma=widths, mu=peak[i], amp=amps_scaled[i]) #...fit this lorentzian
            y_lscaled += lor

        transmittance = 1 - y_lscaled       #this variable will allow a plot of the transmittance with the y-axis scaled down
        
        # PLOTTING
        if xlimits is not None or ylimits is not None:  
            # Plot with custom limits
            figure(figsize=(15, 4))
            plt.plot(x, transmittance)
            plt.xlim(xlimits)
            plt.ylim(ylimits)
            plt.xlabel('Wavelength (um)')
            plt.ylabel('Transmittance')
            plt.title('Fitting a Lorentzian to the spectrum')
            plt.legend(['Lorentzian fitted spectrum'])
            plt.show()
        else: 
            # Plot without custom limits
            figure(figsize=(15, 4))
            plt.plot(x, transmittance)
            plt.xlabel('Wavelength (um)')
            plt.ylabel('Transmittance')
            plt.title('Fitting a Lorentzian to the spectrum')
            plt.legend(['Lorentzian fitted spectrum'])
            plt.show()

    return transmittance

################################################ Blackbody

def blackbodyabsorption(wavelength, spectra_data, temp): 
   """
   Function imprints a theoretical blackbody radiation curve at the chosen temperature with the absorption of the provided gas. 

   The parameters include:   
      - wavelength     : loads in the wavelengths
      - spectra_data   : loads in the spectral data
      - temp           : temperature of the blackbody in Kelvin
   """

   bb = BlackBody(temperature = temp*u.K)
   wavelength_units = (wavelength*10000)*(u.AA) # assuming BlackBody only uses wavelength units in Angstrom (!!) hence conversion
   flux = bb(wavelength_units) #consists of the calculated blackbody flux for the given wavelengths
   transmittance = spectra_data
   transmitted_intensity = transmittance*flux #consists of the absorption applied to a blackbody of given temperature 
      
   return flux, transmitted_intensity


################################################ Converting to photon counts

def convert_fluxunits_to_photoncounts(wavelength, flux, emission_coefficient, area, solid_angle, xlimits = None, ylimits = None):
    """
    Function converts the units from flux values erg /(cm2 Hz s sr) to photon counts (photons/sec) and plots a graph. 
    
    Parameters: 
        - wavelength            : loads in the wavelengths (um)
        - flux                  : input flux value to be converted into photon counts. Must be in erg /(cm2 Hz s sr)!
        - emisson_coefficient   : requires the emission coefficient of the light source filament at a particular temperature 
        - area                  : area of the fibre cross-section (cm^2)
        - solid_angle           : solid angle (steradians) that reaches the fibre core of a particular 
                                  diameter
        - xlimits               : user has a choice to input a range of x values (wavelength) i.e.,  [ , ] for the plot else default is complete range of values 
        - ylimits               : user has a choice to input a range of y values (Transmitted Intensity) i.e.,  [ , ] for the plot else default is complete range of values
    """
    tc = emission_coefficient
    A = area 
    sr = solid_angle 
    c = const.c.value
    h = const.h.value
    unitless_flux = flux 
    frequency = c/(wavelength*(10**(-6))) #calculates the frequency by first converting the wavelength to metres from um
    energy = h*c*frequency
    new_flux = unitless_flux*10**(-7) #such that erg /(cm2 Hz s sr) --> J /(cm2 Hz s sr)
    
    photoncount_persec = (tc*new_flux*A*sr*frequency)/energy # output units in sphotons per sec 

    if xlimits is not None or ylimits is not None: 
      # Plot with custom limits
        figure(figsize=(15, 4))
        plt.plot(wavelength, photoncount_persec)
        plt.title('Converting to photon counts (photons per sec)')
        plt.xlim(xlimits)
        plt.ylim(ylimits)
        plt.xlabel('Wavelength (um)')
        plt.ylabel('Photons / sec')
        plt.show()
    else:
        # Plot without custom limits
        figure(figsize=(15, 4))
        plt.plot(wavelength, photoncount_persec)
        plt.title('Converting to photon counts (photons per sec)')
        plt.xlabel('Wavelength (um)')
        plt.ylabel('Photons / sec')
        plt.show()

    return photoncount_persec

################################################
