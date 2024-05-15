# Funcs.py

# This python script contains the functions used to simulate the straining of a Fibre Bragg Grating. 
  
# This script goes with scratch_SimulatingCode_FBG.ipynb and newSimulatingCode_FBG.ipynb and 
# those files cannot be used without this script


# Written by: Samhita S Sodhi (15/05/24)
  

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

################################################
 
def input_spectra(filepath, wavelengthcol, datacol, separation, isheader = None, wavelengthunits = "um", dataunits = "fractional"):
    """
    This function loads in the datafile for the spectral information and assigns values to the wavelength and spectral data.
    
    Parameters include:
        - filepath                  : requires the path of the datafile contained in "", the file must contain at least one column 
                                      for wavelength given in nanometers (nm) or micrometres (um) and one column for spectral line 
                                      intensity in decibels or fractional values.  
        - wavelengthcol             : user inputs which column consists of the wavelength values.
        - datacol                   : user inputs which column consists of the spectra data values.
        - separation                : parameter takes in what the separation between the variables within the file
                                      e.g. "\t" means values separated by a tab.
        - isheader                  : parameter takes in whether there is a header in the file. Default is None - no header present.
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

def fit_curves(spectra_wav, spectra_data, detection_height, model = 'lorentzian', plot = 'N', xlimits = None, ylimits = None):
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
      - plot            : user has a choice to visualise the peak detection computation and fitted spectral data, default is No plot but input of 'Y' gives plots
      - xlimits         : user has a choice to input a range of x values (wavelength) i.e.,  [ , ] for the plot else default is complete range of values
      - ylimits         : user has a choice to input a range of y values (Transmitted Intensity) i.e.,  [ , ] for the plot else default is complete range of values
   """
    
    # ~~~~~~~~ DETECTING PEAKS ~~~~~~~~
    x_values = np.array(spectra_wav)
    y_values = np.array(1.0 - spectra_data) # flipping the graph so peak detection can occur

    peaks, properties = find_peaks(y_values, height= detection_height, width = 0) #properties include, peak heights, peak widths, etc. 

    if plot == 'Y':
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

        if plot == 'Y': 
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
        
        if plot == 'Y':
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


################################################ apply strain

def apply_strain(strain, wavelengths, transmittedvals, temp_change, thermal_exp_coeff, thermo_optic_coeff, strain_optic_coeff):
    """
    This function takes an input of a strain value and properties of the fibre bragg grating from which it 
    simulates the straining of the fibre bragg grating. 

    Parameters include:   
    - strain              : input a single strain value to stretch the fibre bragg grating given in micro-strain values i.e., XXe-6
    - wavelengths         : requires the input of the original gas spectra wavelength and fibre bragg grating wavelength as a list 
                            i.e., [gas_spectra_wavelength, fibre_wavelength]
    - transmittedvals     : requires the input of the transmitted_intensity calculated by the blackbodyabsorption function as a list
                            i.e., [transmitted_intensity, transmitted_intensityFBG]
    - temp_change         : change in temperature in degrees Celsius; 0 for constant temperature
    - thermal_exp_coeff   : thermal expansion coefficient for material of the fibre bragg grating per degree Celsius 
    - thermo_optic_coeff  : thermo-optic coefficient for material of the fibre bragg grating per degree Celsius
    - strain_optic_coeff  : effective strain-optic coefficient of the fibre 

    Equation used to simulate wavelength change: wav_shift = (1 - pe)*strain + (alpha + nu)*t_shift 
      (assumes stretching of the grating due to strain and temperature).
    """   

    t_shift = temp_change 
    alpha = thermal_exp_coeff
    nu = thermo_optic_coeff 
    pe = strain_optic_coeff 
    fillvalue = max(transmittedvals[0]).value #chooses the max value rather than the boundary values because max_values are typically the boundary values

    wav_shift = (1 - pe)*strain + (alpha + nu)*t_shift
    fibre_wavelength_new = wavelengths[0] + wav_shift*wavelengths[0] #creates new wavelengths for which there is no existing data 
    f1 = interp1d(fibre_wavelength_new, transmittedvals[1], bounds_error = False, fill_value = fillvalue) #bounds_error = False -> interp1d sets the out-of-range values with the fill_value, which is nan by default / here fitted to the max value
    interpolatedvals = f1(wavelengths[1])

    return fibre_wavelength_new, interpolatedvals


################################################  plotting strain

def plot_strainspectra(wavelengths, transmittedvals, xlimits = None, ylimits = None):
    """
    This function plots the spectra once the strain has been applied to the fibre bragg grating. 
    
    Parameters include: 
      - wavelengths     : requires the input of the original gas spectra wavelength, fibre bragg grating wavelength and the new strained wavelengths 
                          calculated from the apply_strain function as a list i.e., [spectra_wav, fibre_wavelength, fibre_wavelength_new]
      - transmittedvals : requires the input of the transmitted_intensity calculated by the blackbodyabsorption function and new data values
                          calculated by apply_strain as a list i.e., [transmitted_intensity, transmitted_intensityFBG, interpolatedvals]
      - xlimits         : user has a choice to input a range of x values (wavelength) i.e.,  [ , ] for the plot else default is complete range of values 
      - ylimits         : user has a choice to input a range of y values (Transmitted Intensity) i.e.,  [ , ] for the plot else default is complete range of values
    """

    # Plotting different curves 
    if xlimits is not None or ylimits is not None: 
      # Plot with custom limits
      figure(figsize=(15, 4))
      plt.plot(wavelengths[0], transmittedvals[0], c = 'navy', label = 'Original gas spectra')
      plt.plot(wavelengths[1], transmittedvals[1], c = 'powderblue', label = 'Original FBG spectra')
      plt.plot(wavelengths[2], transmittedvals[2], c = 'red', label = 'Strained FBG spectra')
      plt.xlim(xlimits)
      plt.ylim(ylimits)
      plt.legend(loc = 'lower right')
      plt.title('Straining the Fiber Bragg Grating')
      plt.xlabel('Wavelength (um)')
      plt.ylabel('Transmitted Intensity (erg / cm2 Hz s sr)')
      plt.show()
      
    else: 
      # Plot without custom limits
      figure(figsize=(15, 4))
      plt.plot(wavelengths[0], transmittedvals[0], c = 'navy', label = 'Original gas spectra')
      plt.plot(wavelengths[1], transmittedvals[1], c = 'powderblue', label = 'Original FBG spectra')
      plt.plot(wavelengths[2], transmittedvals[2], c = 'red', label = 'Strained FBG spectra')
      plt.legend(loc = 'lower right')
      plt.title('Straining the Fiber Bragg Grating')
      plt.xlabel('Wavelength (um)')
      plt.ylabel('Transmitted Intensity (erg / cm2 Hz s sr)')
      plt.show()
    
    return


################################################ amount of transmitted light per strain value

def correlation(strainvalues, wavelengths, spectra_data, initial_transmittedvals, normalisation = 'True'):
    """
    For different input values of strain this function plots the amount of total, transmitted and reflected light once incident light
    passes through the fibre bragg grating.

    Parameters include: 
    - strainvalues            : user inputs values of strain i.e., strain = np.linspace(0, 0.0014, 500) for which the stretching of the fibre bragg 
                                grating is simulated
    - wavelengths             : requires the input of the original gas spectra wavelength and fibre bragg grating wavelength as a list 
                                i.e., [gas_spectra_wavelength, fibre_wavelength]
    - spectra_data            : requires the input of the transmitted_intensity calculated by the blackbodyabsorption function as a list
                                i.e., [transmitted_intensity, transmitted_intensityFBG]
    - initial_transmittedvals : requires input of the original gas spectra data
    - normalisation           : user can choose whether to normalise the calculations to see the outputs as a fraction rather than true values else 
                                default is 'True'. 
    """

    reflectedvals = []
    transmittedvals = []
    totallightvals = []
    strain = strainvalues
    fillvalue = max(initial_transmittedvals[0]).value  

    for i in strain:
        fibre_wavelength_new, interpolatedvals = apply_strain(i, wavelengths, initial_transmittedvals, 0, 0.55e-6, 8.6e-6, 0.22) # values for silica, no temperature change
        totallight = fillvalue*np.ones(len(interpolatedvals*spectra_data)) # calculates the total amount of light going through the system
        reflected = (totallight - interpolatedvals*spectra_data).sum() # reflected light = total - transmitted
        transmitted = np.array(interpolatedvals*spectra_data).sum() # transmitted light
        
        sumtotal_light = totallight.sum()

        if normalisation == 'True':
            normalise_transmitted = transmitted / sumtotal_light #normalises transmitted light
            normalise_reflected = reflected / sumtotal_light #normalises reflected light
            
            transmittedvals.append(normalise_transmitted)
            reflectedvals.append(normalise_reflected)
            totallightvals.append(normalise_transmitted + normalise_reflected)
            
        elif normalisation == 'False':
            transmittedvals.append(transmitted)
            reflectedvals.append(reflected)
            totallightvals.append(transmitted+reflected) 
            
        else: 
            print("Please type either 'True' or 'False'")

    if normalisation == 'True':
        figure(figsize=(15, 4))
        plt.plot(strain, transmittedvals)
        plt.title('Transmitted light')
        plt.xlabel('Strain')
        plt.ylabel('Light Transmitted as a fraction of the total light')
        plt.show()

        figure(figsize=(15, 4))
        plt.plot(strain, reflectedvals)
        plt.title('Reflected light')
        plt.xlabel('Strain')
        plt.ylabel('Light Reflected as a fraction of the total light')
        plt.show()  
        
        totallight = np.mean(np.array(transmittedvals) + np.array(reflectedvals))
        print('Total light (erg / cm2 Hz s sr):', str(np.round(totallight, 1)))

    elif normalisation == 'False':
        figure(figsize=(15, 4))
        plt.plot(strain, transmittedvals)
        plt.title('Transmitted light')
        plt.xlabel('Strain')
        plt.ylabel('Light Transmitted (erg / cm2 Hz s sr)')
        plt.show()

        figure(figsize=(15, 4))
        plt.plot(strain, reflectedvals)
        plt.title('Reflected light')
        plt.xlabel('Strain')
        plt.ylabel('Light Reflected (erg / cm2 Hz s sr)')
        plt.show()  
        
        totallight = np.mean(np.array(transmittedvals) + np.array(reflectedvals))
        print('Total light (erg / cm2 Hz s sr):', str(totallight))

    return transmittedvals, reflectedvals


################################################ Converting to photon counts

def convert_fluxunits_to_photoncounts(wavelength, flux, emission_coefficient, area, solid_angle, xlimits = None, ylimits = None):
    """
    Function converts the units from flux values erg /(cm2 Hz s sr) to photon counts (photons/sec) and plots a graph. 
    Note to user: Use only if flux values are not normalised! 

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
    unitless_flux = np.array(flux)
    frequency = c/(wavelength*(10**(-6))) #calculates the frequency by first converting the wavelength to metres from um
    energy = h*c*frequency
    new_flux = unitless_flux*(10**(-7)) #such that erg /(cm2 Hz s sr) --> J /(cm2 Hz s sr)
    
    photoncount_persec = (tc*new_flux*A*sr*frequency)/energy # output units in photons per sec 

    if xlimits is not None or ylimits is not None: 
      # Plot with custom limits
        figure(figsize=(15, 4))
        plt.plot(wavelength, photoncount_persec)
        plt.title('Light in photon counts (photons per sec)')
        plt.xlim(xlimits)
        plt.ylim(ylimits)
        plt.xlabel('Wavelength (um)')
        plt.ylabel('Photons / sec')
        plt.show()
    else:
        # Plot without custom limits
        figure(figsize=(15, 4))
        plt.plot(wavelength, photoncount_persec)
        plt.title('Light in photon counts (photons per sec)')
        plt.xlabel('Wavelength (um)')
        plt.ylabel('Photons / sec')
        plt.show()

    return photoncount_persec

################################################ 