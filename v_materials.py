#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The `v_materials` module contains tools to work with materials and sources.

It holds tools to work with...
    
    - Meep's medium classes and functions.
    - Complex dielectric constant epsilon and complex refractive index N.
    
Some of its most useful tools are...    

import_module : function
    Returns Medium instance from string with specified length scale
epsilon_function : function
    Generates an interpolation function for epsilon from experimental data.

It's widely based on Meep Materials Library.

@author: vall
"""

from socket import gethostname
if "Nano" in gethostname():
    syshome = "/home/nanofisica/Documents/Vale/ThesisPython"
elif "vall" in gethostname():
    syshome = "/home/vall/Documents/Thesis/ThesisPython"
else:
    raise ValueError("Your PC must be registered at the top of this code")

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import interp1d
try:
    import meep as mp
except:
    print("Meep functions not available. Don't worry! You can still use everything else")
import v_utilities as vu

#%% MEEP MEDIUM IMPORT

def import_medium(name, from_um_factor=1, paper="R"):
    
    """Returns Medium instance from string with specified length scale
    
    It's widely based on Meep Materials Library, merely adapted to change 
    according to the length unit scale used.
    
    Parameters
    ----------
    name: str
        Name of the desired material.
    from_um_factor=1 : int, float, optional
        Factor to transform from SI μm to the chosen length unit. For example, 
        to work with 100 nm as 1 Meep unit, from_um_factor=.1 must be specified.
    paper="R": str
        Name of desired source for experimental input data of medium.
    
    Returns
    -------
    medium : mp.Medium
        The mp.Medium instance of the desired material.
    
    Raises
    ------
    "No media called ... is available" : SyntaxError
        If no material is found whose name matches the string given.
    """
    
    if "Ag" not in name and "Au" not in name:
        raise SyntaxError("No media called {} is available".format(name))
        return
           
    # Default unit length is 1 um
    eV_from_um_factor = from_um_factor/1.23984193 # Conversion factor: eV to 1/um [=1/hc]
    # from_um_factor used to be um_scale (but that's a shady name)
    
    ############ GOLD #################################################
    
    if name=="Au" and paper=="R":
        
    #------------------------------------------------------------------
    # Elemental metals from A.D. Rakic et al., Applied Optics, Vol. 37, No. 22, pp. 5271-83, 1998
    # Wavelength range: 0.2 - 12.4 um
    # Gold (Au)

        metal_range = mp.FreqRange(min=from_um_factor*1e3/6199.2, 
                                   max=from_um_factor*1e3/247.97)
        
        Au_plasma_frq = 9.03*eV_from_um_factor
        Au_f0 = 0.760
        Au_frq0 = 1e-10
        Au_gam0 = 0.053*eV_from_um_factor
        Au_sig0 = Au_f0*Au_plasma_frq**2/Au_frq0**2
        Au_f1 = 0.024
        Au_frq1 = 0.415*eV_from_um_factor      # 2.988 um
        Au_gam1 = 0.241*eV_from_um_factor
        Au_sig1 = Au_f1*Au_plasma_frq**2/Au_frq1**2
        Au_f2 = 0.010
        Au_frq2 = 0.830*eV_from_um_factor      # 1.494 um
        Au_gam2 = 0.345*eV_from_um_factor
        Au_sig2 = Au_f2*Au_plasma_frq**2/Au_frq2**2
        Au_f3 = 0.071
        Au_frq3 = 2.969*eV_from_um_factor      # 0.418 um
        Au_gam3 = 0.870*eV_from_um_factor
        Au_sig3 = Au_f3*Au_plasma_frq**2/Au_frq3**2
        Au_f4 = 0.601
        Au_frq4 = 4.304*eV_from_um_factor      # 0.288 um
        Au_gam4 = 2.494*eV_from_um_factor
        Au_sig4 = Au_f4*Au_plasma_frq**2/Au_frq4**2
        Au_f5 = 4.384
        Au_frq5 = 13.32*eV_from_um_factor      # 0.093 um
        Au_gam5 = 2.214*eV_from_um_factor
        Au_sig5 = Au_f5*Au_plasma_frq**2/Au_frq5**2
        
        Au_susc = [mp.DrudeSusceptibility(frequency=Au_frq0, gamma=Au_gam0, sigma=Au_sig0),
                   mp.LorentzianSusceptibility(frequency=Au_frq1, gamma=Au_gam1, sigma=Au_sig1),
                   mp.LorentzianSusceptibility(frequency=Au_frq2, gamma=Au_gam2, sigma=Au_sig2),
                   mp.LorentzianSusceptibility(frequency=Au_frq3, gamma=Au_gam3, sigma=Au_sig3),
                   mp.LorentzianSusceptibility(frequency=Au_frq4, gamma=Au_gam4, sigma=Au_sig4),
                   mp.LorentzianSusceptibility(frequency=Au_frq5, gamma=Au_gam5, sigma=Au_sig5)]
        
        Au = mp.Medium(epsilon=1.0, E_susceptibilities=Au_susc, 
                       valid_freq_range=metal_range)
        Au.from_um_factor = from_um_factor
        
        return Au

    elif name=="Au" and paper=="JC":
        
    #------------------------------------------------------------------
    # Metals from D. Barchiesi and T. Grosges, J. Nanophotonics, Vol. 8, 08996, 2015
    # Wavelength range: 0.4 - 0.8 um
    # Gold (Au)
    # Fit to P.B. Johnson and R.W. Christy, Physical Review B, Vol. 6, pp. 4370-9, 1972
        
        metal_visible_range = mp.FreqRange(min=from_um_factor*1e3/800, 
                                           max=from_um_factor*1e3/400)
    
        Au_JC_visible_frq0 = 1*from_um_factor/0.139779231751333
        Au_JC_visible_gam0 = 1*from_um_factor/26.1269913352870
        Au_JC_visible_sig0 = 1
        
        Au_JC_visible_frq1 = 1*from_um_factor/0.404064525036786
        Au_JC_visible_gam1 = 1*from_um_factor/1.12834046202759
        Au_JC_visible_sig1 = 2.07118534879440
        
        Au_JC_visible_susc = [mp.DrudeSusceptibility(frequency=Au_JC_visible_frq0, gamma=Au_JC_visible_gam0, sigma=Au_JC_visible_sig0),
                              mp.LorentzianSusceptibility(frequency=Au_JC_visible_frq1, gamma=Au_JC_visible_gam1, sigma=Au_JC_visible_sig1)]
        
        Au_JC_visible = mp.Medium(epsilon=6.1599, 
                                  E_susceptibilities=Au_JC_visible_susc, 
                                  valid_freq_range=metal_visible_range)
        Au_JC_visible.from_um_factor = from_um_factor
        
        return Au_JC_visible

    elif name=="Au" and paper=="P":
        
    #------------------------------------------------------------------
    # Metals from D. Barchiesi and T. Grosges, J. Nanophotonics, Vol. 8, 08996, 2015
    # Wavelength range: 0.4 - 0.8 um
    # Gold (Au)
    # Fit to E.D. Palik, Handbook of Optical Constants, Academic Press, 1985 

        metal_visible_range = mp.FreqRange(min=from_um_factor*1e3/800, 
                                           max=from_um_factor*1e3/400)
        
        Au_visible_frq0 = 1*from_um_factor/0.0473629248511456
        Au_visible_gam0 = 1*from_um_factor/0.255476199605166
        Au_visible_sig0 = 1
        
        Au_visible_frq1 = 1*from_um_factor/0.800619321082804
        Au_visible_gam1 = 1*from_um_factor/0.381870287531951
        Au_visible_sig1 = -169.060953137985
        
        Au_visible_susc = [mp.DrudeSusceptibility(frequency=Au_visible_frq0, gamma=Au_visible_gam0, sigma=Au_visible_sig0),
                           mp.LorentzianSusceptibility(frequency=Au_visible_frq1, gamma=Au_visible_gam1, sigma=Au_visible_sig1)]
        
        Au_visible = mp.Medium(epsilon=0.6888, E_susceptibilities=Au_visible_susc, 
                               valid_freq_range=metal_visible_range)
        Au_visible.from_um_factor = from_um_factor
        
        return Au_visible
    
    elif name=="Au":
        raise ValueError("No source found for Au with that name")

    ############ SILVER ###############################################

    if name=="Ag" and paper=="R":
        
    #------------------------------------------------------------------
    # Elemental metals from A.D. Rakic et al., Applied Optics, Vol. 37, No. 22, pp. 5271-83, 1998
    # Wavelength range: 0.2 - 12.4 um
    # Silver (Ag)
        
        metal_range = mp.FreqRange(min=from_um_factor*1e3/12398, 
                                   max=from_um_factor*1e3/247.97)      
        
        Ag_plasma_frq = 9.01*eV_from_um_factor
        Ag_f0 = 0.845
        Ag_frq0 = 1e-10
        Ag_gam0 = 0.048*eV_from_um_factor
        Ag_sig0 = Ag_f0*Ag_plasma_frq**2/Ag_frq0**2
        Ag_f1 = 0.065
        Ag_frq1 = 0.816*eV_from_um_factor      # 1.519 um
        Ag_gam1 = 3.886*eV_from_um_factor
        Ag_sig1 = Ag_f1*Ag_plasma_frq**2/Ag_frq1**2
        Ag_f2 = 0.124
        Ag_frq2 = 4.481*eV_from_um_factor      # 0.273 um
        Ag_gam2 = 0.452*eV_from_um_factor
        Ag_sig2 = Ag_f2*Ag_plasma_frq**2/Ag_frq2**2
        Ag_f3 = 0.011
        Ag_frq3 = 8.185*eV_from_um_factor      # 0.152 um
        Ag_gam3 = 0.065*eV_from_um_factor
        Ag_sig3 = Ag_f3*Ag_plasma_frq**2/Ag_frq3**2
        Ag_f4 = 0.840
        Ag_frq4 = 9.083*eV_from_um_factor      # 0.137 um
        Ag_gam4 = 0.916*eV_from_um_factor
        Ag_sig4 = Ag_f4*Ag_plasma_frq**2/Ag_frq4**2
        Ag_f5 = 5.646
        Ag_frq5 = 20.29*eV_from_um_factor      # 0.061 um
        Ag_gam5 = 2.419*eV_from_um_factor
        Ag_sig5 = Ag_f5*Ag_plasma_frq**2/Ag_frq5**2
        
        Ag_susc = [mp.DrudeSusceptibility(frequency=Ag_frq0, gamma=Ag_gam0, sigma=Ag_sig0),
                   mp.LorentzianSusceptibility(frequency=Ag_frq1, gamma=Ag_gam1, sigma=Ag_sig1),
                   mp.LorentzianSusceptibility(frequency=Ag_frq2, gamma=Ag_gam2, sigma=Ag_sig2),
                   mp.LorentzianSusceptibility(frequency=Ag_frq3, gamma=Ag_gam3, sigma=Ag_sig3),
                   mp.LorentzianSusceptibility(frequency=Ag_frq4, gamma=Ag_gam4, sigma=Ag_sig4),
                   mp.LorentzianSusceptibility(frequency=Ag_frq5, gamma=Ag_gam5, sigma=Ag_sig5)]
        
        Ag = mp.Medium(epsilon=1.0, E_susceptibilities=Ag_susc, 
                       valid_freq_range=metal_range)
        Ag.from_um_factor = from_um_factor
        
        return Ag
    
    elif name=="Ag" and paper=="P":

    #------------------------------------------------------------------
    # Metals from D. Barchiesi and T. Grosges, J. Nanophotonics, Vol. 8, 08996, 2015
    # Wavelength range: 0.4 - 0.8 um
    ## WARNING: unstable; field divergence may occur
    # Silver (Au)
    # Fit to E.D. Palik, Handbook of Optical Constants, Academic Press, 1985 

        metal_visible_range = mp.FreqRange(min=from_um_factor*1e3/800, 
                                           max=from_um_factor*1e3/400)
        
        Ag_visible_frq0 = 1*from_um_factor/0.142050162130618
        Ag_visible_gam0 = 1*from_um_factor/18.0357292925015
        Ag_visible_sig0 = 1
        
        Ag_visible_frq1 = 1*from_um_factor/0.115692151792108
        Ag_visible_gam1 = 1*from_um_factor/0.257794324096575
        Ag_visible_sig1 = 3.74465275944019
        
        Ag_visible_susc = [mp.DrudeSusceptibility(frequency=Ag_visible_frq0, gamma=Ag_visible_gam0, sigma=Ag_visible_sig0),
                           mp.LorentzianSusceptibility(frequency=Ag_visible_frq1, gamma=Ag_visible_gam1, sigma=Ag_visible_sig1)]
        
        Ag_visible = mp.Medium(epsilon=0.0067526, 
                               E_susceptibilities=Ag_visible_susc, 
                               valid_freq_range=metal_visible_range)
        Ag_visible.from_um_factor = from_um_factor
        
        return Ag_visible

    elif name=="Ag":    
        raise ValueError("No source found for Ag with that name")
    
    else:
        raise ValueError("No source found for that material")

#%% EPSILON INTERPOLATION

def epsilon_interpoler_from_n(wlen, complex_n):
    """
    Generates an interpolation function for epsilon from experimental N data.

    Parameters
    ----------
    wlen : np.array, list
        Wavelength in nm.
    complex_n : np.array, list
        Complex refractive index N = n + ik, dimensionless.

    Returns
    -------
    epsilon_function : function
        Epsilon interpoler that takes wavelength in nm as argument and returns 
        complex dielectric constant or relative permitivitty 
        epsilon = epsilon' + i epsilon'', dimensionless.
    """

    n_function = interp1d(wlen, np.real(complex_n), kind="cubic")
    k_function = interp1d(wlen, np.imag(complex_n), kind="cubic")
    N_function = lambda wl : n_function(wl) + 1j * k_function(wl)
    epsilon_function = lambda wl : np.power(N_function(wl), 2)
    
    return epsilon_function

def epsilon_interpoler_from_epsilon(wlen, complex_epsilon):
    """
    Generates an interpolation function for epsilon from experimental epsilon data.

    Parameters
    ----------
    wlen : np.array, list
        Wavelength in nm.
    complex_epsilon : np.array, list
        Complex dielectric constant or relative permitivitty 
        epsilon = epsilon' + i epsilon'', dimensionless.

    Returns
    -------
    epsilon_function : function
        Epsilon interpoler that takes wavelength in nm as argument and returns 
        complex dielectric constant or relative permitivitty 
        epsilon = epsilon' + i epsilon'', dimensionless.
    """

    real_function = interp1d(wlen, np.real(complex_epsilon), kind="cubic")
    imag_function = interp1d(wlen, np.imag(complex_epsilon), kind="cubic")
    epsilon_function = lambda wl : real_function(wl) + 1j * imag_function(wl)
    
    return epsilon_function

def epsilon_function_from_meep(material="Au", paper="JC", from_um_factor=1e-3):
    
    """
    Generates a function for isotropic epsilon from Meep Drude-Lorentz fit data.
    
    Parameters
    ----------
    material="Au" : str
        Material's chemical symbol. Available: 'Au' for gold.
    paper="JC" : str
        Paper source of experimental data. Available: 'JC' for Johnson 
        and Christy, 'R' for Rakic, 'P' for Palik.
    from_um_factor=1e-3 : float, optional
        Meep factor of length scale implying 1 Meep length unit is 
        from_um_factor length units in μm. If provided, the function takes 
        wavelength in Meep units instead of nm.
        
    Returns
    -------
    epsilon_function : function
        Epsilon function that takes wavelength in nm or Meep units as argument 
        and returns complex dielectric constant or relative permitivitty 
        epsilon = epsilon' + i epsilon'', dimensionless.
        
    Raises
    ------
    ValueError : "Material should either be..."
        When the desired material isn't available.
    ValueError : "Reference paper should either be..."
        When the desired paper reference of experimental data isn't available.
    ValueError : "Data should either be from..."
        When the desired data source isn't available.
    ValueError : "Experimental data couldn't be found. Sorry!"
        When the combination of parameters causes the data not to be found.
    """

    
    available_materials = {"Au": "gold", "Ag": "silver"}
    available_papers = {"JC": "Johnson & Christy", "R": "Rakic", "P": "Palik"}
    
    if material not in available_materials.keys():
        error = "Data should either be from "
        error += vu.enumerate_string(vu.join_strings_dict(available_materials), 
                                     "or", True)
        raise ValueError(error)
    if paper not in available_papers.keys():
        error = "Reference paper for experimental data should either be "
        error += vu.enumerate_string(vu.join_strings_dict(available_papers), 
                                     "or", True)
        raise ValueError(error)
        
    medium = import_medium(material, 
                           paper=paper) # This one has from_um_factor=1
    
    print(f"Data loaded using Meep and '{paper}'")
    epsilon_function = lambda wlen : medium.epsilon(1/(wlen*from_um_factor))[0,0]
    # To pass it to the medium, I transform wavelength from nm (or Meep units) to um
    
    return epsilon_function

def epsilon_function_from_file(material="Au", paper="JC", reference="RIinfo", plot=False):
    """
    Generates an interpolation function for epsilon from experimental data.
    
    Parameters
    ----------
    material="Au" : str
        Material's chemical symbol. Available: 'Au' for gold.
    paper="JC" : str
        Paper source of experimental data. Available: 'JC' for Johnson 
        and Christy, 'R' for Rakic.
    reference="RIinfo" : str
        Reference from which the data was extracted, for example a web page. 
        Available: 'RIinfo' for 'www.refractiveindex.info'
    plot=False : bool
        Parameter that enables a plot of the interpolation and the data used.
        
    Returns
    -------
    epsilon_function : function
        Epsilon interpoler that takes wavelength in nm as argument and returns 
        complex dielectric constant or relative permitivitty 
        epsilon = epsilon' + i epsilon'', dimensionless.
    
    Raises
    ------
    ValueError : "Material should either be..."
        When the desired material isn't available.
    ValueError : "Reference paper should either be..."
        When the desired paper reference of experimental data isn't available.
    ValueError : "Data should either be from..."
        When the desired data source isn't available.
    ValueError : "Experimental data couldn't be found. Sorry!"
        When the combination of parameters causes the data not to be found.
    """
    
    available_materials = {"Au": "gold", "Ag": "silver"}
    available_papers = {"JC": "Johnson & Christy",
                        "R": "Rakic"}
    available_references = {"RIinfo": "www.refractiveindex.info"}
    
    if material not in available_materials.keys():
        error = "Data should either be from "
        error += vu.enumerate_string(vu.join_strings_dict(available_materials), 
                                     "or", True)
        raise ValueError(error)
    if paper not in available_papers.keys():
        error = "Reference paper for experimental data should either be "
        error += vu.enumerate_string(vu.join_strings_dict(available_papers), 
                                     "or", True)
        raise ValueError(error)
    if reference not in available_references.keys():
        error = "Experimental data should either be extracted from "
        error += vu.enumerate_string(vu.join_strings_dict(available_references), 
                                     "or", True)
        raise ValueError(error)
    
    data_series = os.listdir(os.path.join(syshome, 'MaterialsData'))
    
    try:
        data_files = []
        for df in data_series:
            if (f"_{paper}_") in df and material in df and reference in df:
                data_files.append( os.path.join(syshome, 'MaterialsData', df) )
    except:
        raise ValueError("Experimental data couldn't be found. Sorry!")
    
    file = data_files[0]
    print(f"Data loaded from '{file}'")
    data = np.loadtxt(file)
    
    if 'N' in file:
        epsilon_function = epsilon_interpoler_from_n(
            data[:,0], data[:,1] + 1j*data[:,2])
    elif "eps" in file.lower():
        epsilon_function = epsilon_interpoler_from_epsilon(
            data[:,0], data[:,1] + 1j*data[:,2])
    else:
        raise ValueError("Experimental data couldn't be recognized. Sorry!")
            
    if plot:
        wlen = data[:,0]
        wlen_range = [min(data[:,0]), max(data[:,0])]
        wlen_long = np.linspace(*wlen_range, 500)
        
        if "eps" in file.lower():
            epsilon = data[:,1] + 1j*data[:,2]
        else:
            epsilon = np.power(data[:,1] + 1j*data[:,2], 2)
        epsilon_interpolated = epsilon_function(wlen_long)
                
        functions = [np.abs, np.real, np.imag]
        titles = ["Absolute value", "Real part", "Imaginary part"]
        ylabels = [r"|$\epsilon$| [nm$^3$]", r"Re($\epsilon$) [nm$^3$]", 
                   r"Im($\epsilon$) [nm$^3$]"]
        
        nplots = len(functions)
        fig = plt.figure(figsize=(nplots*6.4, 6.4))
        axes = fig.subplots(ncols=nplots)
        
        max_value = []
        min_value = []
        for ax, f, t, y in zip(axes, functions, titles, ylabels):
            ax.set_title(t)
            ax.plot(wlen, f(epsilon), "ob", label="Data")
            ax.plot(wlen_long, f(epsilon_interpolated), "-r", label="Interpolation")
            ax.xaxis.set_label_text("Wavelength [nm]")
            ax.yaxis.set_label_text(y)
            ax.legend()
            ax.set_xlim(*wlen_range)
            max_value.append(max([max(f(epsilon)), max(f(epsilon_interpolated))]))
            min_value.append(min([min(f(epsilon)), min(f(epsilon_interpolated))]))
                
        for ax in axes: ax.set_ylim([min(min_value)-.1*(max(max_value)-min(min_value)), 
                                     max(max_value)+.1*(max(max_value)-min(min_value))])
        axes[0].text(-.1, -.13, file, transform=axes[0].transAxes)
    
    return epsilon_function

def epsilon_function(material="Au", paper="JC", reference="RIinfo",
                     from_um_factor=1e-3):
    """
    Generates an interpolation function for epsilon from experimental data.
    
    Parameters
    ----------
    material="Au" : str
        Material's chemical symbol. Available: 'Au' for gold.
    paper="JC" : str
        Reference paper of experimental data. Available: 'JC' for Johnson and 
        Christy, 'R' for Rakic, 'P' for Palik.
    reference="RIinfo" : str
        Reference from which the data was extracted, for example a web page. 
        Available: 'RIinfo' for 'www.refractiveindex.info' and 'Meep' for Meep 
        materials library that uses a Drude-Lorentz model to fit data.
    from_um_factor=1e-3 : float, optional
        Meep factor of length scale implying 1 Meep length unit is 
        from_um_factor length units in μm. If provided, the function takes 
        wavelength in Meep units instead of nm.
        
    Returns
    -------
    epsilon_function : function
        Epsilon function that takes wavelength in nm or Meep units as argument 
        and returns complex dielectric constant or relative permitivitty 
        epsilon = epsilon' + i epsilon'', dimensionless.
        
    Raises
    ------
    ValueError : "Material should either be..."
        When the desired material isn't available.
    ValueError : "Reference paper should either be..."
        When the desired paper reference of experimental data isn't available.
    ValueError : "Data should either be from..."
        When the desired data source isn't available.
    ValueError : "Experimental data couldn't be found. Sorry!"
        When the combination of parameters causes the data not to be found.
    """
    
    available_references = {"RIinfo": "www.refractiveindex.info",
                           "Meep": "Meep materials library & Drude-Lorentz fit"}
    
    if reference not in available_references.keys():
        error = "Data should either be from"
        error += vu.enumerate_string(vu.join_strings_dict(available_references), 
                                     "or", True)
        raise ValueError(error)
    
    if reference=="Meep":
        epsilon_function = epsilon_function_from_meep(material, paper, 
                                                      from_um_factor)
    elif reference=="RIinfo":
        epsilon_function = epsilon_function_from_file(material, paper, 
                                                      reference)
    
    return epsilon_function


#%% MEEP MEDIUM WITH INTERPOLATION

class MediumInterpoler(mp.Medium):
    
    """A Meep Medium subclass that holds a line instead of a whole volume"""
    
    def __init__(self):
        
        super().__init__()
        
    def epsilon(self,freq):
        """
        Returns the medium's permittivity tensor as a 3x3 Numpy array at the specified
        frequency `freq` which can be either a scalar, list, or Numpy array. In the case
        of a list/array of N frequency points, a Numpy array of size Nx3x3 is returned.
        """
        return self._get_eps(self.epsilon_diag, self.epsilon_offdiag, self.E_susceptibilities, self.D_conductivity_diag, self.D_conductivity_offdiag, freq)

    def mu(self,freq):
        """
        Returns the medium's permeability tensor as a 3x3 Numpy array at the specified
        frequency `freq` which can be either a scalar, list, or Numpy array. In the case
        of a list/array of N frequency points, a Numpy array of size Nx3x3 is returned.
        """
        return self._get_mu(self.mu_diag, self.mu_offdiag, self.H_susceptibilities, self.B_conductivity_diag, self.B_conductivity_offdiag, freq)

 
    def _get_eps(self, diag, offdiag, susceptibilities, conductivity_diag, conductivity_offdiag, freq):
        # Clean the input
        if np.isscalar(freq):
            freqs = np.array(freq)[np.newaxis, np.newaxis, np.newaxis]
        else:
            freqs = np.squeeze(freq)
            freqs = freqs[:, np.newaxis, np.newaxis]

        # Check for values outside of allowed ranges
        if np.min(np.squeeze(freqs)) < self.valid_freq_range.min:
            raise ValueError('User specified frequency {} is below the Medium\'s limit, {}.'.format(np.min(np.squeeze(freqs)),self.valid_freq_range.min))
        if np.max(np.squeeze(freqs)) > self.valid_freq_range.max:
            raise ValueError('User specified frequency {} is above the Medium\'s limit, {}.'.format(np.max(np.squeeze(freqs)),self.valid_freq_range.max))

        # Initialize with instantaneous dielectric tensor
        epsmu = np.expand_dims(mp.Matrix(diag=diag,offdiag=offdiag),axis=0)

        # Iterate through susceptibilities
        for i_sus in range(len(susceptibilities)):
            epsmu = epsmu + susceptibilities[i_sus].eval_susceptibility(freqs)

        # Account for conductivity term (only multiply if nonzero to avoid unnecessary complex numbers)
        conductivity = np.expand_dims(mp.Matrix(diag=conductivity_diag,offdiag=conductivity_offdiag),axis=0)
        if np.count_nonzero(conductivity) > 0:
            epsmu = (1 + 1j/freqs * conductivity) * epsmu

        # Convert list matrix to 3D numpy array size [freqs,3,3]
        return np.squeeze(epsmu)
    
    def _get_mu(self, diag, offdiag, susceptibilities, conductivity_diag, conductivity_offdiag, freq):
        # Clean the input
        if np.isscalar(freq):
            freqs = np.array(freq)[np.newaxis, np.newaxis, np.newaxis]
        else:
            freqs = np.squeeze(freq)
            freqs = freqs[:, np.newaxis, np.newaxis]

        # Check for values outside of allowed ranges
        if np.min(np.squeeze(freqs)) < self.valid_freq_range.min:
            raise ValueError('User specified frequency {} is below the Medium\'s limit, {}.'.format(np.min(np.squeeze(freqs)),self.valid_freq_range.min))
        if np.max(np.squeeze(freqs)) > self.valid_freq_range.max:
            raise ValueError('User specified frequency {} is above the Medium\'s limit, {}.'.format(np.max(np.squeeze(freqs)),self.valid_freq_range.max))

        # Initialize with instantaneous dielectric tensor
        epsmu = np.expand_dims(mp.Matrix(diag=diag,offdiag=offdiag),axis=0)

        # Iterate through susceptibilities
        for i_sus in range(len(susceptibilities)):
            epsmu = epsmu + susceptibilities[i_sus].eval_susceptibility(freqs)

        # Account for conductivity term (only multiply if nonzero to avoid unnecessary complex numbers)
        conductivity = np.expand_dims(mp.Matrix(diag=conductivity_diag,offdiag=conductivity_offdiag),axis=0)
        if np.count_nonzero(conductivity) > 0:
            epsmu = (1 + 1j/freqs * conductivity) * epsmu

        # Convert list matrix to 3D numpy array size [freqs,3,3]
        return np.squeeze(epsmu)
