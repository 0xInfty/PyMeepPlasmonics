# -*- coding: utf-8 -*-
"""
This module contains electromagnetic theory and basic data for materials.

It could be divided into two sections:
    
    1. Complex dielectric constant epsilon and complex refractive index N tools.
        These functions create interpolers from experimental data that can 
        either be retrieved from a file downloaded from any web site or from 
        Meep's materials library that applies a Drude-Lorentz model fit.
        
    2. Polarizability, induced dipolar moment, field, scattering, absorption, 
    power and temperature tools.
        This functions compute physical magnitudes by applying different 
        electromagnetic models and aproximations. Cuasistatic dipolar 
        approximation, Clausius-Mosotti and Kuwata models, Mie theory and 
        Gaussian beam expressions are used.

Some of its most useful tools are...

epsilon_function : function
    Generates an interpolation function for epsilon from experimental data.
alpha_Clausius_Mosotti : function
    Returns Clausius-Mosotti polarizability alpha for a sphere in units of volume.
alpha_Kuwata : function
    Returns Kuwata polarizability alpha for a sphere in units of volume.
p_induced : function
    Returns induced dipolar moment for a sphere in units of volume times field.
E : function
    Returns electric field for a sphere for all positions in units of field.
sigma_scatt : function
    Calculates scattering cross section using Mie theory for a spherical NP.
sigma_abs : function
    Calculates absorption cross section using Mie theory for a spherical NP.
delta_T : function
    Surface temperature increasement caused by a focused Gaussian beam on a NP.
P : function
    Gaussian focused power that causes a surface temperature change on a NP.

@author: Vall
"""

from socket import gethostname
if "Nano" in gethostname():
    syshome = "/home/nanofisica/Documents/Vale/ThesisPython"
elif "vall" in gethostname():
    syshome = "/home/vall/Documents/Thesis/ThesisPython"
else:
    raise ValueError("Your PC must be registered at the top of this code")

import numpy as np
import os
try:
    import PyMieScatt as ps
except:
    raise OSError("Must install PyMieScatt module using, for example, `pip install PyMieScatt`")
from scipy.interpolate import interp1d
try:
    import v_meep as vm
except:
    print("Meep functions not available. Don't worry! You can still use everything else")
import v_utilities as vu

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
        
    medium = vm.import_medium(material, 
                              paper=paper) # This one has from_um_factor=1
    
    print(f"Data loaded using Meep and '{paper}'")
    epsilon_function = lambda wlen : medium.epsilon(1/(wlen*from_um_factor))[0,0]
    # To pass it to the medium, I transform wavelength from nm (or Meep units) to um
    
    return epsilon_function

def epsilon_function_from_file(material="Au", paper="JC", reference="RIinfo"):
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
    elif "eps" in file.lower:
        epsilon_function = epsilon_interpoler_from_epsilon(
            data[:,0], data[:,1] + 1j*data[:,2])
            
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

#%% CLAUSIUS-MOSETTI: POLARIZABILITY

def alpha_Clausius_Mosotti(epsilon, r, epsilon_ext=1):
    """Returns Clausius-Mosotti polarizability alpha in units of volume"""
    alpha = 4 * np.pi * (r**3)
    alpha = alpha * ( epsilon - epsilon_ext ) / ( epsilon + 2 * epsilon_ext )
    return alpha
# In units of nm^3

#%% KUWATA: POLARIZABILITY

def alpha_Kuwata(epsilon, wlen, r, epsilon_ext=1):
    """Returns Kuwata polarizability alpha in units of volume"""
    aux_x = np.pi * r / wlen # Withouth units, so no need for from_um_factor * 1e3
    aux_vol = 4 * np.pi * (r**3) / 3
    alpha = aux_vol * ( 1 - ( (epsilon + epsilon_ext) * ( aux_x**2 ) / 10 ) )
    aux_den = ( 1/3 ) + ( epsilon_ext / ( epsilon - epsilon_ext ) )
    aux_den = aux_den - ( epsilon + 10 * epsilon_ext ) * ( aux_x**2 ) / 30
    aux_den = aux_den - 4j * (np.pi**2) * (epsilon_ext**(3/2) ) * aux_vol / (3 * (wlen**3))
    alpha = alpha / aux_den
    return alpha

#%% DIPOLAR APROXIMATION: INDUCED DIPOLE MOMENT 

def p_induced(epsilon, alpha, E0, epsilon_ext=1):
    """Returns induced dipolar moment in units of volume times field.
    
    Units asume vacuum permitivity epsilon_0 = 1.
    """
    if isinstance(epsilon, np.ndarray):
        aux = np.array([e * epsilon_ext * a for e, a in zip(epsilon, alpha)])
        p = np.array([E0 * a for a in aux])
    else:
        p = epsilon * epsilon_ext * alpha * E0
    return p

#%% DIPOLAR APROXIMATION: ELECTRIC FIELD

def E_in(epsilon, E0, epsilon_ext=1):
    """Returns electric field inside the sphere in units of field.
    
    Units asume vacuum permitivity epsilon_0 = 1.
    """
    aux = 3 * epsilon_ext / (epsilon + 2 * epsilon_ext)
    if isinstance(aux, np.ndarray):
        E_in = np.array([E0 * a for a in aux])
    else:
        E_in = E0 * aux
    return E_in

def E_out(epsilon, alpha, E0, rvec, epsilon_ext=1):
    """Returns electric field outside the sphere in units of field.
    
    Units asume vacuum permitivity epsilon_0 = 1.
    """
    rmod = np.linalg.norm(rvec)
    rver = rvec / rmod
    p = p_induced(epsilon, alpha, E0)
    if p.ndim > 1:
        aux = np.array([3 * rver * np.dot(rver, pi) - pi for pi in p])
    else:
        aux = 3 * rver * np.dot(rver, p) - p
    if isinstance(aux, np.ndarray) and isinstance(epsilon, np.ndarray):
        Eout = np.array([E0 + a / (4 * np.pi * e * epsilon_ext * (rmod**3)) for a, e in zip(aux, epsilon)])
    else:
        Eout = E0 + aux / (4 * np.pi * epsilon * epsilon_ext * (rmod**3))
    # Eout = E0 + aux / (4 * np.pi * epsilon * epsilon_ext * (rmod**3))
    return Eout

def E(epsilon, alpha, E0, rvec, r, epsilon_ext=1):
    """Returns electric field for a sphere for all positions in units of field.
    
    Units asume vacuum permitivity epsilon_0 = 1.
    """
    rmod = np.linalg.norm(rvec)
    if rmod <= r:
        return E_in(epsilon, E0)
    else:
        return E_out(epsilon, alpha, E0, rvec)
# First index: wavelength
# Second index: direction

#%% SCATTERING AND ABSORPTION CROSS SECTION

def sigma_scatt(r, wlen, inner_N=1.458, surrounding_N=1, asEffiency=False):
    """
    Calculates scattering cross section using Mie theory for a spherical NP.

    Parameters
    ----------
    r : float
        Spherical NP radius. Measured in nm.
    wlen : float, list, np.array
        Incident light wavelength. Measured in nm.
    inner_N=1.458 : float, list, np.array
        Spherical NP inner medium's complex refractive index. The default is 
        1.458 for fused-silica.
    surrounding_N=1 : float, list, np.array
        Surrounding medium's complex refractive index. The default is 
        1 for vacuum.
    asEffiency : bool, optional
        If false, scattering cross section sigma is returned, measured in nm^2. 
        If true, scattering effienciency Q is returned, dimensionless and 
        related to scattering cross section by sigma = pi r^2 Q 
        for a spherical NP. The default is False.

    Returns
    -------
    sigma_scatt : float, np.array
        The scattering cross section calculated using Mie theory and measured 
        in nm^2. In case `asEfficiency=True`, scattering effiency is returned 
        instead, dimensionless.
        
    Raises
    ------
    ValueError : "Must have as many inner refractive index values as..."
        If the length of `wlen` and `inner_N` differ.
    ValueError : "Must have as many surrounding refractive index values as..."
        If the length of `wlen` and `surrounding_N` differ.

    """
    
    try:
        wlen = np.array([*wlen])
    except:
        wlen = np.array([wlen])
    
    try:
        if len(inner_N)==len(wlen):
            inner_N = np.array([*inner_N])
        else:
            raise ValueError("Must have as many inner refractive index values as wavelength values")
    except:
        inner_N = [*[inner_N]*len(wlen)]
    
    try:
        if len(inner_N)==len(wlen):
            surrounding_N = np.array([*surrounding_N])
        else:
            raise ValueError("Must have as many surrounding refractive index values as wavelength values")
    except:
        surrounding_N = [*[surrounding_N]*len(wlen)]
    
    sigma_scatt = np.array([ps.MieQ(
        iN, wl, 2*r, sN, asCrossSection=not(asEffiency))[1] 
            for wl, iN, sN in zip(wlen, inner_N, surrounding_N)])
    
    if len(sigma_scatt)>1:
        return sigma_scatt
    else:
        return sigma_scatt[0]
    
def sigma_abs(r, wlen, inner_N=1.458, surrounding_N=1, asEffiency=False):
    """
    Calculates absorption cross section using Mie theory for a spherical NP.

    Parameters
    ----------
    r : float
        Spherical NP radius. Measured in nm.
    wlen : float, list, np.array
        Incident light wavelength. Measured in nm.
    inner_N=1.458 : float, list, np.array
        Spherical NP inner medium's complex refractive index. The default is 
        1.458 for fused-silica.
    surrounding_N=1 : float, list, np.array
        Surrounding medium's complex refractive index. The default is 
        1 for vacuum.
    asEffiency : bool, optional
        If false, scattering cross section sigma is returned, measured in nm^2. 
        If true, scattering effienciency Q is returned, dimensionless and 
        related to scattering cross section by sigma = pi r^2 Q 
        for a spherical NP. The default is False.

    Returns
    -------
    sigma_abs : float, np.array
        The scattering cross section calculated using Mie theory and measured 
        in nm^2. In case `asEfficiency=True`, scattering effiency is returned 
        instead, dimensionless.
        
    Raises
    ------
    ValueError : "Must have as many inner refractive index values as..."
        If the length of `wlen` and `inner_N` differ.
    ValueError : "Must have as many surrounding refractive index values as..."
        If the length of `wlen` and `surrounding_N` differ.
    """
    
    try:
        wlen = np.array([*wlen])
    except:
        wlen = np.array([wlen])
    
    try:
        if len(inner_N)==len(wlen):
            inner_N = np.array([*inner_N])
        else:
            raise ValueError("Must have as many inner refractive index values as wavelength values")
    except:
        inner_N = [*[inner_N]*len(wlen)]
    
    try:
        if len(inner_N)==len(wlen):
            surrounding_N = np.array([*surrounding_N])
        else:
            raise ValueError("Must have as many surrounding refractive index values as wavelength values")
    except:
        surrounding_N = [*[surrounding_N]*len(wlen)]
    
    sigma_abs = np.array([ps.MieQ(
        iN, wl, 2*r, sN, asCrossSection=not(asEffiency))[2] 
            for wl, iN, sN in zip(wlen, inner_N, surrounding_N)])
    
    if len(sigma_scatt)>1:
        return sigma_abs
    else:
        return sigma_abs[0]

#%% TEMPERATURE

def delta_T(P, sigma_abs, w0, r, kappa):
    """
    Surface temperature increasement caused by a focused Gaussian beam on a NP.

    Parameters
    ----------
    P : float
        Total power of the beam. Measured in mW.
    sigma_abs : float
        Absorption cross section of the nanoparticle NP. Measured in nm^2.
    kappa : float
        Thermal conductivity of the sourrounding medium where the NP is 
        inmersed. Measured in W / K m.
    r : float
        Radius of the NP. Measured in nm.
    w0 : float
        Laser beam waist. Measured in nm.

    Returns
    -------
    delta_T : float
        Temperature increasement caused by the focused Gaussian beam on the 
        surface of the NP. Measured in K or ºC.
    """
    
    kappa = kappa * 1000 / (1e9) # From W/Km to mW/Knm
    delta_T = sigma_abs * P / (2 * (np.pi**2) * kappa * r * (w0**2))
    
    return delta_T

def P(delta_T, sigma_abs, w0, r, kappa):
    """
    Gaussian focused power that causes a surface temperature change on a NP.

    Parameters
    ----------
    delta_T : float
        Temperature increasement caused by the focused Gaussian beam on the 
        surface of the NP. Measured in K or ºC.
    sigma_abs : float
        Absorption cross section of the nanoparticle NP. Measured in nm^2.
    kappa : float
        Thermal conductivity of the sourrounding medium where the NP is 
        inmersed. Measured in W / K m.
    r : float
        Radius of the NP. Measured in nm.
    w0 : float
        Laser beam waist. Measured in nm.

    Returns
    -------
    delta_T : float
        Total power of the focused Gaussian beam. Measured in mW.
    """
    
    kappa = kappa * 1000 / (1e9) # From W/Km to mW/Knm
    P = 2 * (np.pi**2) * kappa * r * (w0**2) * delta_T / sigma_abs
    
    return P