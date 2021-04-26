# -*- coding: utf-8 -*-
"""
This module contains electromagnetic theory and base EM data for materials.

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
from scipy.interpolate import interp1d
import v_meep as vm

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

def epsilon_function_from_meep(material="Au", source="JC", from_ref="RIinfo",
                               medium=None, from_um_factor=1):
    
    """
    Generates a function for isotropic epsilon from Meep Drude-Lorentz fit data.
    
    Parameters
    ----------
    material="Au" : str
        Material's chemical symbol. Available: 'Au' for gold.
    source="JC" : str
        Source of experimental data. Available: 'JC' for Johnson and Christy, 
        'R' for Rakic.
    from_ref="RIinfo" : str
        Reference from which the data was extracted, for example a web page. 
        Available: 'RIinfo' for 'www.refractiveindex.info'
        
    Returns
    -------
    epsilon_function : function
        Epsilon function that takes wavelength in nm as argument and returns 
        complex dielectric constant or relative permitivitty 
        epsilon = epsilon' + i epsilon'', dimensionless.
    """

    
    if medium==None:
        medium = vm.import_medium(material, 
                                  from_um_factor=from_um_factor, 
                                  source=source)
    
    epsilon_function = lambda wlen : medium.epsilon(1e3*from_um_factor/wlen)[0,0]
    
    return epsilon_function

def epsilon_function_from_file(material="Au", source="JC", from_ref="RIinfo"):
    """
    Generates an interpolation function for epsilon from experimental data.
    
    Parameters
    ----------
    material="Au" : str
        Material's chemical symbol. Available: 'Au' for gold.
    source="JC" : str
        Source of experimental data. Available: 'JC' for Johnson and Christy, 
        'R' for Rakic.
    from_ref="RIinfo" : str
        Reference from which the data was extracted, for example a web page. 
        Available: 'RIinfo' for 'www.refractiveindex.info'
        
    Returns
    -------
    epsilon_function : function
        Epsilon interpoler that takes wavelength in nm as argument and returns 
        complex dielectric constant or relative permitivitty 
        epsilon = epsilon' + i epsilon'', dimensionless.
    """
    
    data_series = os.listdir(os.path.join(syshome, 'MaterialsData'))
    
    try:
        data_files = []
        for df in data_series:
            if source in df and material in df and from_ref in df:
                data_files.append( os.path.join(syshome, 'MaterialsData', df) )
    except:
        raise ValueError("Experimental data couldn't be found. Sorry!")
    
    if len(data_files)>1:
        for df in data_files:
            if "eps" in df.lower():
                file = df
                break
            file = df
    else:
        file = data_files[0]
    
    data = np.loadtxt(file)
    
    if 'N' in file:
        epsilon_function = epsilon_interpoler_from_n(
            data[:,0], data[:,1] + 1j*data[:,2])
    elif "eps" in file.lower:
        epsilon_function = epsilon_interpoler_from_epsilon(
            data[:,0], data[:,1] + 1j*data[:,2])
            
    return epsilon_function

#%% CLAUSIUS-MOSETTI: POLARIZABILITY

def alpha_Clausius_Mosotti(epsilon, r, epsilon_ext=1):
    """Returns Clausius-Mosotti polarizability alpha in units of cubic length"""
    alpha = 4 * np.pi * (r**3)
    alpha = alpha * ( epsilon - epsilon_ext ) / ( epsilon + 2 * epsilon_ext )
    return alpha
# In units of nm^3

#%% KUWATA: POLARIZABILITY

def alpha_Kuwata(epsilon, wlen, r, epsilon_ext=1):
    """Returns Kuwata polarizability alpha in units of nm³"""
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
    """Returns induced dipolar moment in units of cubic length"""
    if isinstance(epsilon, np.ndarray):
        aux = np.array([e * epsilon_ext * a for e, a in zip(epsilon, alpha)])
        p = np.array([E0 * a for a in aux])
    else:
        p = epsilon * epsilon_ext * alpha * E0
    return p

#%% DIPOLAR APROXIMATION: ELECTRIC FIELD

def E_in(epsilon, E0, epsilon_ext=1):
    """Returns electric field inside the sphere in units of cubic length"""
    aux = 3 * epsilon_ext / (epsilon + 2 * epsilon_ext)
    if isinstance(aux, np.ndarray):
        E_in = np.array([E0 * a for a in aux])
    else:
        E_in = E0 * aux
    return E_in

def E_out(epsilon, alpha, E0, rvec, epsilon_ext=1):
    """Returns electric field outside the sphere in units of cubic length"""
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
    """Returns electric field"""
    rmod = np.linalg.norm(rvec)
    if rmod <= r:
        return E_in(epsilon, E0)
    else:
        return E_out(epsilon, alpha, E0, rvec)
# First index: wavelength
# Second index: direction