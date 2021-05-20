# -*- coding: utf-8 -*-
"""
This module contains electromagnetic theory.

It holds tools to work with...

    - Polarizability
    - Induced dipolar moment
    - Electric field
    - Scattering and absorption
    - Power and temperature
    
These functions compute physical magnitudes by applying different 
electromagnetic models and aproximations. Cuasistatic dipolar 
approximation, Clausius-Mosotti and Kuwata models, Mie theory and 
Gaussian beam expressions are used.

Some of its most useful tools are...

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
try:
    import PyMieScatt as ps
except:
    raise OSError("Must install PyMieScatt module using, for example, `pip install PyMieScatt`")

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