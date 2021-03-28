# -*- coding: utf-8 -*-
"""
This module contains electromagnetism's functions.

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