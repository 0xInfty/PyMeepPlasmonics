#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 22:51:24 2021

@author: vall
"""

from socket import gethostname
if "Nano" in gethostname():
    syshome = "/home/nanofisica/Documents/Vale/ThesisPython"
elif "vall" in gethostname():
    syshome = "/home/vall/Documents/Thesis/ThesisPython"
else:
    raise ValueError("Your PC must be registered at the top of this code")

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
import v_materials as vm
import v_save as vs

#%% PARAMETERS

from_um_factor = 10e-3
r = 3
wlen_range = np.array([35, 70])
epsilon_ext = 1 # Vacuum
medium = vm.import_medium("Au", from_um_factor)
E0 = lambda n : np.array([[0,0,1] for i in range(n)])

npoints = 500

data_folder = "AuMieSphere/AuSphereField"
data_file = "RefractiveIndexJohnson.csv"
plot_folder = "DataAnalysis"

#%% SAVING CONFIGURATION

home = vs.get_home()
plot_file = lambda n : os.path.join(
    home, plot_folder, f"VacuumField{2*r*from_um_factor*1e3:1.0f}" + n + '.png')
data_file = os.path.join(home, data_folder, data_file)

#%% INNER MEDIUM EPSILON

wlen_meep = np.linspace(*wlen_range, npoints)

freq_meep = 1/wlen_meep
epsilon_meep = np.array([medium.epsilon(f) for f in freq_meep]) # Tensor
epsilon_meep = np.array([et[0,0] for et in epsilon_meep]) # Isotropic

data_juli = np.loadtxt(data_file, delimiter=',', skiprows=1)
wlen_juli = data_juli[:,0] * 1e3 # Wavelength in nm
n_juli = data_juli[:,1] + 1j * data_juli[:,2] # Refractive Index
epsilon_juli = np.power(n_juli, 2) # Because permeability is mu = 1

n_real_john = interp1d(wlen_juli, data_juli[:,1], kind="cubic")
n_imag_john = interp1d(wlen_juli, data_juli[:,2], kind="cubic")
n_func_john = lambda wl : n_real_john(wl) + 1j * n_imag_john(wl)
epsilon_func_john = lambda wl : np.power(n_func_john(wl), 2)
epsilon_john = epsilon_func_john(wlen_meep * from_um_factor * 1e3)

epsilon = [epsilon_meep, epsilon_john]
wlen = [*[wlen_meep * from_um_factor * 1e3]*2]
# epsilon = [epsilon_meep, epsilon_juli]
# wlen = [wlen_meep * from_um_factor * 1e3, wlen_juli]

#%% CLAUSIUS-MOSETTI: POLARIZABILITY

def alpha_Clausius_Mosotti(epsilon, r):
    """Returns Clausius-Mosotti polarizability alpha in units of cubic length"""
    alpha = 4 * np.pi * (r**3)
    alpha = alpha * ( epsilon - epsilon_ext ) / ( epsilon + 2 * epsilon_ext )
    return alpha
# In units of nm^3

alpha_cm = [alpha_Clausius_Mosotti(e, r * from_um_factor * 1e3) for e in epsilon]   

#%% KUWATA: POLARIZABILITY

def alpha_Kuwata(epsilon, wlen, r):
    """Returns Kuwata polarizability alpha in units of nm³"""
    aux_x = np.pi * r / wlen # Withouth units, so no need for from_um_factor * 1e3
    aux_vol = 4 * np.pi * (r**3) / 3
    alpha = aux_vol * ( 1 - ( (epsilon + epsilon_ext) * ( aux_x**2 ) / 10 ) )
    aux_den = ( 1/3 ) + ( epsilon_ext / ( epsilon - epsilon_ext ) )
    aux_den = aux_den - ( epsilon + 10 * epsilon_ext ) * ( aux_x**2 ) / 30
    aux_den = aux_den - 4j * (np.pi**2) * (epsilon_ext**(3/2) ) * aux_vol / (3 * (wlen**3))
    alpha = alpha / aux_den
    return alpha
    
alpha_k = [alpha_Kuwata(eps, wl, r * from_um_factor * 1e3) for eps, wl in zip(epsilon, wlen)]

#%% PLOT POLARIZABILITY

functions = [np.abs, np.real, np.imag]
titles = ["Absolute value", "Real part", "Imaginary part"]
ylabels = [r"|$\alpha$| [nm$^3$]", r"Re($\alpha$) [nm$^3$]", r"Im($\alpha$) [nm$^3$]"]
wlen_alpha = [*[*wlen]*2]
alpha = [*alpha_cm, *alpha_k]
labels = ["Meep: Claussius-Mosotti", "Jhonson: Claussius-Mosotti", 
          "Meep: Kuwata", "Jhonson: Kuwata"]

nplots = len(functions)
fig = plt.figure(figsize=(nplots*6.4, 6.4))
axes = fig.subplots(ncols=nplots)

max_value = []
for ax, f, t, y in zip(axes, functions, titles, ylabels):
    for wl, a, l in zip(wlen_alpha, alpha, labels):
        ax.set_title(t)
        ax.plot(wl, f(a), label=l)
        ax.xaxis.set_label_text("Wavelength [nm]")
        ax.yaxis.set_label_text(y)
        ax.legend()
        ax.set_xlim(*(wlen_range * from_um_factor * 1e3))
        max_value.append(max(f(a)))
        
for ax in axes: ax.set_ylim([0, 1.1*max(max_value)])
    
plt.savefig(plot_file("Polarizability.png"))

#%% DIPOLAR APROXIMATION: INDUCED DIPOLE MOMENT 

def p(epsilon, alpha, E0):
    """Returns induced dipolar moment in units of cubic length"""
    return epsilon * epsilon_ext * np.matmul(alpha, E0)

# alpha_cm = [alpha_Clausius_Mosotti(e, r * from_um_factor * 1e3) for e in epsilon]   

p_plane_cm = [p(epsilon_john, alpha, E0(npoints)) for alpha in [alpha_cm[1]]]

#%% DIPOLAR APROXIMATION: ELECTRIC FIELD

def E_in(epsilon, E0):
    """Returns electric field inside the sphere in units of cubic length"""
    return 3 * epsilon_ext * E0 / (epsilon + 2 * epsilon_ext)

def E_out(epsilon, E0, p, r):
    """Returns electric field outside the sphere in units of cubic length"""
    rmod = np.linalg.norm(r)
    rver = r / rmod
    aux = 3 * rver * np.dot(rver, p) - p
    Eout = E0 + aux / (4 * np.pi * epsilon * epsilon_ext * (rmod**3))
    return Eout

