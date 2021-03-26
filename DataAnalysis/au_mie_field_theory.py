#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 22:51:24 2021

@author: vall
"""

import numpy as np
import matplotlib.pyplot as plt
import v_materials as vm

#%% PARAMETERS

from_um_factor = 10e-3
r = 3
wlen_range = np.array([35, 70])
epsilon_ext = 1 # Vacuum
medium = vm.import_medium("Au", from_um_factor)

npoints = 500

#%% INNER MEDIUM EPSILON

wlen = np.linspace(*wlen_range, npoints)

freq = 1/wlen
epsilon_tensor_meep = np.array([medium.epsilon(f) for f in freq])
epsilon_meep = np.array([et[0,0] for et in epsilon_tensor_meep]) # Isotropic

n_juli = 0.27732 + 1j * 2.9278 
epsilon_juli = np.power(n_juli, 2) # Because permeability is mu = 1
epsilon_juli = np.array([epsilon_juli]*npoints)

#%% CLAUSIUS-MOSETTI POLARIZABILITY

alpha_cm = []
for epsilon in [epsilon_meep, epsilon_juli]:
    this_alpha = 4 * np.pi * (r * from_um_factor * 1e3)**3 
    this_alpha = this_alpha * ( epsilon - epsilon_ext ) / ( epsilon + 2 * epsilon_ext )
    alpha_cm.append(this_alpha)
# In units of nm^3

#%% KUWATA POLARIZABILITY

alpha_k = []
for epsilon in [epsilon_meep, epsilon_juli]:
    aux_x = np.pi * r / wlen # Withouth units, so no need for from_um_factor * 1e3
    aux_vol = 4 * np.pi * (r * from_um_factor * 1e3)**3 / 3
    this_alpha = aux_vol * ( 1 - ( (epsilon + epsilon_ext) * ( aux_x**2 ) / 10 ) )
    aux_den = ( 1/3 ) + ( epsilon_ext / ( epsilon - epsilon_ext ) )
    aux_den = aux_den - ( epsilon + 10 * epsilon_ext ) * ( aux_x**2 ) / 30
    aux_den = aux_den - 4j * (np.pi**2) * (epsilon_ext**(3/2) ) * aux_vol / (3 * (wlen * from_um_factor * 1e3)**3)
    this_alpha = this_alpha / aux_den
    alpha_k.append(this_alpha)
del aux_x, aux_vol, aux_den

#%% PLOT POLARIZABILITY

functions = [np.abs, np.real, np.imag]
titles = ["Absolute value", "Real part", "Imaginary part"]
ylabels = [r"|$\alpha$| [nm$^3$]", r"Re($\alpha$) [nm$^3$]", r"Im($\alpha$) [nm$^3$]"]
alpha = [*alpha_cm, *alpha_k]
labels = ["Jhonson: Claussius-Mosotti", "Jhonson: Kuwata",
          "Meep: Claussius-Mosotti", "Meep: Kuwata"]

nplots = len(functions)
fig = plt.figure(figsize=(nplots*6.4, 6.4))
axes = fig.subplots(ncols=nplots)

max_value = []
for ax, f, t, y in zip(axes, functions, titles, ylabels):
    for a, l in zip(alpha, labels):
        ax.set_title(t)
        ax.plot(wlen * from_um_factor * 1e3, f(a), label=l)
        ax.xaxis.set_label_text("Wavelength [nm]")
        ax.yaxis.set_label_text(y)
        ax.legend()
        ax.set_xlim(*(wlen_range * from_um_factor * 1e3))
        max_value.append(max(f(a)))
        
for ax in axes: ax.set_ylim([0, 1.1*max(max_value)])
    
# plt.savefig(file("MaxFieldPlane.png"))
