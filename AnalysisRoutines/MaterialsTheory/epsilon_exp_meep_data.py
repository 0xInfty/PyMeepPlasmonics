#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 13:59:19 2021

@author: vall
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import vmp_materials as vml
import v_save as vs

home = vs.get_home()

#%% PARAMETERS

from_um_factor = 1

material = "Au"
paper = "JC"
r = 60
submerged_index = 1

plot_file = lambda n : os.path.join(home, "DataAnalysis/MaterialsPaper", n)

#%% VARIABLES

medium = vml.import_medium(material, from_um_factor=from_um_factor, paper=paper)

freq_range = medium.valid_freq_range

freqs = np.linspace(*freq_range, 100)
wlens = 1 / freqs # nm

#%% GET N

wlen_data_exp, n_data_exp = vml.n_data_from_file(material, paper, "RIinfo")

n_function_meep = vml.n_function_from_meep(material, paper, from_um_factor)

n_function_exp = vml.n_function_from_file(material, paper, "RIinfo", from_um_factor)

#%% PLOT N

functions = [np.abs, np.real, np.imag]
titles = ["Absolute value", "Real part", "Imaginary part"]
ylabels = [r"|N|", r"Re(N)", r"Im(N)"]
# ylabels = [r"|$\epsilon$|", r"Re($\epsilon$)", r"Im($\epsilon$)"]
long_wlen = [wlen_data_exp, *[1e3 * from_um_factor * wlens]*2]
long_n = [n_data_exp,
          np.array([n_function_exp(wl) for wl in wlens]), 
          np.array([n_function_meep(wl) for wl in wlens])]
labels = ["JC Experimental Data", 
          "JC Experimental Data Interpolation", 
          "Meep Drude-Lorentz Model JC Interpolation"]
linestyles = [".", "-", ":"]
colors = ["k", 'k', 'r']

nplots = len(functions)
fig = plt.figure(figsize=(nplots*6.4, 6.4), tight_layout=True)
axes = fig.subplots(ncols=nplots)

max_value = []
min_value = []
for ax, f, t, y in zip(axes, functions, titles, ylabels):
    for wl, n, l, lst, col in zip(long_wlen, long_n, labels, linestyles, colors):
        ax.set_title(t)
        ax.plot(wl, f(n), lst, color=col, label=l)
        ax.xaxis.set_label_text(rr"Wavelength $\lambda$ [nm]")
        ax.yaxis.set_label_text(y)
        ax.legend()
        ax.set_xlim(min(wlens) * 1e3 * from_um_factor, 
                    max(wlens) * 1e3 * from_um_factor)
        max_value.append(max(f(n)[wl < 1e3 * from_um_factor * max(wlens)]))
        min_value.append(min(f(n)[wl < 1e3 * from_um_factor * max(wlens)]))
        
for ax in axes: ax.set_ylim([min(min_value)-.1*(max(max_value)-min(min_value)), 
                              max(max_value)+.1*(max(max_value)-min(min_value))])

vs.saveplot( plot_file(f"N{material}MeepN.png"), overwrite=True )

#%% GET EPSILON

wlen_data_exp, epsilon_data_exp = vml.epsilon_data_from_file(material, 
                                                             paper, "RIinfo")

epsilon_function_meep = vml.epsilon_function_from_meep(material, paper, 
                                                       from_um_factor)

epsilon_function_exp = vml.epsilon_function_from_file(material, paper, 
                                                      "RIinfo", from_um_factor)

#%% PLOT N

functions = [np.abs, np.real, np.imag]
titles = ["Absolute value", "Real part", "Imaginary part"]
ylabels = [r"|$\epsilon$|", r"Re($\epsilon$)", r"Im($\epsilon$)"]
long_wlen = [wlen_data_exp, *[1e3 * from_um_factor * wlens]*2]
long_epsilon = [epsilon_data_exp,
                np.array([epsilon_function_exp(wl) for wl in wlens]), 
                np.array([epsilon_function_meep(wl) for wl in wlens])]
labels = ["JC Experimental Data", 
          "JC Experimental Data Interpolation", 
          "Meep Drude-Lorentz Model JC Interpolation"]
linestyles = [".", "-", ":"]
colors = ["k", 'k', 'r']

nplots = len(functions)
fig = plt.figure(figsize=(nplots*6.4, 6.4), tight_layout=True)
axes = fig.subplots(ncols=nplots)

max_value = []
min_value = []
for ax, f, t, y in zip(axes, functions, titles, ylabels):
    for wl, eps, l, lst, col in zip(long_wlen, long_epsilon, labels, linestyles, colors):
        ax.set_title(t)
        ax.plot(wl, f(eps), lst, color=col, label=l)
        ax.xaxis.set_label_text(rr"Wavelength $\lambda$ [nm]")
        ax.yaxis.set_label_text(y)
        ax.legend()
        ax.set_xlim(min(wlens) * 1e3 * from_um_factor, 
                    max(wlens) * 1e3 * from_um_factor)
        max_value.append(max(f(eps)[wl < 1e3 * from_um_factor * max(wlens)]))
        min_value.append(min(f(eps)[wl < 1e3 * from_um_factor * max(wlens)]))
        
for ax in axes: ax.set_ylim([min(min_value)-.1*(max(max_value)-min(min_value)), 
                              max(max_value)+.1*(max(max_value)-min(min_value))])

vs.saveplot( plot_file(f"Eps{material}MeepN.png"), overwrite=True )