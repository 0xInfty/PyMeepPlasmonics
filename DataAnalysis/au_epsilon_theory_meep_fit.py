#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 15:52:32 2021

@author: vall
"""

import lmfit as lm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plab
import os
import sys
import v_save as vs
import v_materials as vmt

syshome = vs.get_sys_home()
home = vs.get_home()
epsilon = sys.float_info.epsilon

#%%

material = "Ag"
from_um_factor = 1

plot_file = lambda n : os.path.join(home, "DataAnalysis/MaterialsPaper", n)

#%%

epsilon_function_r = vmt.epsilon_function_from_file(material=material,
                                                    paper="R",
                                                    reference="RIinfo",
                                                    from_um_factor=from_um_factor)

wlen_range_r = epsilon_function_r._wlen_range_

wlens_r = np.linspace(*wlen_range_r, 500)#+2)[1:-1]
freqs_r = 1/wlens_r

#%%

def drude_susceptibility_fit(freq, freq_0, gamma_0, sigma_0):
    """Single dispersive susceptibility of Drude form."""
    return - sigma_0 * freq_0 * freq_0 / (freq*(freq + 1j*gamma_0))

def lorentz_susceptibility_fit(freq, freq_i, gamma_i, sigma_i):
    """Dispersive susceptibility of Lorentzian (damped harmonic oscillator) form"""
    return sigma_i * freq_i * freq_i / (freq_i*freq_i - freq*freq - 1j*gamma_i*freq)

do_print = True

def log_print(*strings): 
    if do_print: print(*strings)

def added_susceptibilities_fit_maker(n_terms):
    
    def added_susceptibilities_fit(freq, *args):
        
        result = drude_susceptibility_fit(freq, args[0], args[1], args[2])
        log_print("Added 1 Drude susceptibility with ",
              f"\n> freq_0 = {args[0]}",
              f"\n> gamma_0 = {args[1]}",
              f"\n> sigma_0 = {args[2]}")
        for j in range(3, 3+3*n_terms, 3):
            result += lorentz_susceptibility_fit(freq, args[j], args[j+1], args[j+2])
            log_print("Added 1 Lorentz susceptibility with ",
                  f"\n> freq_{int(j/3)} = {args[j]}",
                  f"\n> gamma_{int(j/3)} = {args[j+1]}",
                  f"\n> sigma_{int(j/3)} = {args[j+2]}")
        return result
    
    return added_susceptibilities_fit

#%%

def fit_params_r(from_um_factor, material):
    
    eV_from_um_factor = from_um_factor/1.23984193 # Conversion factor: eV to 1/um [=1/hc]
    
    params = {}
    
    if material=="Au":
    
        Au_plasma_frq = 9.03*eV_from_um_factor
        Au_f0 = 0.760
        params["freq_0"] = 1e-10
        params["gamma_0"] = 0.053*eV_from_um_factor
        params["sigma_0"] = Au_f0 * Au_plasma_frq**2 / params["freq_0"]**2
        Au_f1 = 0.024
        params["freq_1"] = 0.415*eV_from_um_factor      # 2.988 um
        params["gamma_1"] = 0.241*eV_from_um_factor
        params["sigma_1"] = Au_f1 * Au_plasma_frq**2 / params["freq_1"]**2
        Au_f2 = 0.010
        params["freq_2"] = 0.830*eV_from_um_factor      # 1.494 um
        params["gamma_2"] = 0.345*eV_from_um_factor
        params["sigma_2"] = Au_f2 * Au_plasma_frq**2 / params["freq_2"]**2
        Au_f3 = 0.071
        params["freq_3"] = 2.969*eV_from_um_factor      # 0.418 um
        params["gamma_3"] = 0.870*eV_from_um_factor
        params["sigma_3"] = Au_f3 * Au_plasma_frq**2 / params["freq_3"]**2
        Au_f4 = 0.601
        params["freq_4"] = 4.304*eV_from_um_factor      # 0.288 um
        params["gamma_4"] = 2.494*eV_from_um_factor
        params["sigma_4"] = Au_f4 * Au_plasma_frq**2 / params["freq_4"]**2
        Au_f5 = 4.384
        params["freq_5"] = 13.32*eV_from_um_factor      # 0.093 um
        params["gamma_5"] = 2.214*eV_from_um_factor
        params["sigma_5"] = Au_f5 * Au_plasma_frq**2 / params["freq_5"]**2
    
    elif material=="Ag":
    
        Ag_plasma_frq = 9.01*eV_from_um_factor
        Ag_f0 = 0.845
        params["freq_0"] = 1e-10
        params["gamma_0"] = 0.048*eV_from_um_factor
        params["sigma_0"] = Ag_f0 * Ag_plasma_frq**2 / Ag_f0**2
        Ag_f1 = 0.065
        params["freq_1"] = 0.816*eV_from_um_factor      # 1.519 um
        params["gamma_1"] = 3.886*eV_from_um_factor
        params["sigma_1"] = Ag_f1 * Ag_plasma_frq**2 / params["freq_1"]**2
        Ag_f2 = 0.124
        params["freq_2"] = 4.481*eV_from_um_factor      # 0.273 um
        params["gamma_2"] = 0.452*eV_from_um_factor
        params["sigma_2"] = Ag_f2 * Ag_plasma_frq**2 / params["freq_2"]**2
        Ag_f3 = 0.011
        params["freq_3"] = 8.185*eV_from_um_factor      # 0.152 um
        params["gamma_3"] = 0.065*eV_from_um_factor
        params["sigma_3"] = Ag_f3 * Ag_plasma_frq**2 / params["freq_3"]**2
        Ag_f4 = 0.840
        params["freq_4"] = 9.083*eV_from_um_factor      # 0.137 um
        params["gamma_4"] = 0.916*eV_from_um_factor
        params["sigma_4"] = Ag_f4 * Ag_plasma_frq**2 / params["freq_4"]**2
        Ag_f5 = 5.646
        params["freq_5"] = 20.29*eV_from_um_factor      # 0.061 um
        params["gamma_5"] = 2.419*eV_from_um_factor
        params["sigma_5"] = Ag_f5 * Ag_plasma_frq**2 / params["freq_5"]**2
        
    else:
        
        raise ValueError("Material not recognized. Choose 'Au' or 'Ag'")
    
    return params
    
params_r = fit_params_r(from_um_factor, material)

#%%

fit_function_r = added_susceptibilities_fit_maker(5)
    
fitted_epsilon_r = lambda freqs : fit_function_r(freqs, 
    params_r["freq_0"], params_r["gamma_0"], params_r["sigma_0"],
    params_r["freq_1"], params_r["gamma_1"], params_r["sigma_1"],
    params_r["freq_2"], params_r["gamma_2"], params_r["sigma_2"],
    params_r["freq_3"], params_r["gamma_3"], params_r["sigma_3"],
    params_r["freq_4"], params_r["gamma_4"], params_r["sigma_4"],
    params_r["freq_5"], params_r["gamma_5"], params_r["sigma_5"])

#%%

functions = [np.abs, np.real, np.imag]
titles = ["Absolute value", "Real part", "Imaginary part"]
ylabels = [r"|$\epsilon$|", r"Re($\epsilon$)", r"Im($\epsilon$)"]
long_wlen = [*[wlens_r]*2]
long_epsilon = [epsilon_function_r(wlens_r), fitted_epsilon_r(freqs_r)]
labels = ["Interpolation for Rakic", "Fit function using Meep parameters"]
styles = ["-C0", "--r"]

nplots = len(functions)
fig = plt.figure(figsize=(nplots*6.4, 6.4))
axes = fig.subplots(ncols=nplots)

max_value = []
min_value = []
for ax, f, t, y in zip(axes, functions, titles, ylabels):
    for wl, eps, l, st in zip(long_wlen, long_epsilon, labels, styles):
        ax.set_title(t)
        print(l, eps[:10])
        ax.plot(wl, f(eps), st, label=l)
        ax.xaxis.set_label_text(r"Wavelength [$\mu$m]")
        ax.yaxis.set_label_text(y)
        ax.legend()
        ax.set_xlim(*wlen_range_r)
        max_value.append(max(f(eps)))
        min_value.append(min(f(eps)))
        
for ax in axes: ax.set_ylim([min(min_value)-.1*(max(max_value)-min(min_value)), 
                             max(max_value)+.1*(max(max_value)-min(min_value))])

#%%

filename = f"{material}_JC_ComplexN_RIinfo.txt"
path = os.path.join(syshome, "MaterialsData", filename)
data_n_jc = np.loadtxt(path)
    
wlens_jc = data_n_jc[:,0] / (1e3 * from_um_factor)
wlen_range_jc = np.array([min(wlens_jc), max(wlens_jc)])
freqs_jc = 1/wlens_jc

epsilon_data_jc = np.power(data_n_jc[:,1] + 1j*data_n_jc[:,2], 2)

#%%

def added_susceptibilities_fit_0(freq, 
                                 freq_0, gamma_0, sigma_0):
        
    fit = drude_susceptibility_fit(freq, freq_0, gamma_0, sigma_0)
    
    return fit

def added_susceptibilities_fit_1(freq, 
                                 freq_0, gamma_0, sigma_0,
                                 freq_1, gamma_1, sigma_1):
    
    fit = drude_susceptibility_fit(freq, freq_0, gamma_0, sigma_0)
    fit += lorentz_susceptibility_fit(freq, freq_1, gamma_1, sigma_1)
    
    return fit

def added_susceptibilities_fit_2(freq, 
                                 freq_0, gamma_0, sigma_0,
                                 freq_1, gamma_1, sigma_1,
                                 freq_2, gamma_2, sigma_2):
    
    fit = drude_susceptibility_fit(freq, freq_0, gamma_0, sigma_0)
    fit += lorentz_susceptibility_fit(freq, freq_1, gamma_1, sigma_1)
    fit += lorentz_susceptibility_fit(freq, freq_2, gamma_2, sigma_2)
    
    return fit

def added_susceptibilities_fit_3(freq, 
                                 freq_0, gamma_0, sigma_0,
                                 freq_1, gamma_1, sigma_1,
                                 freq_2, gamma_2, sigma_2,
                                 freq_3, gamma_3, sigma_3):
    
    fit = drude_susceptibility_fit(freq, freq_0, gamma_0, sigma_0)
    fit += lorentz_susceptibility_fit(freq, freq_1, gamma_1, sigma_1)
    fit += lorentz_susceptibility_fit(freq, freq_2, gamma_2, sigma_2)
    fit += lorentz_susceptibility_fit(freq, freq_3, gamma_3, sigma_3)
    
    return fit

def added_susceptibilities_fit_4(freq, 
                                 freq_0, gamma_0, sigma_0,
                                 freq_1, gamma_1, sigma_1,
                                 freq_2, gamma_2, sigma_2,
                                 freq_3, gamma_3, sigma_3,
                                 freq_4, gamma_4, sigma_4):
    
    fit = drude_susceptibility_fit(freq, freq_0, gamma_0, sigma_0)
    fit += lorentz_susceptibility_fit(freq, freq_1, gamma_1, sigma_1)
    fit += lorentz_susceptibility_fit(freq, freq_2, gamma_2, sigma_2)
    fit += lorentz_susceptibility_fit(freq, freq_3, gamma_3, sigma_3)
    fit += lorentz_susceptibility_fit(freq, freq_4, gamma_4, sigma_4)
    
    return fit

def added_susceptibilities_fit_5(freq, 
                                 freq_0, gamma_0, sigma_0,
                                 freq_1, gamma_1, sigma_1,
                                 freq_2, gamma_2, sigma_2,
                                 freq_3, gamma_3, sigma_3,
                                 freq_4, gamma_4, sigma_4,
                                 freq_5, gamma_5, sigma_5):
    
    fit = drude_susceptibility_fit(freq, freq_0, gamma_0, sigma_0)
    fit += lorentz_susceptibility_fit(freq, freq_1, gamma_1, sigma_1)
    fit += lorentz_susceptibility_fit(freq, freq_2, gamma_2, sigma_2)
    fit += lorentz_susceptibility_fit(freq, freq_3, gamma_3, sigma_3)
    fit += lorentz_susceptibility_fit(freq, freq_4, gamma_4, sigma_4)
    fit += lorentz_susceptibility_fit(freq, freq_5, gamma_5, sigma_5)
    
    return fit

def added_susceptibilities_fit_6(freq, 
                                 freq_0, gamma_0, sigma_0,
                                 freq_1, gamma_1, sigma_1,
                                 freq_2, gamma_2, sigma_2,
                                 freq_3, gamma_3, sigma_3,
                                 freq_4, gamma_4, sigma_4,
                                 freq_5, gamma_5, sigma_5,
                                 freq_6, gamma_6, sigma_6):
    
    fit = drude_susceptibility_fit(freq, freq_0, gamma_0, sigma_0)
    fit += lorentz_susceptibility_fit(freq, freq_1, gamma_1, sigma_1)
    fit += lorentz_susceptibility_fit(freq, freq_2, gamma_2, sigma_2)
    fit += lorentz_susceptibility_fit(freq, freq_3, gamma_3, sigma_3)
    fit += lorentz_susceptibility_fit(freq, freq_4, gamma_4, sigma_4)
    fit += lorentz_susceptibility_fit(freq, freq_5, gamma_5, sigma_5)
    fit += lorentz_susceptibility_fit(freq, freq_6, gamma_6, sigma_6)
    
    return fit

def added_susceptibilities_fit_7(freq, 
                                 freq_0, gamma_0, sigma_0,
                                 freq_1, gamma_1, sigma_1,
                                 freq_2, gamma_2, sigma_2,
                                 freq_3, gamma_3, sigma_3,
                                 freq_4, gamma_4, sigma_4,
                                 freq_5, gamma_5, sigma_5,
                                 freq_6, gamma_6, sigma_6,
                                 freq_7, gamma_7, sigma_7):
    
    fit = drude_susceptibility_fit(freq, freq_0, gamma_0, sigma_0)
    fit += lorentz_susceptibility_fit(freq, freq_1, gamma_1, sigma_1)
    fit += lorentz_susceptibility_fit(freq, freq_2, gamma_2, sigma_2)
    fit += lorentz_susceptibility_fit(freq, freq_3, gamma_3, sigma_3)
    fit += lorentz_susceptibility_fit(freq, freq_4, gamma_4, sigma_4)
    fit += lorentz_susceptibility_fit(freq, freq_5, gamma_5, sigma_5)
    fit += lorentz_susceptibility_fit(freq, freq_6, gamma_6, sigma_6)
    fit += lorentz_susceptibility_fit(freq, freq_7, gamma_7, sigma_7)
    
    return fit

def added_susceptibilities_fit_8(freq, 
                                 freq_0, gamma_0, sigma_0,
                                 freq_1, gamma_1, sigma_1,
                                 freq_2, gamma_2, sigma_2,
                                 freq_3, gamma_3, sigma_3,
                                 freq_4, gamma_4, sigma_4,
                                 freq_5, gamma_5, sigma_5,
                                 freq_6, gamma_6, sigma_6,
                                 freq_7, gamma_7, sigma_7,
                                 freq_8, gamma_8, sigma_8):
    
    fit = drude_susceptibility_fit(freq, freq_0, gamma_0, sigma_0)
    fit += lorentz_susceptibility_fit(freq, freq_1, gamma_1, sigma_1)
    fit += lorentz_susceptibility_fit(freq, freq_2, gamma_2, sigma_2)
    fit += lorentz_susceptibility_fit(freq, freq_3, gamma_3, sigma_3)
    fit += lorentz_susceptibility_fit(freq, freq_4, gamma_4, sigma_4)
    fit += lorentz_susceptibility_fit(freq, freq_5, gamma_5, sigma_5)
    fit += lorentz_susceptibility_fit(freq, freq_6, gamma_6, sigma_6)
    fit += lorentz_susceptibility_fit(freq, freq_7, gamma_7, sigma_7)
    fit += lorentz_susceptibility_fit(freq, freq_8, gamma_8, sigma_8)
    
    return fit

def added_susceptibilities_fit_9(freq, 
                                 freq_0, gamma_0, sigma_0,
                                 freq_1, gamma_1, sigma_1,
                                 freq_2, gamma_2, sigma_2,
                                 freq_3, gamma_3, sigma_3,
                                 freq_4, gamma_4, sigma_4,
                                 freq_5, gamma_5, sigma_5,
                                 freq_6, gamma_6, sigma_6,
                                 freq_7, gamma_7, sigma_7,
                                 freq_8, gamma_8, sigma_8,
                                 freq_9, gamma_9, sigma_9):
    
    fit = drude_susceptibility_fit(freq, freq_0, gamma_0, sigma_0)
    fit += lorentz_susceptibility_fit(freq, freq_1, gamma_1, sigma_1)
    fit += lorentz_susceptibility_fit(freq, freq_2, gamma_2, sigma_2)
    fit += lorentz_susceptibility_fit(freq, freq_3, gamma_3, sigma_3)
    fit += lorentz_susceptibility_fit(freq, freq_4, gamma_4, sigma_4)
    fit += lorentz_susceptibility_fit(freq, freq_5, gamma_5, sigma_5)
    fit += lorentz_susceptibility_fit(freq, freq_6, gamma_6, sigma_6)
    fit += lorentz_susceptibility_fit(freq, freq_7, gamma_7, sigma_7)
    fit += lorentz_susceptibility_fit(freq, freq_8, gamma_8, sigma_8)
    fit += lorentz_susceptibility_fit(freq, freq_9, gamma_9, sigma_9)

    return fit

def added_susceptibilities_fit_10(freq, 
                                  freq_0, gamma_0, sigma_0,
                                  freq_1, gamma_1, sigma_1,
                                  freq_2, gamma_2, sigma_2,
                                  freq_3, gamma_3, sigma_3,
                                  freq_4, gamma_4, sigma_4,
                                  freq_5, gamma_5, sigma_5,
                                  freq_6, gamma_6, sigma_6,
                                  freq_7, gamma_7, sigma_7,
                                  freq_8, gamma_8, sigma_8,
                                  freq_9, gamma_9, sigma_9,
                                  freq_10, gamma_10, sigma_10):
    
    fit = drude_susceptibility_fit(freq, freq_0, gamma_0, sigma_0)
    fit += lorentz_susceptibility_fit(freq, freq_1, gamma_1, sigma_1)
    fit += lorentz_susceptibility_fit(freq, freq_2, gamma_2, sigma_2)
    fit += lorentz_susceptibility_fit(freq, freq_3, gamma_3, sigma_3)
    fit += lorentz_susceptibility_fit(freq, freq_4, gamma_4, sigma_4)
    fit += lorentz_susceptibility_fit(freq, freq_5, gamma_5, sigma_5)
    fit += lorentz_susceptibility_fit(freq, freq_6, gamma_6, sigma_6)
    fit += lorentz_susceptibility_fit(freq, freq_7, gamma_7, sigma_7)
    fit += lorentz_susceptibility_fit(freq, freq_8, gamma_8, sigma_8)
    fit += lorentz_susceptibility_fit(freq, freq_9, gamma_9, sigma_9)
    fit += lorentz_susceptibility_fit(freq, freq_10, gamma_10, sigma_10)    
    
    return fit

def added_susceptibilities_fit_11(freq, 
                                  freq_0, gamma_0, sigma_0,
                                  freq_1, gamma_1, sigma_1,
                                  freq_2, gamma_2, sigma_2,
                                  freq_3, gamma_3, sigma_3,
                                  freq_4, gamma_4, sigma_4,
                                  freq_5, gamma_5, sigma_5,
                                  freq_6, gamma_6, sigma_6,
                                  freq_7, gamma_7, sigma_7,
                                  freq_8, gamma_8, sigma_8,
                                  freq_9, gamma_9, sigma_9,
                                  freq_10, gamma_10, sigma_10,
                                  freq_11, gamma_11, sigma_11):
    
    fit = drude_susceptibility_fit(freq, freq_0, gamma_0, sigma_0)
    fit += lorentz_susceptibility_fit(freq, freq_1, gamma_1, sigma_1)
    fit += lorentz_susceptibility_fit(freq, freq_2, gamma_2, sigma_2)
    fit += lorentz_susceptibility_fit(freq, freq_3, gamma_3, sigma_3)
    fit += lorentz_susceptibility_fit(freq, freq_4, gamma_4, sigma_4)
    fit += lorentz_susceptibility_fit(freq, freq_5, gamma_5, sigma_5)
    fit += lorentz_susceptibility_fit(freq, freq_6, gamma_6, sigma_6)
    fit += lorentz_susceptibility_fit(freq, freq_7, gamma_7, sigma_7)
    fit += lorentz_susceptibility_fit(freq, freq_8, gamma_8, sigma_8)
    fit += lorentz_susceptibility_fit(freq, freq_9, gamma_9, sigma_9)
    fit += lorentz_susceptibility_fit(freq, freq_10, gamma_10, sigma_10)
    fit += lorentz_susceptibility_fit(freq, freq_11, gamma_11, sigma_11)
    
    return fit

def added_susceptibilities_fit_12(freq, 
                                  freq_0, gamma_0, sigma_0,
                                  freq_1, gamma_1, sigma_1,
                                  freq_2, gamma_2, sigma_2,
                                  freq_3, gamma_3, sigma_3,
                                  freq_4, gamma_4, sigma_4,
                                  freq_5, gamma_5, sigma_5,
                                  freq_6, gamma_6, sigma_6,
                                  freq_7, gamma_7, sigma_7,
                                  freq_8, gamma_8, sigma_8,
                                  freq_9, gamma_9, sigma_9,
                                  freq_10, gamma_10, sigma_10,
                                  freq_11, gamma_11, sigma_11,
                                  freq_12, gamma_12, sigma_12):
    
    fit = drude_susceptibility_fit(freq, freq_0, gamma_0, sigma_0)
    fit += lorentz_susceptibility_fit(freq, freq_1, gamma_1, sigma_1)
    fit += lorentz_susceptibility_fit(freq, freq_2, gamma_2, sigma_2)
    fit += lorentz_susceptibility_fit(freq, freq_3, gamma_3, sigma_3)
    fit += lorentz_susceptibility_fit(freq, freq_4, gamma_4, sigma_4)
    fit += lorentz_susceptibility_fit(freq, freq_5, gamma_5, sigma_5)
    fit += lorentz_susceptibility_fit(freq, freq_6, gamma_6, sigma_6)
    fit += lorentz_susceptibility_fit(freq, freq_7, gamma_7, sigma_7)
    fit += lorentz_susceptibility_fit(freq, freq_8, gamma_8, sigma_8)
    fit += lorentz_susceptibility_fit(freq, freq_9, gamma_9, sigma_9)
    fit += lorentz_susceptibility_fit(freq, freq_10, gamma_10, sigma_10)
    fit += lorentz_susceptibility_fit(freq, freq_11, gamma_11, sigma_11)
    fit += lorentz_susceptibility_fit(freq, freq_12, gamma_12, sigma_12)
    
    return fit

#%%

added_susceptibilities_fits = [added_susceptibilities_fit_0,
                               added_susceptibilities_fit_1,
                               added_susceptibilities_fit_2,
                               added_susceptibilities_fit_3,
                               added_susceptibilities_fit_4,
                               added_susceptibilities_fit_5,
                               added_susceptibilities_fit_6,
                               added_susceptibilities_fit_7,
                               added_susceptibilities_fit_8,
                               added_susceptibilities_fit_9,
                               added_susceptibilities_fit_10,
                               added_susceptibilities_fit_11,
                               added_susceptibilities_fit_12]

min_fit_grade = 0
max_fit_grade = 8

#%%

models = []
params = []
results = []
fitted_epsilon_jc = []

# weird_params = ["freq_0", "gamma_0", "freq_1", "gamma_1", "sigma_1", "freq_2", "gamma_2", "sigma_2"]

for n in range(min_fit_grade, max_fit_grade+1):
    
    models.append( lm.Model(added_susceptibilities_fits[n]) )
    
    params.append( models[-1].make_params(**params_r) )
    for key in params[-1].keys():
        params[-1][key].min = 0
        if params[-1][key].value < 0:
            params[-1][key].value = 1
        # params[-1][key].value = 1
        # if key in weird_params:
            # params[-1][key].value = 3
    
    results.append( models[-1].fit(epsilon_data_jc, params[-1], freq=freqs_jc) )
    
    fitted_epsilon_jc.append( results[-1].best_fit )

#%%

functions = [np.abs, np.real, np.imag]
titles = ["Absolute value", "Real part", "Imaginary part"]
ylabels = [r"|$\epsilon$|", r"Re($\epsilon$)", r"Im($\epsilon$)"]
long_wlen = [*[wlens_jc]*(len(results)+1)]
long_epsilon = [*fitted_epsilon_jc, epsilon_data_jc]
labels = [*[f"Drude-Lorentz fit {n}" for n in range(min_fit_grade, max_fit_grade+1)], "Johnson & Christy"]
linestyles = [*["--"]*(len(results)), "."]
colors = [*[plab.cm.hsv(n/len(results)) for n in range(len(results))], 'k']

nplots = len(functions)
fig = plt.figure(figsize=(nplots*6.4, 6.4), tight_layout=True)
axes = fig.subplots(ncols=nplots)

max_value = []
min_value = []
for ax, f, t, y in zip(axes, functions, titles, ylabels):
    for wl, eps, l, lst, col in zip(long_wlen, long_epsilon, labels, linestyles, colors):
        ax.set_title(t)
        ax.plot(wl, f(eps), lst, color=col, label=l)
        ax.xaxis.set_label_text(r"Wavelength [$\mu$m]")
        ax.yaxis.set_label_text(y)
        ax.legend()
        ax.set_xlim(*wlen_range_jc)
        max_value.append(max(f(eps)))
        min_value.append(min(f(eps)))
        
for ax in axes: ax.set_ylim([min(min_value)-.1*(max(max_value)-min(min_value)), 
                              max(max_value)+.1*(max(max_value)-min(min_value))])

vs.saveplot( plot_file(f"{material}DrudeLorentzJCFit.png"), overwrite=True )

#%%

for ax in axes: 
    ax.set_ylim([-50,50])
    if material=="Au":
        ax.set_xlim([.4, .8])
    elif material=="Ag":
        ax.set_xlim([.25, .8])
    else:
        raise ValueError("Material not recognized for zoom in plot")
    
vs.saveplot( plot_file(f"{material}DrudeLorentzJCFitZoom.png"), overwrite=True )

#%%

functions = [np.abs, np.real, np.imag]
titles = ["Absolute value", "Real part", "Imaginary part"]
ylabels = [r"|$\epsilon - \epsilon_f$|", r"Re($\epsilon - \epsilon_f$)", r"Im($\epsilon - \epsilon_f$)"]
long_wlen = [*[wlens_jc]*(len(results)+1)]
long_residuals = [epsilon_data_jc - fitted_epsilon for fitted_epsilon in fitted_epsilon_jc]
labels = [*[f"Drude-Lorentz fit {n}" for n in range(min_fit_grade, max_fit_grade+1)]]
linestyles = ["."]*(len(results))
colors = [plab.cm.hsv(n/len(results)) for n in range(len(results))]

nplots = len(functions)
fig = plt.figure(figsize=(nplots*6.4, 6.4), tight_layout=True)
axes = fig.subplots(ncols=nplots)

max_value = []
min_value = []
for ax, f, t, y in zip(axes, functions, titles, ylabels):
    for wl, res, l, lst, col in zip(long_wlen, long_residuals, labels, linestyles, colors):
        ax.set_title(t)
        ax.hlines(0, *wlen_range_jc, color="k", linewidth=.5)
        ax.plot(wl, f(res), lst, color=col, label=l)
        ax.xaxis.set_label_text(r"Wavelength [$\mu$m]")
        ax.yaxis.set_label_text(y)
        ax.legend(ncol=2)
        # ax.set_xlim(*[.4,.8])
        ax.set_xlim(*wlen_range_jc)
        max_value.append(max(f(res)))
        min_value.append(min(f(res)))
        
for ax in axes: ax.set_ylim([min(min_value)-.1*(max(max_value)-min(min_value)), 
                             max(max_value)+.1*(max(max_value)-min(min_value))])
# for ax in axes: ax.set_ylim([-50,50])

vs.saveplot( plot_file(f"{material}DrudeLorentzJCResiduals.png"), overwrite=True )

#%%

for ax in axes: 
    ax.set_ylim([-50,50])
    if material=="Au":
        ax.set_xlim([.4, .8])
    elif material=="Ag":
        ax.set_xlim([.25, .8])
    else:
        raise ValueError("Material not recognized for zoom in plot")
    
vs.saveplot( plot_file(f"{material}DrudeLorentzJCResidualsZoom.png"), overwrite=True )
