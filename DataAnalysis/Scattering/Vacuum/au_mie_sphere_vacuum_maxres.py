#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 12:24:04 2020

@author: vall
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plab
import os
import v_analysis as va
import v_save as vs
import v_utilities as vu

#%% PARAMETERS

# Saving directories
folder = ["Scattering/AuSphere/AllVacTest/10)MaxRes/Max103FUMixRes",
          "Scattering/AuSphere/AllVacTest/10)MaxRes/Max103FUMixRes"]
home = vs.get_home()

# Sorting and labelling data series
sorting_function = [lambda ls : vu.sort_by_number(ls, -1), 
                    lambda ls : vu.sort_by_number(ls, -1), ]
series_label = [lambda s : f"Meep Resolution {vu.find_numbers(s)[-1]}",
                lambda s : "Mie Theory"]
series_must = ["", ""] # leave "" per default
series_column = [1, 2]

# Scattering plot options
plot_title = "Scattering for 103 nm Au sphere in vacuum"
series_colors = [plab.cm.Reds, plab.cm.Reds]
series_linestyles = ["solid", "dashed"]
plot_make_big = True
plot_file = lambda n : os.path.join(home, "DataAnalysis", "VacuumMax103FUMixRes"+n)

#%% LOAD DATA

path = []
file = []
series = []
data = []
params = []
header = []

for f, sf, sm in zip(folder, sorting_function, series_must):

    path.append( os.path.join(home, f) )
    file.append( lambda f, s : os.path.join(path[-1], f, s) )
    
    series.append( os.listdir(path[-1]) )
    series[-1] = vu.filter_to_only_directories(series[-1])
    series[-1] = vu.filter_by_string_must(series[-1], sm)
    series[-1] = sf(series[-1])
    
    data.append( [] )
    params.append( [] )
    for s in series[-1]:
        data[-1].append(np.loadtxt(file[-1](s, "Results.txt")))
        params[-1].append(vs.retrieve_footer(file[-1](s, "Results.txt")))
    header.append( vs.retrieve_header(file[-1](s, "Results.txt")) )
    
    for i in range(len(params[-1])):
        if not isinstance(params[-1][i], dict): 
            params[-1][i] = vu.fix_params_dict(params[-1][i])
    
    # r = [p["r"] for p in params]
    # from_um_factor = [p["from_um_factor"] for p in params]

wlens = data[0][0][:,0]
resolution = [vu.find_numbers(s)[-1] for s in series[0]]
elapsed = [p["elapsed"] for p in params[0]]

def special_label(s):
    if str(resolution[-1]) in s:
        return "Mie Theory"
    else:
        return ""
series_label[-1] = special_label

#%% GET MAX WAVELENGTH

max_wlen = []
for d, sc in zip(data, series_column):
    max_wlen.append( [d[i][np.argmax(d[i][:,sc]), 0] for i in range(len(d))] )
dif_max_wlen = [max_wlen[0][i] - max_wlen[1][i] for i in range(len(data[0]))]

def exponential_fit(X, A, b, C):
    return A * np.exp(-b*X) + C
rsq, parameters = va.nonlinear_fit(np.array(resolution), 
                                   np.array(dif_max_wlen), 
                                   exponential_fit,
                                   par_units=["nm", "", "nm"])

plt.title("Difference in scattering maximum for Au 103 nm sphere in vacuum")
plt.legend(["Data", r"Fit $f(r)=a_0 e^{-a_1 r} + a_2$"])
plt.xlabel("Resolution")
plt.ylabel("Difference in wavelength $\lambda_{max}^{MEEP}-\lambda_{max}^{MIE}$")
vs.saveplot(plot_file("WLenDiff.png"), overwrite=True)

#%% GET ELAPSED TIME

elapsed_time = [params[0][i]["elapsed"] for i in range(len(data[0]))]
total_elapsed_time = [sum(et) for et in elapsed_time]

def quartic_fit(X, A, b):
    return A * (X)**4 + b
rsq, parameters = va.nonlinear_fit(np.array(resolution), 
                                   np.array(total_elapsed_time), 
                                   quartic_fit,
                                   par_units=["s","s"])

plt.title("elapsed time simulation of Au 103 nm sphere in vacuum")
# plt.plot(resolution, total_elapsed_time)
plt.legend(["Data", r"Fit $f(r)=a_0 r^4 + a_1$"], loc="lower right")
plt.xlabel("Resolution")
plt.ylabel("elapsed time [s]")
vs.saveplot(plot_file("TotTime.png"), overwrite=True)

plt.figure()
plt.title("elapsed time for simulations of Au 103 nm sphere in vacuum")
plt.plot(resolution, [et[1] for et in elapsed_time], 'D-b', label="Sim I")
plt.plot(resolution, [et[-1] for et in elapsed_time], 's-b', label="Sim II")
plt.xlabel("Resolution")
plt.ylabel("elapsed time in simulations [s]")
plt.legend()
plt.savefig(plot_file("SimTime.png"), bbox_inches='tight')

plt.figure()
plt.title("elapsed time for building of Au 103 nm sphere in vacuum")
plt.plot(resolution, [et[0] for et in elapsed_time], 'D-r', label="Sim I")
plt.plot(resolution, [et[2] for et in elapsed_time], 's-r', label="Sim II")
plt.xlabel("Resolution")
plt.ylabel("elapsed time in building [s]")
plt.legend()
plt.savefig(plot_file("BuildTime.png"), bbox_inches='tight')

plt.figure()
plt.title("elapsed time for loading flux of Au 103 nm sphere in vacuum")
plt.plot(resolution, [et[3] for et in elapsed_time], 's-m')
plt.xlabel("Resolution")
plt.ylabel("elapsed time in loading flux [s]")
plt.savefig(plot_file("LoadTime.png"), bbox_inches='tight')

#%% PLOT NORMALIZED

colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

plt.figure()
plt.title(plot_title)
for s, d, p, sc, psl, pc, pls in zip(series, data, params, series_column, 
                                     series_label, colors, series_linestyles):

    for ss, sd, sp, spc in zip(s, d, p, pc):
        if ss!=series[0][0]:
            plt.plot(sd[:,0], sd[:,sc] / max(sd[:,sc]), 
                     linestyle=pls, color=spc, label=psl(ss))

plt.xlabel(r"Wavelength $\lambda$ [nm]")
plt.ylabel("Normalized Scattering Cross Section")
plt.legend()
if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
del mng
vs.saveplot(plot_file("AllScatt.png"), overwrite=True)

#%% PLOT EFFIENCIENCY

colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

plt.figure()
plt.title(plot_title)
for s, d, p, sc, psl, pc, pls in zip(series, data, params, series_column, 
                                     series_label, colors, series_linestyles):

    for ss, sd, sp, spc in zip(s, d, p, pc):
        if ss!=series[0][0]:
            plt.plot(sd[:,0], sd[:,sc],# / max(sd[:,sc]), 
                     linestyle=pls, color=spc, label=psl(ss))
            
plt.xlabel(r"Wavelength $\lambda$ [nm]")
plt.ylabel("Scattering Effiency")
plt.legend()
if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
del mng
vs.saveplot(plot_file("AllScattEff.png"), overwrite=True)

#%% PLOT IN UNITS

colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

plt.figure()
plt.title(plot_title)
for s, d, p, sc, psl, pc, pls in zip(series, data, params, series_column, 
                                     series_label, colors, series_linestyles):

    for ss, sd, sp, spc in zip(s, d, p, pc):
        if ss!=series[0][0]:
            plt.plot(sd[:,0], sd[:,sc] * np.pi * (sp['r'] * sp['from_um_factor'] * 1e3)**2,
                     linestyle=pls, color=spc, label=psl(ss))
            
plt.xlabel(r"Wavelength $\lambda$ [nm]")
plt.ylabel(r"Scattering Cross Section [nm$^2$]")
plt.legend()
if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
del mng
vs.saveplot(plot_file("AllScatt.png"), overwrite=True)