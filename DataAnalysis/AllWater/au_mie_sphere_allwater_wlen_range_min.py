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
import PyMieScatt as ps
import v_analysis as va
from v_materials import import_medium
import v_save as vs
import v_utilities as vu

#%% PARAMETERS

# Saving directories
folder = ["AuMieMediums/AllWaterTest/9)BoxDimensions/WLenMin",
          "AuMieMediums/AllWaterTest/9)BoxDimensions/WLenMin/DoubleFreq"]
home = vs.get_home()

# Sorting and labelling data series
sorting_function = [lambda l : vu.sort_by_number(l, -2)]*2
# def special_label(s):
#     if "5" in s:
#         return "Mie"
#     else:
#         return ""
series_label = [lambda s : f"Min $\lambda$ {vu.find_numbers(s)[-2]:.0f} nm NFreq 100",
                lambda s : f"Min $\lambda$ {vu.find_numbers(s)[-2]:.0f} nm NFreq 200"]
series_must = [""]*2 # leave "" per default
series_mustnt = ["DoubleFreq"]*2 # leave "" per default
series_column = [1]*2

# Scattering plot options
plot_title = "Au 103 nm sphere in water"
series_colors = [plab.cm.Reds, plab.cm.Blues]
series_linestyles = ["solid"]*2
plot_make_big = False
plot_file = lambda n : os.path.join(home, "DataAnalysis/WLenMin" + n)

#%% LOAD DATA

path = []
file = []
series = []
data = []
params = []
header = []

for f, sf, sm, smn in zip(folder, sorting_function, series_must, series_mustnt):

    path.append( os.path.join(home, f) )
    file.append( lambda f, s : os.path.join(path[-1], f, s) )
    
    series.append( os.listdir(path[-1]) )
    series[-1] = vu.filter_by_string_must(series[-1], sm)
    if smn!="": series[-1] = vu.filter_by_string_must(series[-1], smn, False)
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
    
r = []
from_um_factor = []
resolution = []
paper = []
index = []
sysname = []
for p in params:
    r.append( [pi["r"] for pi in p] )
    from_um_factor.append( [pi["from_um_factor"] for pi in p] )
    resolution.append( [pi["resolution"] for pi in p] )
    index.append( [pi["submerged_index"] for pi in p] )
    sysname.append( [pi["sysname"] for pi in p] )
    try:
        paper.append( [pi["paper"] for pi in p])
    except:
        paper.append( ["R" for pi in p] )

#%% LOAD MIE DATA

theory = [] # Scattering effiency
for di, ri, fi, resi, ppi, ii in zip(data, r, from_um_factor, resolution, paper, index):
    theory.append([])    
    for dj, rj, fj, resj, ppij, ij in zip(di, ri, fi, resi, ppi, ii):
        wlenj = dj[:,0] # nm
        freqj = 1 / wlenj # 1/nm
        freqmeepj = (1e3 * fj) / wlenj # Meep units
        mediumj = import_medium("Au", from_um_factor=fj, paper=ppij)
        theory[-1].append(np.array(
            [ps.MieQ(np.sqrt(mediumj.epsilon(fqm)[0,0]*mediumj.mu(fqm)[0,0]), 
                     wl, # Wavelength (nm)
                     2*rj*1e3*fj, # Diameter (nm)
                     nMedium=ij, # Refraction Index of Medium
                     asDict=True)['Qsca'] 
             for wl, fq, fqm in zip(wlenj, freqj, freqmeepj)]))

#%% GET MAX WAVELENGTH

max_wlen = []
for d, sc in zip(data, series_column):
    max_wlen.append( [d[i][np.argmax(d[i][:,sc]), 0] for i in range(len(d))] )

max_wlen_theory = []
for t, d in zip(theory, data):
    max_wlen_theory.append( [d[i][np.argmax(t[i]), 0] for i in range(len(t))] )

max_wlen_diff = []
for md, mt in zip(max_wlen, max_wlen_theory):
    max_wlen_diff.append( [d - t for d,t in zip(md, mt)] )

#%% WAVELENGTH MAXIMUM DIFFERENCE VS WAVELENGTH MINIMUM

wlen_range_min = [[vu.find_numbers(s)[0] for s in ser] for ser in series]

plt.figure()
plt.title("Difference in scattering maximum for " + plot_title)
plt.plot(wlen_range_min[0], max_wlen_diff[0], '.', markersize=12)
plt.plot(wlen_range_min[1], max_wlen_diff[1], '.k', markersize=12)
plt.grid(True)
plt.legend(["NFreq100", r"NFreq200"])
plt.xlabel("Wavelength range minimum [nm]")
plt.ylabel("Difference in wavelength $\lambda_{max}^{MEEP}-\lambda_{max}^{MIE}$ [nm]")
vs.saveplot(plot_file("WLenDiff.png"), overwrite=True)

#%% GET ENLAPSED TIME COMPARED

wlen_range_min = [[vu.find_numbers(s)[0] for s in ser] for ser in series]
enlapsed_time = [[p["enlapsed"] for p in par] for par in params]
total_enlapsed_time = [[sum(p["enlapsed"]) for p in par] for par in params]

first_wlen_range_min = []
second_wlen_range_min = []
first_build_time = []
first_sim_time = []
second_flux_time = []
second_build_time = []
second_sim_time = []
for enl, wlm in zip(enlapsed_time, wlen_range_min):
    first_wlen_range_min.append( [] )
    first_build_time.append( [] )
    first_sim_time.append( [] )
    second_wlen_range_min.append( [] )
    second_flux_time.append( [] )
    second_build_time.append( [] )
    second_sim_time.append( [] )
    for e, wl in zip(enl, wlm):
        if len(e)==5:
            first_wlen_range_min[-1].append( wl )
            second_wlen_range_min[-1].append( wl )
            first_build_time[-1].append( e[0] )
            first_sim_time[-1].append( e[1] )
            second_build_time[-1].append( e[2] )
            second_flux_time[-1].append( e[3] )
            second_sim_time[-1].append( e[4] )
        elif len(e)==3:
            second_wlen_range_min[-1].append( wl )
            second_build_time[-1].append( e[0] )
            second_flux_time[-1].append( e[1] )
            second_sim_time[-1].append( e[2] )
        else:
            print(f"Unknown error in wlen_range_min {wl} of", wlm)

plt.figure()
plt.title("Enlapsed total time for simulation of " + plot_title)
for wlm, tot, col in zip(wlen_range_min, total_enlapsed_time, ["darkgray", "black"]):
    plt.plot(wlm, tot, '.', color=col, markersize=12)
plt.legend(["NFreq100", r"NFreq200"])
plt.xlabel("Wavelength range minimum [nm]")
plt.ylabel("Enlapsed time [s]")
vs.saveplot(plot_file("ComparedTotTime.png"), overwrite=True)
        
plt.figure()
plt.title("Enlapsed time for simulations of " + plot_title)
plt.plot(first_wlen_range_min[0], first_sim_time[0], 'D-', color="r", label="NFreq 100 Sim I")
plt.plot(first_wlen_range_min[1], first_sim_time[1], 'D-', color="maroon", label="NFreq 200 Sim I")
plt.plot(second_wlen_range_min[0], second_sim_time[0], 's-', color="r", label="NFreq 100 Sim II")
plt.plot(second_wlen_range_min[1], second_sim_time[1], 's-', color="maroon", label="NFreq 200 Sim II")
plt.xlabel("Wavelength range minimum [nm]")
plt.ylabel("Enlapsed time in simulations [s]")
plt.legend()
plt.savefig(plot_file("ComparedSimTime.png"), bbox_inches='tight')

plt.figure()
plt.title("Enlapsed time for building of " + plot_title)
plt.plot(first_wlen_range_min[0], first_build_time[0], 'D-', color="b", label="NFreq 100 Sim I")
plt.plot(first_wlen_range_min[1], first_build_time[1], 'D-', color="navy", label="NFreq 200 Sim I")
plt.plot(second_wlen_range_min[0], second_build_time[0], 's-', color="b", label="NFreq 100 Sim II")
plt.plot(second_wlen_range_min[1], second_build_time[1], 's-', color="navy", label="NFreq 200 Sim II")
plt.xlabel("Wavelength range minimum [nm]")
plt.ylabel("Enlapsed time in building [s]")
plt.legend()
plt.savefig(plot_file("ComparedBuildTime.png"), bbox_inches='tight')

plt.figure()
plt.title("Enlapsed time for loading flux of " + plot_title)
plt.plot(second_wlen_range_min[0], second_flux_time[0], 's-', color="m", label="Sim II")
plt.plot(second_wlen_range_min[1], second_flux_time[1], 's-', color="darkmagenta", label="Sim II")
plt.xlabel("Wavelength range minimum [nm]")
plt.ylabel("Enlapsed time in loading flux [s]")
plt.legend()
plt.savefig(plot_file("ComparedLoadTime.png"), bbox_inches='tight')

#%% PLOT NORMALIZED

colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

plt.figure()
plt.title("Normalized scattering for " + plot_title)
for s, d, p, sc, psl, pc, pls in zip(series, data, params, series_column, 
                                     series_label, colors, series_linestyles):

    for ss, sd, sp, spc in zip(s, d, p, pc):
        index_argmax = np.argmax(abs(sd[:,sc]))
        plt.plot(sd[:,0], sd[:,sc] / sd[index_argmax, sc], 
                 linestyle=pls, color=spc, label=psl(ss))

plt.plot(data[0][0][:,0], theory[0][0] / max(theory[0][-1]), 
         linestyle="dashed", color='red', label="Mie Theory")
plt.xlabel("Wavelength [nm]")
plt.ylabel("Normalized Scattering Cross Section")
plt.legend()
if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    del mng
vs.saveplot(plot_file("AllScattNorm.png"), overwrite=True)

#%% PLOT IN UNITS

colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

plt.figure()
plt.title("Scattering for " + plot_title)
for s, d, p, sc, psl, pc, pls in zip(series, data, params, series_column, 
                                     series_label, colors, series_linestyles):

    for ss, sd, sp, spc in zip(s, d, p, pc):
        sign = np.sign( sd[ np.argmax(abs(sd[:,sc])), sc ] )
        plt.plot(sd[:,0], sign * sd[:,sc] * np.pi * (sp["r"] * sp["from_um_factor"] * 1e3)**2,
                 linestyle=pls, color=spc, label=psl(ss))
            
plt.plot(data[0][0][:,0], theory[0][0] * np.pi * (params[0][-1]["r"] * params[0][-1]["from_um_factor"] * 1e3)**2,
         linestyle="dashed", color='red', label="Mie Theory")
plt.xlabel("Wavelength [nm]")
plt.ylabel(r"Scattering Cross Section [nm$^2$]")
plt.legend()
if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    del mng
vs.saveplot(plot_file("AllScatt.png"), overwrite=True)
