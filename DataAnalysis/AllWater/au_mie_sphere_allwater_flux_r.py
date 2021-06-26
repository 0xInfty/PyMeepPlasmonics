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
folder = ["AuMieMediums/AllWaterTest/9)BoxDimensions/FluxR",
          "AuMieMediums/AllWaterTest/9)BoxDimensions/FluxR/MoreAir"]
home = vs.get_home()

# Sorting and labelling data series
sorting_function = [lambda l : vu.sort_by_number(l, -2)]*2
# def special_label(s):
#     if "5" in s:
#         return "Mie"
#     else:
#         return ""
series_label = [lambda s : f"Air 0.5 + Flux {vu.find_numbers(s)[-2]:.2f}",
                lambda s : f"Air 2.0 + Flux {vu.find_numbers(s)[-2]:.2f}"]
series_must = [""]*2 # leave "" per default
series_mustnt = ["MoreAir", ""] # leave "" per default
series_column = [1]*2

# Scattering plot options
plot_title = "Au 103 nm sphere in water"
series_colors = [plab.cm.Reds, plab.cm.Purples]
series_linestyles = ["solid"]*2
plot_make_big = False
plot_file = lambda n : os.path.join(home, "DataAnalysis/ComparedFluxR" + n)

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
flux_box_size = []
sysname = []
for p in params:
    r.append( [pi["r"] for pi in p] )
    from_um_factor.append( [pi["from_um_factor"] for pi in p] )
    resolution.append( [pi["resolution"] for pi in p] )
    index.append( [pi["submerged_index"] for pi in p] )
    flux_box_size.append([p["flux_box_size"] for p in p])
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
    
#%% WAVELENGTH MAXIMUM DIFFERENCE VS FLUX R FACTOR COMPARED

flux_r_factor = [[vu.find_numbers(s)[0] for s in ser] for ser in series]
air_width_factor = [[p["air_width"] for p in par] for par in params]
flux_box_padding = [[p["flux_box_size"]/2-p["r"] for p in par] for par in params]
flux_air_relation = [[f/a for f,a in zip(flux, air)] for flux, air in zip(flux_box_padding, air_width_factor)]
colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

plt.figure()
plt.title("Difference in scattering maximum for " + plot_title)
for f, d, c in zip(flux_air_relation, max_wlen_diff, colors):
    plt.plot(f, d, '.', markersize=12, color=c[3])
plt.grid(True)
plt.legend(["Air-r factor 0.5", "Air-r factor 2.0"])
plt.xlabel("Flux box padding expressed in multiples of empty space padding")
plt.ylabel("Difference in wavelength $\lambda_{max}^{MEEP}-\lambda_{max}^{MIE}$ [nm]")
vs.saveplot(plot_file("WLenDiff.png"), overwrite=True)

plt.xlim(-.05,1.05)
plt.ylim(-2, 14)
vs.saveplot(plot_file("WLenDiffZoom.png"), overwrite=True)

#%% WAVELENGTH MAXIMUM DIFFERENCE VS FLUX R FACTOR

flux_r_factor = [[vu.find_numbers(s)[0] for s in ser] for ser in series]
colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

plt.title("Difference in scattering maximum for " + plot_title)
plt.plot(flux_r_factor[0], max_wlen_diff[0], '.', markersize=12, color=c[0])
plt.grid(True)
plt.xlabel("Flux box side expressed in multiples of radius")
plt.ylabel("Difference in wavelength $\lambda_{max}^{MEEP}-\lambda_{max}^{MIE}$ [nm]")
vs.saveplot(plot_file("WLenDiff.png"), overwrite=True)

plt.xlim(-.05,.55)
plt.ylim(-2, 14)
vs.saveplot(plot_file("WLenDiffZoom.png"), overwrite=True)

#%% WAVELENGTH MAXIMUM DIFFERENCE VS FLUX BOX SIZE AND PADDING

flux_box_size_nm = np.array([p["flux_box_size"]*p["from_um_factor"]*1e3 for p in params[0]])
r_nm = np.array([p["r"]*p["from_um_factor"]*1e3 for p in params[0]])
flux_box_padding_nm = flux_box_size_nm - 2 * r_nm

fig = plt.figure()
plt.title("Difference in scattering maximum for " + plot_title)
plt.plot(flux_box_size_nm[0], max_wlen_diff[0], '.', markersize=12)
plt.grid(True)
plt.xlabel("Flux box side length [nm]")
ax2 = plt.twiny()
ax2.plot(flux_box_padding_nm[0], max_wlen_diff[0], '.', markersize=12)
plt.xlabel("Flux box padding [nm]")
plt.ylabel("Difference in wavelength $\lambda_{max}^{MEEP}-\lambda_{max}^{MIE}$ [nm]")
vs.saveplot(plot_file("WLenDiff2.png"), overwrite=True)

fig.axes[0].set_xlim(103-5, 103+55)
fig.axes[1].set_xlim(-5, 55)
plt.ylim(-2, 14)
vs.saveplot(plot_file("WLenDiff2Zoom.png"), overwrite=True)

#%% GET ENLAPSED TIME COMPARED

flux_r_factor = [[vu.find_numbers(s)[0] for s in ser] for ser in series]
enlapsed_time = [[p["enlapsed"] for p in par] for par in params]
total_enlapsed_time = [[sum(p["enlapsed"]) for p in par] for par in params]

first_flux_r_factor = []
second_flux_r_factor = []
first_build_time = []
first_sim_time = []
second_flux_time = []
second_build_time = []
second_sim_time = []
for enl, frf in zip(enlapsed_time, flux_r_factor):
    first_flux_r_factor.append( [] )
    first_build_time.append( [] )
    first_sim_time.append( [] )
    second_flux_r_factor.append( [] )
    second_flux_time.append( [] )
    second_build_time.append( [] )
    second_sim_time.append( [] )
    for e, fr in zip(enl, frf):
        if len(e)==5:
            first_flux_r_factor[-1].append( fr )
            second_flux_r_factor[-1].append( fr )
            first_build_time[-1].append( e[0] )
            first_sim_time[-1].append( e[1] )
            second_build_time[-1].append( e[2] )
            second_flux_time[-1].append( e[3] )
            second_sim_time[-1].append( e[4] )
        elif len(e)==3:
            second_flux_r_factor[-1].append( fr )
            second_build_time[-1].append( e[0] )
            second_flux_time[-1].append( e[1] )
            second_sim_time[-1].append( e[2] )
        else:
            print(f"Unknown error in flux_r_factor {fr} of", frf)

plt.figure()
plt.title("Enlapsed total time for simulation of " + plot_title)
for frf, tot, col in zip(flux_r_factor, total_enlapsed_time, colors):
    plt.plot(frf, tot, '.', markersize=12, color=col[3])
plt.legend(["Air-r factor 0.5", "Air-r factor 2.0"])
plt.xlabel("Flux box side in multiples of radius")
plt.ylabel("Enlapsed time [s]")
vs.saveplot(plot_file("ComparedTotTime.png"), overwrite=True)
        
plt.figure()
plt.title("Enlapsed time for simulations of " + plot_title)
plt.plot(first_flux_r_factor[0], first_sim_time[0], 'D-', color="r", label="Air 0.5 Sim I")
plt.plot(first_flux_r_factor[1], first_sim_time[1], 'D-', color="maroon", label="Air 2.0 Sim I")
plt.plot(second_flux_r_factor[0], second_sim_time[0], 's-', color="r", label="Air 0.5 Sim II")
plt.plot(second_flux_r_factor[1], second_sim_time[1], 's-', color="maroon", label="Air 2.0 Sim II")
plt.xlabel("Flux box side in multiples of radius")
plt.ylabel("Enlapsed time in simulations [s]")
plt.legend()
plt.savefig(plot_file("ComparedSimTime.png"), bbox_inches='tight')

plt.figure()
plt.title("Enlapsed time for building of " + plot_title)
plt.plot(first_flux_r_factor[0], first_build_time[0], 'D-', color="b", label="Air 0.5 Sim I")
plt.plot(first_flux_r_factor[1], first_build_time[1], 'D-', color="navy", label="Air 2.0 Sim I")
plt.plot(second_flux_r_factor[0], second_build_time[0], 's-', color="b", label="Air 0.5 Sim II")
plt.plot(second_flux_r_factor[1], second_build_time[1], 's-', color="navy", label="Air 2.0 Sim II")
plt.xlabel("Flux box side in multiples of radius")
plt.ylabel("Enlapsed time in building [s]")
plt.legend()
plt.savefig(plot_file("ComparedBuildTime.png"), bbox_inches='tight')

plt.figure()
plt.title("Enlapsed time for loading flux of " + plot_title)
plt.plot(second_flux_r_factor[0], second_flux_time[0], 's-', color="m", label="Air 0.5 Sim II")
plt.plot(second_flux_r_factor[1], second_flux_time[1], 's-', color="darkmagenta", label="Air 2.0 Sim II")
plt.xlabel("Flux box side in multiples of radius")
plt.ylabel("Enlapsed time in loading flux [s]")
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

plt.plot(data[0][-1][:,0], theory[0][-1] / max(theory[0][-1]), 
         linestyle="dashed", color='red', label="Mie Theory")
plt.xlabel("Wavelength [nm]")
plt.ylabel("Normalized Scattering Cross Section")
plt.legend(ncol=2)
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
        if ss in series[0]:
            plt.plot(sd[:,0], sign * sd[:,sc] * np.pi * (sp["r"] * sp["from_um_factor"] * 1e3)**2 * ( sp["flux_box_size"] / (2*sp["r"]) )**2,
                     linestyle=pls, color=spc, label=psl(ss))
        else:
            plt.plot(sd[:,0], sign * sd[:,sc] * np.pi * (sp["r"] * sp["from_um_factor"] * 1e3)**2,
                     linestyle=pls, color=spc, label=psl(ss))
            
plt.plot(data[0][-1][:,0], theory[0][-1] * np.pi * (sp["r"] * sp["from_um_factor"] * 1e3)**2,
         linestyle="dashed", color='red', label="Mie Theory")
plt.xlabel("Wavelength [nm]")
plt.ylabel(r"Scattering Cross Section [nm$^2$]")
plt.legend(ncol=3)
if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    del mng
vs.saveplot(plot_file("AllScatt.png"), overwrite=True)

#%% PLOT NORMALIZED JUST INSIDE INNER BOX

colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

plt.figure()
plt.title("Normalized scattering for " + plot_title)
for s, d, p, sc, psl, pc, pls in zip(series, data, params, series_column, 
                                     series_label, colors, series_linestyles):

    for ss, sd, sp, spc in zip(s, d, p, pc):
        if ss not in series[0][-3:]:
            index_argmax = np.argmax(abs(sd[:,sc]))
            plt.plot(sd[:,0], sd[:,sc] / sd[index_argmax, sc], 
                     linestyle=pls, color=spc, label=psl(ss))

plt.plot(data[0][-1][:,0], theory[0][-1] / max(theory[0][-1]), 
         linestyle="dashed", color='red', label="Mie Theory")
plt.xlabel("Wavelength [nm]")
plt.ylabel("Normalized Scattering Cross Section")
plt.legend(ncol=2)
if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    del mng
vs.saveplot(plot_file("AllScattNormZoom.png"), overwrite=True)

#%% PLOT IN UNITS

colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

plt.figure()
plt.title("Scattering for " + plot_title)
for s, d, p, sc, psl, pc, pls in zip(series, data, params, series_column, 
                                     series_label, colors, series_linestyles):

    for ss, sd, sp, spc in zip(s, d, p, pc):
        if ss not in series[0][-3:]:
            sign = np.sign( sd[ np.argmax(abs(sd[:,sc])), sc ] )
            if ss in series[0]:
                plt.plot(sd[:,0], sign * sd[:,sc] * np.pi * (sp["r"] * sp["from_um_factor"] * 1e3)**2 * ( sp["flux_box_size"] / (2*sp["r"]) )**2,
                         linestyle=pls, color=spc, label=psl(ss))
            else:
                plt.plot(sd[:,0], sign * sd[:,sc] * np.pi * (sp["r"] * sp["from_um_factor"] * 1e3)**2,
                         linestyle=pls, color=spc, label=psl(ss))
            
plt.plot(data[0][-1][:,0], theory[0][-1] * np.pi * (sp["r"] * sp["from_um_factor"] * 1e3)**2,
         linestyle="dashed", color='red', label="Mie Theory")
plt.xlabel("Wavelength [nm]")
plt.ylabel(r"Scattering Cross Section [nm$^2$]")
plt.legend(ncol=3)
if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    del mng
vs.saveplot(plot_file("AllScattZoom.png"), overwrite=True)
