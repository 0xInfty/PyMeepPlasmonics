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
folder = ["AuMieMediums/AllWaterTest/9)BoxDimensions/LongRes/500to700"]
home = vs.get_home()

# Parameter for the test
test_param_string = "resolution"
test_param_in_params = True
test_param_position = -1
test_param_label = "Resolution"

# Sorting and labelling data series
sorting_function = [lambda l : vu.sort_by_number(l, -1)]
series_label = [lambda s : f"Resolution {vu.find_numbers(s)[test_param_position]:.0f}"]
series_must = [""] # leave "" per default
series_mustnt = ["Failed"] # leave "" per default
series_column = [1]

# Scattering plot options
plot_title = "Au 103 nm sphere in water"
series_legend = ["Data"]
series_colors = [plab.cm.Reds]
series_linestyles = ["solid"]
plot_make_big = False
plot_file = lambda n : os.path.join(home, "DataAnalysis/LongRes" + n)

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

if test_param_in_params:
    test_param = [[p[test_param_string] for p in par] for par in params]
else:
    test_param = [[vu.find_numbers(s)[test_param_position] for s in ser] for ser in series]

minor_division = [[fum * 1e3 / res for fum, res in zip(frum, reso)] for frum, reso in zip(from_um_factor, resolution)]
width_points = [[int(p["cell_width"] * p["resolution"]) for p in par] for par in params] 
grid_points = [[wp**3 for wp in wpoints] for wpoints in width_points]
memory_B = [[2 * 12 * gp * 32 for p, gp in zip(par, gpoints)] for par, gpoints in zip(params, grid_points)] # in bytes

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

mean_residual = [[np.mean(np.square(d[:,1] - t)) for d,t in zip(dat, theo)] for dat, theo in zip(data, theory)]
mean_residual_left = [[np.mean(np.square(d[:np.argmax(t),1] - t[:np.argmax(t)])) for d,t in zip(dat, theo)] for dat, theo in zip(data, theory)]
mean_residual_right = [[np.mean(np.square(d[np.argmax(t):,1] - t[np.argmax(t):])) for d,t in zip(dat, theo)] for dat, theo in zip(data, theory)]

#%% WAVELENGTH MAXIMUM DIFFERENCE

if len(series)>1:
    colors = ["darkgrey", "k"]
else:
    colors = ["k"]

fig, [ax1, ax2] = plt.subplots(nrows=2, sharex=True, gridspec_kw={"hspace":0})

plt.suptitle("Difference in scattering for " + plot_title)

ax1.set_ylabel("$\lambda_{max}^{MEEP}-\lambda_{max}^{MIE}$ [nm]")
for tp, mwl, col, leg in zip(test_param, max_wlen_diff, colors, series_legend):
    ax1.plot(tp, mwl, '.', color=col, markersize=12)
ax1.grid(True)
ax1.legend(series_legend)

ax2.set_ylabel("MSD( $\sigma^{MEEP} - \sigma^{MIE}$ )")
for tp, mr, col, leg in zip(test_param, mean_residual, colors, series_legend):
    ax2.plot(tp, mr, '.', color=col, markersize=12)
ax2.grid(True)
ax2.legend(series_legend)

ax2.set_xlabel(test_param_label)
# plt.figlegend(bbox_to_anchor=(.95, 0.7), bbox_transform=ax2.transAxes)
# ax2.legend()
plt.tight_layout()
plt.show()

vs.saveplot(plot_file("TheoryDiff.png"), overwrite=True)

#%% DIFFERENCE IN SCATTERING MAXIMUM

if len(series)>1:
    colors = ["darkgrey", "k"]
else:
    colors = ["k"]

plt.figure()
plt.title("Difference in scattering maximum for " + plot_title)
for tp, mwl, col in zip(test_param, max_wlen_diff, colors):
    plt.plot(tp, mwl, '.', color=col, markersize=12)
plt.grid(True)
plt.legend(series_legend)
plt.xlabel(test_param_label)
plt.ylabel("Difference in wavelength $\lambda_{max}^{MEEP}-\lambda_{max}^{MIE}$ [nm]")
vs.saveplot(plot_file("WLenDiff.png"), overwrite=True)

#%% MEAN RESIDUAL

if len(series)>1:
    colors = ["darkgrey", "k"]
    colors_right = ["red", "maroon"]
    colors_left = ["darkviolet", "rebeccapurple"]
else:
    colors = ["k"]
    colors_right = ["red"]
    colors_left = ["darkviolet"]

plt.figure()
plt.title("Mean quadratic difference in effiency for " + plot_title)
for i in range(len(test_param)):
    plt.plot(test_param[i], mean_residual_left[i], '.', color=colors_left[i], markersize=12, label=series_legend[i]+" left")
    plt.plot(test_param[i], mean_residual_right[i], '.', color=colors_right[i], markersize=12, label=series_legend[i]+" right")
    plt.plot(test_param[i], mean_residual[i], '.', color=colors[i], markersize=12, label=series_legend[i]+" all")
plt.grid(True)
plt.legend()
plt.xlabel(test_param_label)
plt.ylabel("Mean squared difference MSD( $C^{MEEP} - C^{MIE}$ )")
vs.saveplot(plot_file("QuaDiff.png"), overwrite=True)

#%% DIFFERENCE IN SCATTERING MAXIMUM VS MINOR DIVISION

if len(series)>1:
    colors = ["darkgrey", "k"]
else:
    colors = ["k"]

plt.figure()
plt.title("Difference in scattering maximum for " + plot_title)
for md, mwl, col in zip(minor_division, max_wlen_diff, colors):
    plt.plot(md, mwl, '.', color=col, markersize=12)
plt.grid(True)
plt.legend(series_legend)
plt.xlabel("Minor spatial division [nm]")
plt.ylabel("Difference in wavelength $\lambda_{max}^{MEEP}-\lambda_{max}^{MIE}$ [nm]")
vs.saveplot(plot_file("WLenDiffDiv.png"), overwrite=True)

#%% MEAN RESIDUAL VS MINOR DIVISION

if len(series)>1:
    colors = ["darkgrey", "k"]
    colors_right = ["red", "maroon"]
    colors_left = ["darkviolet", "rebeccapurple"]
else:
    colors = ["k"]
    colors_right = ["red"]
    colors_left = ["darkviolet"]

plt.figure()
plt.title("Mean quadratic difference in effiency for " + plot_title)
for md, mr, col in zip(minor_division, mean_residual, colors):
    plt.plot(minor_division[i], mean_residual_left[i], '.', color=colors_left[i], markersize=12, label=series_legend[i]+" left")
    plt.plot(minor_division[i], mean_residual_right[i], '.', color=colors_right[i], markersize=12, label=series_legend[i]+" right")
    plt.plot(minor_division[i], mean_residual[i], '.', color=colors[i], markersize=12, label=series_legend[i]+" all")
plt.grid(True)
plt.legend()
plt.xlabel("Minor spatial division [nm]")
plt.ylabel("Mean squared difference MSD( $C^{MEEP} - C^{MIE}$ )")
vs.saveplot(plot_file("QuaDiffDiv.png"), overwrite=True)

#%% GET elapsed TIME COMPARED

elapsed_time = [[p["elapsed"] for p in par] for par in params]
total_elapsed_time = [[sum(p["elapsed"]) for p in par] for par in params]

first_test_param = []
second_test_param = []
first_build_time = []
first_sim_time = []
second_flux_time = []
second_build_time = []
second_sim_time = []
for enl, tpar in zip(elapsed_time, test_param):
    first_test_param.append( [] )
    first_build_time.append( [] )
    first_sim_time.append( [] )
    second_test_param.append( [] )
    second_flux_time.append( [] )
    second_build_time.append( [] )
    second_sim_time.append( [] )
    for e, tp in zip(enl, tpar):
        if len(e)==5:
            first_test_param[-1].append( tp )
            second_test_param[-1].append( tp )
            first_build_time[-1].append( e[0] )
            first_sim_time[-1].append( e[1] )
            second_build_time[-1].append( e[2] )
            second_flux_time[-1].append( e[3] )
            second_sim_time[-1].append( e[4] )
        elif len(e)==3:
            second_test_param[-1].append( tp )
            second_build_time[-1].append( e[0] )
            second_flux_time[-1].append( e[1] )
            second_sim_time[-1].append( e[2] )
        else:
            print(f"Unknown error in '{test_param_string}' {tp} of", tpar)

if len(series)>1:
    colors = ["darkgrey", "k"]
else:
    colors = ["k"]
plt.figure()
plt.title("elapsed total time for simulation of " + plot_title)
for tp, tot, col in zip(test_param, total_elapsed_time, colors):
    plt.plot(tp, tot, '.-', color=col, markersize=14)
plt.legend(series_legend)
plt.xlabel(test_param_label)
plt.ylabel("elapsed time [s]")
vs.saveplot(plot_file("ComparedTotTime.png"), overwrite=True)
        
colors = ["r", "maroon"]
plt.figure()
plt.title("elapsed time for simulations of " + plot_title)
for tp, tim, col, leg in zip(first_test_param, first_sim_time, colors, series_legend):
    plt.plot(tp, tim, 'D-', color=col, label=leg + " Sim I")
for tp, tim, col, leg in zip(second_test_param, second_sim_time, colors, series_legend):
    plt.plot(tp, tim, 's-', color=col, label=leg + " Sim II")
plt.xlabel(test_param_label)
plt.ylabel("elapsed time in simulations [s]")
plt.legend()
plt.savefig(plot_file("ComparedSimTime.png"), bbox_inches='tight')

colors = ["b", "navy"]
plt.figure()
plt.title("elapsed time for building of " + plot_title)
for tp, tim, col, leg in zip(first_test_param, first_build_time, colors, series_legend):
    plt.plot(tp, tim, 'D-', color=col, label=leg + " Sim I")
for tp, tim, col, leg in zip(second_test_param, second_build_time, colors, series_legend):
    plt.plot(tp, tim, 's-', color=col, label=leg + " Sim II")
plt.xlabel(test_param_label)
plt.ylabel("elapsed time in building [s]")
plt.legend()
plt.savefig(plot_file("ComparedBuildTime.png"), bbox_inches='tight')

colors = ["m", "darkmagenta"]
plt.figure()
plt.title("elapsed time for loading flux of " + plot_title)
for tp, tim, col, leg in zip(second_test_param, second_flux_time, colors, series_legend):
    plt.plot(tp, tim, 's-', color=col, label=leg + " Sim II")
plt.xlabel(test_param_label)
plt.ylabel("elapsed time in loading flux [s]")
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
        plt.plot(sd[:,0], sign * sd[:,sc] * np.pi * (sp["r"] * sp["from_um_factor"] * 1e3)**2,
                 linestyle=pls, color=spc, label=psl(ss))
            
plt.plot(data[0][-1][:,0], theory[0][-1] * np.pi * (params[0][-1]["r"] * params[0][-1]["from_um_factor"] * 1e3)**2,
         linestyle="dashed", color='red', label="Mie Theory")
plt.xlabel("Wavelength [nm]")
plt.ylabel(r"Scattering Cross Section [nm$^2$]")
plt.legend(ncol=2)
if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    del mng
vs.saveplot(plot_file("AllScatt.png"), overwrite=True)

#%% PLOT DIFFERENCE IN UNITS

colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

plt.figure()
plt.title("Scattering for " + plot_title)
for s, d, t, p, sc, psl, pc, pls in zip(series, data, theory, params, series_column, 
                                        series_label, colors, series_linestyles):

    for ss, sd, st, sp, spc in zip(s, d, t, p, pc):
        sign = np.sign( sd[ np.argmax(abs(sd[:,sc])), sc ] )
        plt.plot(sd[:,0],  (sd[:,1] - st) * np.pi * (sp["r"] * sp["from_um_factor"] * 1e3)**2,
                 linestyle=pls, color=spc, label=psl(ss))
            
plt.xlabel("Wavelength [nm]")
plt.ylabel(r"Difference in Scattering Cross Section [nm$^2$]")
plt.legend(ncol=2)
if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    del mng
vs.saveplot(plot_file("AllScattDiff.png"), overwrite=True)

#%% PLOT DERIVATIVE IN UNITS

colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

fig = plt.figure()
plt.title("Scattering for " + plot_title)
for s, d, t, p, sc, psl, pc, pls in zip(series, data, theory, params, series_column, 
                                        series_label, colors, series_linestyles):

    for ss, sd, st, sp, spc in zip(s, d, t, p, pc):
        sign = np.sign( sd[ np.argmax(abs(sd[:,sc])), sc ] )
        plt.plot(sd[:-1,0],  np.diff(sd[:,1]) * np.pi * (sp["r"] * sp["from_um_factor"] * 1e3)**2 / np.diff(sd[:,0]),
                 linestyle=pls, color=spc, label=psl(ss))
xlims = fig.axes[0].get_xlim()
plt.hlines(0, *xlims, color="k")
fig.axes[0].set_xlim(*xlims)

        
plt.xlabel("Wavelength [nm]")
plt.ylabel(r"Derivative of Scattering Cross Section [nm]")
plt.legend()
if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    del mng
vs.saveplot(plot_file("AllScattDer.png"), overwrite=True)

#%% PLOT SECOND DERIVATIVE IN UNITS

colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

fig = plt.figure()
plt.title("Scattering for " + plot_title)
for s, d, t, p, sc, psl, pc, pls in zip(series, data, theory, params, series_column, 
                                        series_label, colors, series_linestyles):

    for ss, sd, st, sp, spc in zip(s, d, t, p, pc):
        sign = np.sign( sd[ np.argmax(abs(sd[:,sc])), sc ] )
        plt.plot(sd[:-2,0],  np.diff(np.diff(sd[:,1]) / np.diff(sd[:,0])) * np.pi * (sp["r"] * sp["from_um_factor"] * 1e3)**2 / np.diff(sd[:-1,0]),
                 linestyle=pls, color=spc, label=psl(ss))
xlims = fig.axes[0].get_xlim()
plt.hlines(0, *xlims, color="k")
fig.axes[0].set_xlim(*xlims)

plt.xlabel("Wavelength [nm]")
plt.ylabel(r"Second derivative of Scattering Cross Section [nm]")
plt.legend()
if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    del mng
vs.saveplot(plot_file("AllScattDer2.png"), overwrite=True)