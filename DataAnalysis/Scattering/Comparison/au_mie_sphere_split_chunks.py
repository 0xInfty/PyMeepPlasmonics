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
# folder = [*["Test/TestChunks/AllVac450to600"]*2, 
#           *["Test/TestChunks/AllWat500to650/Res2"]*2]
folder = [*["Test/TestChunks/AllWat500to650/Res2"]*2, 
          *["Test/TestChunks/AllWat500to650/Res4"]*2]
# folder = [*["Test/TestChunks/AllWat500to650/Res2"]*2, 
#           *["Test/TestChunks/AllWat500to650/Res2SC"]*2]
# folder = [*["Test/TestChunks/AllVac450to600/Res2"]*2, 
#           *["Test/TestChunks/AllVac450to600/Res2SC"]*2]
home = vs.get_home()

# Parameter for the test
test_param_string = "n_processes"
test_param_in_params = True
test_param_position = -1
test_param_label = "Number of processes used in parallel"

# Sorting and labelling data series
sorting_function = [lambda l : vu.sort_by_number(l, -1)]*4
series_label = [lambda s : f"Vacuum True {vu.find_numbers(s)[test_param_position]:.0f} Processes",
                lambda s : f"Vacuum False {vu.find_numbers(s)[test_param_position]:.0f} Processes",
                lambda s : f"Water True {vu.find_numbers(s)[test_param_position]:.0f} Processes",
                lambda s : f"Water False {vu.find_numbers(s)[test_param_position]:.0f} Processes"]
# series_label = [lambda s : f"Res 2 True {vu.find_numbers(s)[test_param_position]:.0f} Processes",
#                 lambda s : f"Res 2 False {vu.find_numbers(s)[test_param_position]:.0f} Processes",
#                 lambda s : f"Res 4 True {vu.find_numbers(s)[test_param_position]:.0f} Processes",
#                 lambda s : f"Res 4 False {vu.find_numbers(s)[test_param_position]:.0f} Processes"]
# series_label = [lambda s : f"MC True {vu.find_numbers(s)[test_param_position]:.0f} Processes",
#                 lambda s : f"MC False {vu.find_numbers(s)[test_param_position]:.0f} Processes",
#                 lambda s : f"SC True {vu.find_numbers(s)[test_param_position]:.0f} Processes",
#                 lambda s : f"SC False {vu.find_numbers(s)[test_param_position]:.0f} Processes"]
series_must = [*["True", "False"]*2] # leave "" per default
series_mustnt = ["Failed"]*4 # leave "" per default
series_column = [1]*4

# Scattering plot options
#plot_title = "Au 103 nm sphere"
plot_title = "Au 103 nm sphere in water"
# plot_title = "Au 103 nm sphere in vacuum"
#series_legend = ["Vacuum True", "Vacuum False", "Water True", "Water False"]
series_legend = ["Res 2 True", "Res 2 False", "Res 4 True", "Res 4 False"]
# series_legend = ["MC True", "MC False", "SC True", "SC False"]
# series_colors = [plab.cm.Reds, plab.cm.Reds, plab.cm.Blues, plab.cm.Blues]
series_colors = [plab.cm.Greens, plab.cm.Greens, plab.cm.Blues, plab.cm.Blues]
# series_colors = [plab.cm.Greens, plab.cm.Greens, plab.cm.Reds, plab.cm.Reds]
series_linestyles = [*["solid", "dashed"]*2]
plot_make_big = False
#plot_file = lambda n : os.path.join(home, "DataAnalysis/Chunks" + n)
# plot_file = lambda n : os.path.join(home, "DataAnalysis/ChunksRes4" + n)
# plot_file = lambda n : os.path.join(home, "DataAnalysis/ChunksSC" + n)
plot_file = lambda n : os.path.join(home, "DataAnalysis/ChunksSCVac" + n)

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

#%% WAVELENGTH MAXIMUM DIFFERENCE

if len(series)>1:
    colors = [*["darkgrey", "k"]*2]
else:
    colors = ["k"]

markers = ["o", "o", "D", "D"]

fig, [ax1, ax2] = plt.subplots(nrows=2, sharex=True, gridspec_kw={"hspace":0})

plt.suptitle("Difference in scattering for " + plot_title)

ax1.set_ylabel("$\lambda_{max}^{MEEP}-\lambda_{max}^{MIE}$ [nm]")
for tp, mwl, col, mar, leg in zip(test_param, max_wlen_diff, colors, markers, series_legend):
    ax1.plot(tp, mwl, color=col, marker=mar, markersize=7, linestyle="")
ax1.grid(True)
ax1.legend(series_legend)

ax2.set_ylabel("MSD( $\sigma^{MEEP} - \sigma^{MIE}$ )")
for tp, mr, col, mar, leg in zip(test_param, mean_residual, colors, markers, series_legend):
    ax2.plot(tp, mr, color=col, marker=mar, markersize=7, linestyle="")
ax2.grid(True)
ax2.legend(series_legend)

plt.xlabel(test_param_label)
# plt.figlegend(bbox_to_anchor=(.95, 0.7), bbox_transform=ax2.transAxes)
# ax2.legend()
plt.tight_layout()
plt.show()

vs.saveplot(plot_file("TheoryDiff.png"), overwrite=True)

#%% DIFFERENCE IN SCATTERING MAXIMUM

if len(series)>1:
    colors = [*["darkgrey", "k"]*2]
else:
    colors = ["k"]

markers = ["o", "o", "D", "D"]
plt.figure()
plt.title("Difference in scattering maximum for " + plot_title)
for tp, mwl, col, mar in zip(test_param, max_wlen_diff, colors, markers):
    plt.plot(tp, mwl, color=col, marker=mar, markersize=7, linestyle="")
plt.grid(True)
plt.legend(series_legend)
plt.xlabel(test_param_label)
plt.ylabel("Difference in wavelength $\lambda_{max}^{MEEP}-\lambda_{max}^{MIE}$ [nm]")
vs.saveplot(plot_file("WLenDiff.png"), overwrite=True)

#%% MEAN RESIDUAL

if len(series)>1:
    colors = [*["darkgrey", "k"]*2]
else:
    colors = ["k"]

markers = ["o", "o", "D", "D"]

plt.figure()
plt.title("Mean quadratic difference in effiency for " + plot_title)
for tp, mr, col, mar in zip(test_param, mean_residual, colors, markers):
    plt.plot(tp, mr, marker=mar, color=col, markersize=7, linestyle="")
plt.grid(True)
plt.legend(series_legend)
plt.xlabel(test_param_label)
plt.ylabel("Mean squared difference MSD( $C^{MEEP} - C^{MIE}$ )")
vs.saveplot(plot_file("QuaDiff.png"), overwrite=True)

#%% GET ELAPSED TIME COMPARED

elapsed_time = [[p["elapsed"] for p in par] for par in params]
# total_elapsed_time = [[sum(p["elapsed"]) for p in par] for par in params]

first_test_param = []
second_test_param = []
first_build_time = []
first_sim_time = []
second_flux_time = []
second_build_time = []
second_sim_time = []
total_elapsed_time = []
for enl, tpar, par in zip(elapsed_time, test_param, params):
    first_test_param.append( [] )
    first_build_time.append( [] )
    first_sim_time.append( [] )
    second_test_param.append( [] )
    second_flux_time.append( [] )
    second_build_time.append( [] )
    second_sim_time.append( [] )
    total_elapsed_time.append( [] )
    for e, tp, p in zip(enl, tpar, par):
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
        if p["split_chunks_evenly"]:
            total_elapsed_time[-1].append(sum(p["elapsed"]))
        else:
            if len(e)==5:
                total_elapsed_time[-1].append(sum(p["elapsed"]) + e[2])
            else:
                total_elapsed_time[-1].append(sum(p["elapsed"]) + e[0])

#%%

if len(series)>1:
    colors = [*["darkgrey", "k"]*2]
else:
    colors = ["k"]

markers = ["o", "o", "D", "D"]

plt.figure()
plt.title("elapsed total time for simulation of " + plot_title)
for tp, tot, col, mark in zip(test_param, total_elapsed_time, colors, markers):
    plt.plot(tp, tot, '-', marker=mark, color=col, markersize=7)
plt.legend(series_legend)
plt.xlabel(test_param_label)
plt.ylabel("elapsed time [s]")
vs.saveplot(plot_file("ComparedTotTime.png"), overwrite=True)
        
colors = ["r", "maroon", "darkorange", "orangered"]
fig = plt.figure()
plt.title("elapsed time for simulations of " + plot_title)
for tp, tim, col, leg in zip(first_test_param, first_sim_time, colors, series_legend):
    plt.plot(tp, tim, 'D-', color=col, label=leg + " Sim I")
for tp, tim, col, leg in zip(second_test_param, second_sim_time, colors, series_legend):
    plt.plot(tp, tim, 's-', color=col, label=leg + " Sim II")
plt.xlabel(test_param_label)
plt.ylabel("elapsed time in simulations [s]")
box = fig.axes[0].get_position()
box.y0 = box.y0 + .25 * (box.y1 - box.y0)
box.y1 = box.y1 + .10 * (box.y1 - box.y0)
box.x1 = box.x1 + .10 * (box.x1 - box.x0)
fig.axes[0].set_position(box)
leg = plt.legend(ncol=2, bbox_to_anchor=(.5, -.47), loc="lower center", frameon=False)
plt.savefig(plot_file("ComparedSimTime.png"), bbox_inches='tight')

colors = ["b", "navy", "cyan", "deepskyblue"]
fig = plt.figure()
plt.title("elapsed time for building of " + plot_title)
for tp, tim, col, leg in zip(first_test_param, first_build_time, colors, series_legend):
    plt.plot(tp, tim, 'D-', color=col, label=leg + " Sim I")
for tp, tim, col, leg in zip(second_test_param, second_build_time, colors, series_legend):
    plt.plot(tp, tim, 's-', color=col, label=leg + " Sim II")
plt.xlabel(test_param_label)
plt.ylabel("elapsed time in building [s]")
box = fig.axes[0].get_position()
box.y0 = box.y0 + .25 * (box.y1 - box.y0)
box.y1 = box.y1 + .10 * (box.y1 - box.y0)
box.x1 = box.x1 + .10 * (box.x1 - box.x0)
fig.axes[0].set_position(box)
leg = plt.legend(ncol=2, bbox_to_anchor=(.5, -.47), loc="lower center", frameon=False)
plt.savefig(plot_file("ComparedBuildTime.png"), bbox_inches='tight')

colors = ["m", "darkmagenta", "blueviolet", "indigo"]
fig = plt.figure()
plt.title("elapsed time for loading flux of " + plot_title)
for tp, tim, col, leg in zip(second_test_param, second_flux_time, colors, series_legend):
    plt.plot(tp, tim, 's-', color=col, label=leg + " Sim II")
plt.xlabel(test_param_label)
plt.ylabel("elapsed time in loading flux [s]")
box = fig.axes[0].get_position()
box.y0 = box.y0 + .15 * (box.y1 - box.y0)
box.y1 = box.y1 + .10 * (box.y1 - box.y0)
box.x1 = box.x1 + .10 * (box.x1 - box.x0)
fig.axes[0].set_position(box)
leg = plt.legend(ncol=2, bbox_to_anchor=(.5, -.3), loc="lower center", frameon=False)
plt.savefig(plot_file("ComparedLoadTime.png"), bbox_inches='tight')

#%%

if len(series)>1:
    colors = [*["darkgrey", "k"]*2]
else:
    colors = ["k"]

markers = ["o", "o", "D", "D"]

fig, axes = plt.subplots(nrows=2, sharex=True, gridspec_kw={"hspace":0})
axes = [*[axes[0]]*2, *[axes[1]]*2]
plt.suptitle("elapsed total time for simulation of " + plot_title)
for ax, tp, tot, col, mark in zip(axes, test_param, total_elapsed_time, colors, markers):
    ax.plot(tp, tot, '-', marker=mark, color=col, markersize=7)
axes[0].legend(series_legend[:2])
axes[2].legend(series_legend[2:])
axes[0].set_ylabel("elapsed time [s]")
axes[2].set_ylabel("elapsed time [s]")
plt.xlabel(test_param_label)
vs.saveplot(plot_file("ComparedTotTime.png"), overwrite=True)
        
colors = ["r", "maroon", "darkorange", "orangered"]
fig, axes = plt.subplots(nrows=2, sharex=True, gridspec_kw={"hspace":0})
axes = [*[axes[0]]*2, *[axes[1]]*2]
plt.suptitle("elapsed time for simulations of " + plot_title)
for ax, tp, tim, col, leg in zip(axes, first_test_param, first_sim_time, colors, series_legend):
    ax.plot(tp, tim, 'D-', color=col, label=leg + " Sim I")
for ax, tp, tim, col, leg in zip(axes, second_test_param, second_sim_time, colors, series_legend):
    ax.plot(tp, tim, 's-', color=col, label=leg + " Sim II")
plt.xlabel(test_param_label)
axes[0].legend()
axes[2].legend()
axes[0].set_ylabel("elapsed time [s]")
axes[2].set_ylabel("elapsed time [s]")
plt.savefig(plot_file("ComparedSimTime.png"), bbox_inches='tight')

colors = ["b", "navy", "cyan", "deepskyblue"]
fig, axes = plt.subplots(nrows=2, sharex=True, gridspec_kw={"hspace":0})
axes = [*[axes[0]]*2, *[axes[1]]*2]
plt.suptitle("elapsed time for building of " + plot_title)
for ax, tp, tim, col, leg in zip(axes, first_test_param, first_build_time, colors, series_legend):
    ax.plot(tp, tim, 'D-', color=col, label=leg + " Sim I")
for ax, tp, tim, col, leg in zip(axes, second_test_param, second_build_time, colors, series_legend):
    ax.plot(tp, tim, 's-', color=col, label=leg + " Sim II")
plt.xlabel(test_param_label)
axes[0].legend(ncol=2)
axes[2].legend(ncol=2)
axes[0].set_ylabel("elapsed time [s]")
axes[2].set_ylabel("elapsed time [s]")
plt.savefig(plot_file("ComparedBuildTime.png"), bbox_inches='tight')

colors = ["m", "darkmagenta", "blueviolet", "indigo"]
fig, axes = plt.subplots(nrows=2, sharex=True, gridspec_kw={"hspace":0})
axes = [*[axes[0]]*2, *[axes[1]]*2]
plt.suptitle("elapsed time for loading flux of " + plot_title)
for ax, tp, tim, col, leg in zip(axes, second_test_param, second_flux_time, colors, series_legend):
    ax.plot(tp, tim, 's-', color=col, label=leg + " Sim II")
plt.xlabel(test_param_label)
axes[0].legend()
axes[2].legend()
axes[0].set_ylabel("elapsed time [s]")
axes[2].set_ylabel("elapsed time [s]")
plt.savefig(plot_file("ComparedLoadTime.png"), bbox_inches='tight')

#%% PLOT NORMALIZED

colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

fig = plt.figure()
plt.title("Normalized scattering for " + plot_title)
for s, d, p, sc, psl, pc, pls in zip(series, data, params, series_column, 
                                     series_label, colors, series_linestyles):

    for ss, sd, sp, spc in zip(s, d, p, pc):
        index_argmax = np.argmax(abs(sd[:,sc]))
        plt.plot(sd[:,0], sd[:,sc] / sd[index_argmax, sc], 
                 linestyle=pls, color=spc, label=psl(ss))

# plt.plot(data[1][-1][:,0], theory[1][-1] / max(theory[0][-1]), 
#           linestyle="dotted", color='red', label="Mie Theory Vacuum")
plt.plot(data[-1][-1][:,0], theory[-1][-1] / max(theory[-1][-1]), 
          linestyle="dotted", color='blue', label="Mie Theory Water")
plt.xlabel(r"Wavelength $\lambda$ [nm]")
plt.ylabel("Normalized Scattering Cross Section")
box = fig.axes[0].get_position()
box.x1 = box.x1 - .15 * (box.x1 - box.x0)
fig.axes[0].set_position(box)
leg = plt.legend(bbox_to_anchor=(1.20, .5), loc="center right", frameon=False)
if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    del mng
vs.saveplot(plot_file("AllScattNorm.png"), overwrite=True)

#%% PLOT IN UNITS

colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

fig = plt.figure()
plt.title("Scattering for " + plot_title)
for s, d, p, sc, psl, pc, pls in zip(series, data, params, series_column, 
                                     series_label, colors, series_linestyles):

    for ss, sd, sp, spc in zip(s, d, p, pc):
        sign = np.sign( sd[ np.argmax(abs(sd[:,sc])), sc ] )
        plt.plot(sd[:,0], sign * sd[:,sc] * np.pi * (sp["r"] * sp["from_um_factor"] * 1e3)**2,
                 linestyle=pls, color=spc, label=psl(ss))
            
# plt.plot(data[1][-1][:,0], theory[1][-1] * np.pi * (params[0][-1]["r"] * params[0][-1]["from_um_factor"] * 1e3)**2,
#           linestyle="dotted", color='red', label="Mie Theory Vacuum")
plt.plot(data[-1][-1][:,0], theory[-1][-1] * np.pi * (params[-1][-1]["r"] * params[-1][-1]["from_um_factor"] * 1e3)**2,
          linestyle="dotted", color='blue', label="Mie Theory Water")
plt.xlabel(r"Wavelength $\lambda$ [nm]")
plt.ylabel(r"Scattering Cross Section [nm$^2$]")
box = fig.axes[0].get_position()
box.x1 = box.x1 - .15 * (box.x1 - box.x0)
fig.axes[0].set_position(box)
leg = plt.legend(bbox_to_anchor=(1.20, .5), loc="center right", frameon=False)
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
            
plt.xlabel(r"Wavelength $\lambda$ [nm]")
plt.ylabel(r"Difference in Scattering Cross Section [nm$^2$]")
plt.legend()
if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    del mng
vs.saveplot(plot_file("AllScattDiff.png"), overwrite=True)
