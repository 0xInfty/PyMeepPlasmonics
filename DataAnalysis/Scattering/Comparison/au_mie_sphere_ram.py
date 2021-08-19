#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 12:24:04 2020

@author: vall
"""

import h5py as h5
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
# folder = ["Test/TestResetMeep", 
#           "Test/TestResetMeep",
#           "Test/TestResetMeep"]
folder = ["Test/TestRAM/TestRAMRes/AllVac", 
          "Test/TestRAM/TestRAMRes/AllWat"]
home = vs.get_home()

# Parameter for the test
test_param_string = "resolution"
test_param_in_params = True
test_param_position = -1
test_param_label = "Resolution"

# Sorting and labelling data series
# sorting_function = [lambda l : l]*3
sorting_function = [lambda l : vu.sort_by_number(l, -1)]*2
series_label = [lambda s : f"Vacuum Resolution {vu.find_numbers(s)[-1]}",
                lambda s : f"Water Resolution {vu.find_numbers(s)[-1]}"]
# series_label = [lambda s : "Reset Meep",
#                 lambda s : "Reset Meep Refined",
#                 lambda s : "No Reset"]
# series_must = ["Original", "New", "No"] # leave "" per default
# series_mustnt = [""]*3 # leave "" per default
series_must = [""]*2 # leave "" per default
series_mustnt = [""]*2 # leave "" per default
# series_column = [1]*3
series_column = [1]*2

# Scattering plot options
plot_title = "Au 103 nm sphere"
# series_legend = ["Reset Meep", "Reset Meep Refined", "No Reset"]
series_legend = ["Vacuum", "Water"]
series_colors = [plab.cm.Reds, plab.cm.Blues]
# series_colors = [plab.cm.Reds, plab.cm.Reds, plab.cm.Blues]
# series_linestyles = ["solid"]*3
series_linestyles = ["solid"]*2
plot_make_big = False
# plot_file = lambda n : os.path.join(home, "DataAnalysis/Reset" + n)
plot_file = lambda n : os.path.join(home, "DataAnalysis/RAMRes" + n)

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

    for s, p in zip(series[-1], params[-1]):
        f = h5.File(file[-1](s, "RAM.h5"))
        p["used_ram"] = np.array(f["RAM"])
    
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
ax1.set_xticks(resolution[ np.argmax([len(res) for res in resolution]) ])
ax1.legend(series_legend)

ax2.set_ylabel("MSD( $\sigma^{MEEP} - \sigma^{MIE}$ )")
for tp, mr, col, mar, leg in zip(test_param, mean_residual, colors, markers, series_legend):
    ax2.plot(tp, mr, color=col, marker=mar, markersize=7, linestyle="")
ax2.grid(True)
ax2.set_xticks(resolution[ np.argmax([len(res) for res in resolution]) ])
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
plt.xticks( resolution[ np.argmax([len(res) for res in resolution]) ] )
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
plt.xticks( resolution[ np.argmax([len(res) for res in resolution]) ] )
plt.ylabel("Mean squared difference MSD( $C^{MEEP} - C^{MIE}$ )")
vs.saveplot(plot_file("QuaDiffJoint.png"), overwrite=True)

#%% MEAN RESIDUAL LEFT AND RIGHT

if len(series)>1:
    colors = ["darkgrey", "k"]
    colors_right = ["red", "maroon"]
    colors_left = ["darkviolet", "rebeccapurple"]
else:
    colors = ["k"]
    colors_right = ["red"]
    colors_left = ["darkviolet"]

colors = colors*2
colors_right = colors_right*2
colors_left = colors_left*2

markers = ["o", "o", "D", "D"]

plt.figure()
plt.title("Mean quadratic difference in effiency for " + plot_title)
lines = []
medium_lines = []
lines_legend = []
medium_lines_legend = []
for i in range(len(test_param)):
    l1, = plt.plot(test_param[i], mean_residual_left[i], '.', color=colors_left[i], 
                 marker=markers[i], markersize=7)
    if i <= 1:
        lines.append(l1)
        lines_legend.append(series_legend[i]+" left")
    if i == 0 or i == 2:
        medium_lines.append(l1)
        medium_lines_legend.append(series_legend[i])
    l2, = plt.plot(test_param[i], mean_residual_right[i], '.', color=colors_right[i], 
                  marker=markers[i], markersize=7)
    if i <= 1:
        lines.append(l2)
        lines_legend.append(series_legend[i]+" right")
    l3, = plt.plot(test_param[i], mean_residual[i], '.', color=colors[i], 
                  marker=markers[i], markersize=7)
    if i <= 1:
        lines.append(l3)
        lines_legend.append(series_legend[i]+" all")
plt.grid(True)
# plt.legend(lines, lines_legend)
first_legend = plt.legend(lines, lines_legend)
second_legend = plt.legend(medium_lines, medium_lines_legend, loc="center left")
plt.gca().add_artist(first_legend)
plt.xlabel(test_param_label)
plt.ylabel("Mean squared difference MSD( $C^{MEEP} - C^{MIE}$ )")
vs.saveplot(plot_file("QuaDiff.png"), overwrite=True)

#%% MEAN RESIDUAL SUBPLOTS

if len(series)>1:
    colors = [*["darkgrey", "k"]*2]
else:
    colors = ["k"]

markers = ["o", "D"]

fig, axes = plt.subplots(2, sharex=True, gridspec_kw={"hspace":0})
plt.suptitle("Mean quadratic difference in effiency for " + plot_title)
for i in range(len(axes)):
    axes[i].plot(test_param[i], mean_residual[i], "k", 
                 marker=markers[i], markersize=7, linestyle="")
    axes[i].grid(True)
    axes[i].legend([series_legend[i]])
    axes[i].set_ylabel("MSD( $C^{MEEP} - C^{MIE}$ )")
axes[1].yaxis.tick_right()
axes[1].yaxis.set_label_position("right")
plt.xlabel(test_param_label)
plt.xticks( resolution[ np.argmax([len(res) for res in resolution]) ] )
vs.saveplot(plot_file("QuaDiff.png"), overwrite=True)

#%% MEAN RESIDUAL SUBPLOTS LEFT AND RIGHT

if len(series)>1:
    colors = ["darkgrey", "k"]
    colors_right = ["red", "maroon"]
    colors_left = ["darkviolet", "rebeccapurple"]
else:
    colors = ["k"]
    colors_right = ["red"]
    colors_left = ["darkviolet"]

markers = ["o", "D"]

fig, axes = plt.subplots(2, sharex=True, gridspec_kw={"hspace":0})
plt.suptitle("Mean quadratic difference in effiency for " + plot_title)
for i in range(len(axes)):
    axes[i].plot(test_param[i], mean_residual_right[i], "red", 
                 marker=markers[i], markersize=7, linestyle="")
    axes[i].plot(test_param[i], mean_residual_left[i], "darkviolet", 
                 marker=markers[i], markersize=7, linestyle="")
    axes[i].plot(test_param[i], mean_residual[i], "k", 
                 marker=markers[i], markersize=7, linestyle="")
    axes[i].grid(True)
    axes[i].legend([series_legend[i] + " left",
                    series_legend[i] + " right",
                    series_legend[i] + " all"])
    axes[i].set_ylabel("MSD( $C^{MEEP} - C^{MIE}$ )")
axes[1].yaxis.tick_right()
axes[1].yaxis.set_label_position("right")
plt.xlabel(test_param_label)
plt.xticks( resolution[ np.argmax([len(res) for res in resolution]) ] )
plt.tight_layout()
vs.saveplot(plot_file("QuaDiff.png"), overwrite=True)

#%% GET elapsed TIME COMPARED

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
plt.xticks( resolution[ np.argmax([len(res) for res in resolution]) ] )
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
plt.xticks( resolution[ np.argmax([len(res) for res in resolution]) ] )
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
plt.xticks( resolution[ np.argmax([len(res) for res in resolution]) ] )
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
plt.xticks( resolution[ np.argmax([len(res) for res in resolution]) ] )
plt.ylabel("elapsed time in loading flux [s]")
box = fig.axes[0].get_position()
box.y0 = box.y0 + .15 * (box.y1 - box.y0)
box.y1 = box.y1 + .10 * (box.y1 - box.y0)
box.x1 = box.x1 + .10 * (box.x1 - box.x0)
fig.axes[0].set_position(box)
leg = plt.legend(ncol=2, bbox_to_anchor=(.5, -.3), loc="lower center", frameon=False)
plt.savefig(plot_file("ComparedLoadTime.png"), bbox_inches='tight')

#%% ALL RAM MEMORY

reference = ["Módulos",
             "Parámetros",
             "Inicial (Sim I)",
             "mp.Simulation (Sim I)",
             "Flujo (Sim I)",
             "sim.init_sim() (Sim I)",
             "Principio (Sim I)",
             "Mitad (Sim I)",
             "Final (Sim I)",
             "Inicial (Sim II)",
             r"mp.Simulation (Sim II)",
             "Flujo (Sim II)",
             "sim.init_sim() (Sim II)",
             "load_midflux() (Sim II)",
             "Flujo negado (Sim II)",
             "Principio (Sim II)",
             "Mitad (Sim II)",
             "Final (Sim II)"]

total_ram = [[np.array([sum(ur) for ur in p["used_ram"]]) for p in par] for par in params]
max_ram = [[p["used_ram"][:, np.argmax([sum(ur) for ur in p["used_ram"].T]) ] for p in par] for par in params]
min_ram = [[p["used_ram"][:, np.argmin([sum(ur) for ur in p["used_ram"].T]) ] for p in par] for par in params]
mean_ram = [[np.array([np.mean(ur) for ur in p["used_ram"]]) for p in par] for par in params]

total_ram = [[ tr / (1024)**2 for tr in tram] for tram in total_ram]
max_ram = [[ tr / (1024)**2 for tr in tram] for tram in max_ram]
min_ram = [[ tr / (1024)**2 for tr in tram] for tram in min_ram]
mean_ram = [[ tr / (1024)**2 for tr in tram] for tram in mean_ram]

#%% RAM PER SUBPROCESS

# colors = [["C0"], ["C4"], ["C3"]]
colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

fig = plt.figure()

lines = []
lines_legend = []
for i in range(len(series)):
    for j in range(len(series[i])):
        l = plt.errorbar(reference, mean_ram[i][j], 
                         np.array([mean_ram[i][j] - min_ram[i][j], 
                                   max_ram[i][j] - mean_ram[i][j]]),
                         marker="o", color=colors[i][j], linestyle="-",
                         elinewidth=1.5, capsize=7, capthick=1.5, markersize=7,
                         linewidth=1, zorder=2)
        lines.append(l)
        lines_legend.append(series_label[i](series[i][j]))
        
patches = []
patches.append( plt.axvspan(*["Inicial (Sim I)", "Principio (Sim I)"], alpha=0.15, color='grey', zorder=1) )
patches.append( plt.axvspan(*["Principio (Sim I)", "Final (Sim I)"], alpha=0.3, color='grey', zorder=1) )
patches.append( plt.axvspan(*["Inicial (Sim II)", "Principio (Sim II)"], alpha=0.1, color='gold', zorder=1) )
patches.append( plt.axvspan(*["Principio (Sim II)", "Final (Sim II)"], alpha=0.2, color='gold', zorder=1) )
patches_legend = ["Configuration Sim I", "Running Sim I",
                  "Configuration Sim II", "Running Sim II"]
        
plt.xticks(rotation=-30, ha="left")
plt.grid(True)
plt.ylabel("RAM Memory Per Subprocess [GiB]")

plt.legend(lines, lines_legend)
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.tight_layout()

# first_legend = plt.legend(lines, lines_legend)
first_legend = plt.legend(lines, lines_legend, ncol=2)
second_legend = plt.legend(patches, patches_legend, loc="center left")
plt.gca().add_artist(first_legend)

plt.savefig(plot_file("AllRAM.png"), bbox_inches='tight')

#%% RAM TOTAL

# colors = [["C0"], ["C4"], ["C3"]]
colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

fig = plt.figure()

lines = []
lines_legend = []
k = 0
for i in range(len(series)):
    for j in range(len(series[i])):
        l = plt.bar(np.arange(len(reference)) + (k+.5)/sum([len(s) for s in series]), 
                    total_ram[i][j], color=colors[i][j], alpha=1,
                    width=1/sum([len(s) for s in series]),
                    zorder=2)
        lines.append(l)
        lines_legend.append(series_label[i](series[i][j]))
        k += 1

patches = []
patches.append( plt.axvspan(reference.index("Inicial (Sim I)"), 
                            reference.index("Principio (Sim I)"), 
                            alpha=0.15, color='grey', zorder=1) )
patches.append( plt.axvspan(reference.index("Principio (Sim I)"), 
                            reference.index("Final (Sim I)"), 
                            alpha=0.3, color='grey', zorder=1) )
patches.append( plt.axvspan(reference.index("Inicial (Sim II)"),
                            reference.index("Principio (Sim II)"), 
                            alpha=0.1, color='gold', zorder=1) )
patches.append( plt.axvspan(reference.index("Principio (Sim II)"),
                            reference.index("Final (Sim II)"), 
                            alpha=0.2, color='gold', zorder=1) )
patches_legend = ["Configuration Sim I", "Running Sim I",
                  "Configuration Sim II", "Running Sim II"]

if params[0][0]["sysname"]=="MC":
    plt.ylim(0, 16)
elif params[0][0]["sysname"]=="SC":
    plt.ylim(0, 48)
else:
    plt.ylim(0, 128)
plt.xlim(0, len(reference))
plt.xticks(np.arange(len(reference)), reference)
plt.xticks(rotation=-30, ha="left")
plt.grid(True)
plt.ylabel("Total RAM Memory [GiB]")

plt.legend(lines, lines_legend)
# first_legend = plt.legend(lines, lines_legend)
first_legend = plt.legend(lines, lines_legend, ncol=2)
second_legend = plt.legend(patches, patches_legend, loc="center left")
plt.gca().add_artist(first_legend)

second_ax = plt.twinx()
second_ax.set_ylabel("Total RAM Memory [%]")
second_ax.set_ylim(0, 100)
second_ax.set_zorder(0)

mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.tight_layout()

plt.savefig(plot_file("AllTotalRAM.png"), bbox_inches='tight')

#%% COMPARE SIMULATIONS I AND II

crossed_reference = ["Inicial",
                     "mp.Simulation",
                     "Flujo",
                     "sim.init_sim()",
                     "load_midflux()",
                     "Flujo negado",
                     "Principio",
                     "Mitad",
                     "Final"]

index_sim_I = []
reference_sim_I = []
index_sim_II = []
reference_sim_II = []
for ref in crossed_reference:
    try:
        i = reference.index(ref + " (Sim I)")
        index_sim_I.append(i)
        reference_sim_I.append(ref)
    except:
        pass
for ref in crossed_reference:
    try:
        i = reference.index(ref + " (Sim II)")
        index_sim_II.append(i)
        reference_sim_II.append(ref)
    except:
        pass
    
common_index = []
common_reference = []
for i, ref in enumerate(reference_sim_I):
    if ref in reference_sim_II:
        common_index.append(reference_sim_II.index(ref))
        common_reference.append(ref)
    
total_ram_sim_I = [[np.array([tr[i] for i in index_sim_I]) for tr in tram] for tram in total_ram]
max_ram_sim_I = [[np.array([tr[i] for i in index_sim_I]) for tr in tram] for tram in max_ram]
min_ram_sim_I = [[np.array([tr[i] for i in index_sim_I]) for tr in tram] for tram in min_ram]
mean_ram_sim_I = [[np.array([tr[i] for i in index_sim_I]) for tr in tram] for tram in mean_ram]

total_ram_sim_II = [[np.array([tr[i] for i in index_sim_II]) for tr in tram] for tram in total_ram]
max_ram_sim_II = [[np.array([tr[i] for i in index_sim_II]) for tr in tram] for tram in max_ram]
min_ram_sim_II = [[np.array([tr[i] for i in index_sim_II]) for tr in tram] for tram in min_ram]
mean_ram_sim_II = [[np.array([tr[i] for i in index_sim_II]) for tr in tram] for tram in mean_ram]

total_ram_common_sim_II = [[np.array([tr[i] for i in common_index]) for tr in tram] for tram in total_ram_sim_II]
max_ram_common_sim_II = [[np.array([tr[i] for i in common_index]) for tr in tram] for tram in max_ram_sim_II]
min_ram_common_sim_II = [[np.array([tr[i] for i in common_index]) for tr in tram] for tram in min_ram_sim_II]
mean_ram_common_sim_II = [[np.array([tr[i] for i in common_index]) for tr in tram] for tram in mean_ram_sim_II]

#%% RAM SIM I AND II

# colors = [["C0"], ["C4"], ["C3"]]
colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

fig = plt.figure()

lines = []
lines_legend = []
second_lines = []
second_lines_legend = []
for i in range(len(series)):
    for j in range(len(series[i])):
        l = plt.errorbar(common_reference, mean_ram_sim_I[i][j] - mean_ram_sim_I[i][j][0], 
                         np.array([mean_ram_sim_I[i][j] - min_ram_sim_I[i][j], 
                                   max_ram_sim_I[i][j] - mean_ram_sim_I[i][j]]),
                         marker="s", color=colors[i][j], linestyle="-",
                         elinewidth=1.5, capsize=7, capthick=1.5, markersize=6,
                         linewidth=1, zorder=2)
        lines.append(l)
        lines_legend.append(series_label[i](series[i][j]))
        # if j==0 and i==0:
        if j==4 and i==0:
            second_lines.append(l)
            second_lines_legend.append("Sim I")
        l = plt.errorbar(common_reference, mean_ram_common_sim_II[i][j] - mean_ram_common_sim_II[i][j][0], 
                         np.array([mean_ram_common_sim_II[i][j] - min_ram_common_sim_II[i][j], 
                                   max_ram_common_sim_II[i][j] - mean_ram_common_sim_II[i][j]]),
                         marker="D", color=colors[i][j], linestyle="dashed",
                         elinewidth=1.5, capsize=7, capthick=1.5, markersize=5,
                         linewidth=1, zorder=2)
        # if j==0 and i==0:
        if j==4 and i==0:
            second_lines.append(l)
            second_lines_legend.append("Sim II")
        
patches = []
patches.append( plt.axvspan(*["Inicial", "Principio"], alpha=0.15, color='peru', zorder=1) )
patches.append( plt.axvspan(*["Principio", "Final"], alpha=0.3, color='peru', zorder=1) )
patches_legend = ["Configuration", "Running"]
        
plt.xticks(rotation=-30, ha="left")
plt.grid(True)
plt.ylabel("Dedicated RAM Memory Per Subprocess [GiB]")

plt.legend(lines, lines_legend)
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.tight_layout()

# first_legend = plt.legend(lines, lines_legend)
first_legend = plt.legend(lines, lines_legend, ncol=2)
second_legend = plt.legend(patches, patches_legend, loc="center left")
third_legend = plt.legend(second_lines, second_lines_legend, loc="lower left")
ax = plt.gca()
ax.add_artist(first_legend)
ax.add_artist(second_legend)

plt.savefig(plot_file("AllRAMCommon.png"), bbox_inches='tight')

#%% KEY POINTS OF COMPARISON

key_reference = ["Principio", "Mitad"]
key_mask = ["Antes", "Durante"]

key_index_sim_I = []
key_index_sim_II = []
for ref in key_reference:
    try:
        i = reference.index(ref + " (Sim I)")
        key_index_sim_I.append(i)
    except:
        pass
for ref in key_reference:
    try:
        i = reference.index(ref + " (Sim II)")
        key_index_sim_II.append(i)
    except:
        pass
    
total_ram_key_sim_I = [[np.array([tr[i] for i in key_index_sim_I]) for tr in tram] for tram in total_ram]
max_ram_key_sim_I = [[np.array([tr[i] for i in key_index_sim_I]) for tr in tram] for tram in max_ram]
min_ram_key_sim_I = [[np.array([tr[i] for i in key_index_sim_I]) for tr in tram] for tram in min_ram]
mean_ram_key_sim_I = [[np.array([tr[i] for i in key_index_sim_I]) for tr in tram] for tram in mean_ram]

total_ram_key_sim_II = [[np.array([tr[i] for i in key_index_sim_II]) for tr in tram] for tram in total_ram]
max_ram_key_sim_II = [[np.array([tr[i] for i in key_index_sim_II]) for tr in tram] for tram in max_ram]
min_ram_key_sim_II = [[np.array([tr[i] for i in key_index_sim_II]) for tr in tram] for tram in min_ram]
mean_ram_key_sim_II = [[np.array([tr[i] for i in key_index_sim_II]) for tr in tram] for tram in mean_ram]

#%%

# colors = [["C0"], ["C4"], ["C3"]]
colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

fig = plt.figure()

lines = []
lines_legend = []
second_lines = []
second_lines_legend = []
k = 0
for i in range(len(series)):
    for j in range(len(series[i])):
        l = plt.bar(np.arange(len(key_mask)) + (k+.5)/(2*sum([len(s) for s in series])), 
                    total_ram_key_sim_I[i][j], color=colors[i][j], 
                    alpha=1,
                    width=1/(2*sum([len(s) for s in series])),
                    zorder=2)
        lines.append(l)
        lines_legend.append(series_label[i](series[i][j]))
        k += 1
        if j==0 and i==0:
            second_lines.append(l)
            second_lines_legend.append("Sim I")
        l = plt.bar(np.arange(len(key_mask)) + (k+.5)/(2*sum([len(s) for s in series])), 
                    total_ram_key_sim_II[i][j], color=colors[i][j], 
                    hatch="*", alpha=1,
                    width=1/(2*sum([len(s) for s in series])),
                    zorder=2)
        if j==0 and i==0:
            second_lines.append(l)
            second_lines_legend.append("Sim II")
        k += 1

if params[0][0]["sysname"]=="MC":
    plt.ylim(0, 16)
elif params[0][0]["sysname"]=="SC":
    plt.ylim(0, 48)
else:
    plt.ylim(0, 128)
plt.xlim(0, len(key_reference))
plt.xticks(np.arange(len(key_reference)), key_mask)
plt.xticks(rotation=0, ha="left")
plt.grid(True)
plt.ylabel("Total RAM Memory [GiB]")

first_legend = plt.legend(lines, lines_legend, loc="upper left", ncol=2)
second_legend = plt.legend(second_lines, second_lines_legend, loc="center left")
plt.gca().add_artist(first_legend)

second_ax = plt.twinx()
second_ax.set_ylabel("Total RAM Memory [%]")
second_ax.set_ylim(0, 100)
second_ax.set_zorder(0)

mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.tight_layout()

plt.savefig(plot_file("AllKeyRAMStage.png"), bbox_inches='tight')

#%%

# colors = [["C0"], ["C3"], ["C4"]]
colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

fig, axes = plt.subplots(ncols=2, nrows=1, sharey=True, gridspec_kw={"wspace":0})

lines = []
lines_legend = []
second_lines = []
second_lines_legend = []
k = 0
for i in range(len(series)):
    for j in range(len(series[i])):
        l = axes[0].bar(np.arange(len(key_mask)) + (k+.5)/sum([len(s) for s in series]), 
                        total_ram_key_sim_I[i][j], color=colors[i][j], 
                        alpha=1, width=1/sum([len(s) for s in series]),
                        zorder=2)
        lines.append(l)
        lines_legend.append(series_label[i](series[i][j]))
        # k += 1
        l = axes[1].bar(np.arange(len(key_mask)) + (k+.5)/sum([len(s) for s in series]), 
                        total_ram_key_sim_II[i][j], color=colors[i][j], 
                        # hatch="*", 
                        alpha=1, width=1/sum([len(s) for s in series]),
                        zorder=2)
        k += 1

axes[0].set_title("Simulation I")
axes[1].set_title("Simulation II")

if params[0][0]["sysname"]=="MC":
    axes[0].set_ylim(0, 16)
elif params[0][0]["sysname"]=="SC":
    axes[0].set_ylim(0, 48)
else:
    axes[0].set_ylim(0, 128)
for ax in axes:
    ax.set_xlim(0, len(key_reference))
    ax.set_xticks(np.arange(len(key_reference)))
    ax.set_xticklabels(key_mask, rotation=0, ha="left")
    ax.grid(True, axis="y")
axes[0].set_ylabel("Total RAM Memory [GiB]")

# axes[0].legend(lines, lines_legend, loc="upper left")
axes[0].legend(lines, lines_legend, loc="upper left", ncol=2)

second_ax = axes[1].twinx()
second_ax.set_ylabel("Total RAM Memory [%]")
second_ax.set_ylim(0, 100)
second_ax.set_zorder(0)

mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.tight_layout()

plt.savefig(plot_file("AllKeyRAMSim.png"), bbox_inches='tight')

#%%

def cubic_fit(X, A, B, C, D):
    return A * X**3 + B * X**2 + C * X + D

def cubic_simple_fit(X, A, B, C):
    return A * X**3 + B * X**2 + C

rsq_sim_I = []
parameters_sim_I = []
for i in range(len(total_ram_key_sim_I)):
    rsq_sim_I.append([])
    parameters_sim_I.append([])
    for j in range(len(total_ram_key_sim_I[i][0])):
        rs, pars = va.nonlinear_fit(np.array(test_param[i]), 
                                    np.array([tr[j] for tr in total_ram_key_sim_I[i]]), 
                                    cubic_simple_fit,
                                    par_units=["GiB", "GiB"],
                                    showplot=False)
        rsq_sim_I[-1].append(rs)
        parameters_sim_I[-1].append(pars)

rsq_sim_II = []
parameters_sim_II = []
for i in range(len(total_ram_key_sim_II)):
    rsq_sim_II.append([])
    parameters_sim_II.append([])
    for j in range(len(total_ram_key_sim_II[i][0])):
        rs, pars = va.nonlinear_fit(np.array(test_param[i]), 
                                   np.array([tr[j] for tr in total_ram_key_sim_II[i]]), 
                                   cubic_simple_fit,
                                   par_units=["GiB", "GiB"],
                                   showplot=True)
        rsq_sim_II[-1].append(rs)
        parameters_sim_II[-1].append(pars)

# rsq, parameters = va.nonlinear_fit(np.array(test_param[0]), 
#                                    np.array([tr[0] for tr in total_ram_key_sim_I[0]]), 
#                                    cubic_simple_fit,
#                                    par_units=["GiB", "GiB"])#, "GiB", "GiB"])

#%%
    
fig = plt.figure(constrained_layout=True)
subplot_grid = fig.add_gridspec(3, 2)
axes = [[fig.add_subplot(subplot_grid[0:2, 0]),
         fig.add_subplot(subplot_grid[2, 0])],
        [fig.add_subplot(subplot_grid[0:2, 1]),
         fig.add_subplot(subplot_grid[2, 1])]]

colors = [["r", "maroon"], ["b", "navy"]]

for i in range(len(series)):
    
    # Fits
    axes[i][0].plot(test_param[i], cubic_simple_fit(np.array(test_param[i]), 
                                                    parameters_sim_I[i][0][0][0],
                                                    parameters_sim_I[i][0][1][0],
                                                    parameters_sim_I[i][0][2][0]), 
                    marker="", linestyle="-", color=colors[i][0], label="Before Sim I")
    axes[i][0].plot(test_param[i], cubic_simple_fit(np.array(test_param[i]), 
                                                    parameters_sim_I[i][1][0][0],
                                                    parameters_sim_I[i][1][1][0],
                                                    parameters_sim_I[i][1][2][0]),
                    marker="", linestyle="-", color=colors[i][1], label="After Sim I")
    axes[i][0].plot(test_param[i], cubic_simple_fit(np.array(test_param[i]), 
                                                    parameters_sim_II[i][0][0][0],
                                                    parameters_sim_II[i][0][1][0],
                                                    parameters_sim_II[i][0][2][0]), 
                    marker="", linestyle=":", color=colors[i][0], label="Before Sim II")
    axes[i][0].plot(test_param[i], cubic_simple_fit(np.array(test_param[i]), 
                                                    parameters_sim_II[i][1][0][0],
                                                    parameters_sim_II[i][1][1][0],
                                                    parameters_sim_II[i][1][2][0]),
                    marker="", linestyle=":", color=colors[i][1], label="After Sim II")
    
    # Points
    axes[i][0].plot(test_param[i], [tr[0] for tr in total_ram_key_sim_I[i]], 
                    marker="o", linestyle="", color=colors[i][0], label="Before Sim I")
    axes[i][0].plot(test_param[i], [tr[1] for tr in total_ram_key_sim_I[i]], 
                    marker="o", linestyle="", color=colors[i][1], label="During Sim I")
    axes[i][0].plot(test_param[i], [tr[0] for tr in total_ram_key_sim_II[i]], 
                    marker="D", linestyle="", color=colors[i][0], label="Before Sim II")
    axes[i][0].plot(test_param[i], [tr[1] for tr in total_ram_key_sim_II[i]], 
                    marker="D", linestyle="", color=colors[i][1], label="During Sim II")
    
    # Residuum
    axes[i][1].plot(test_param[i], np.array([tr[0] for tr in total_ram_key_sim_I[i]]) - 
                                    cubic_simple_fit(np.array(test_param[i]), 
                                                     parameters_sim_I[i][0][0][0],
                                                     parameters_sim_I[i][0][1][0],
                                                     parameters_sim_I[i][0][2][0]), 
                    marker="o", linestyle="", color=colors[i][0], label="Before Sim I")
    axes[i][1].plot(test_param[i], np.array([tr[1] for tr in total_ram_key_sim_I[i]]) - 
                                    cubic_simple_fit(np.array(test_param[i]), 
                                                     parameters_sim_I[i][1][0][0],
                                                     parameters_sim_I[i][1][1][0],
                                                     parameters_sim_I[i][1][2][0]), 
                    marker="o", linestyle="", color=colors[i][1], label="After Sim I")
    axes[i][1].plot(test_param[i], np.array([tr[0] for tr in total_ram_key_sim_II[i]]) - 
                                    cubic_simple_fit(np.array(test_param[i]), 
                                                     parameters_sim_II[i][0][0][0],
                                                     parameters_sim_II[i][0][1][0],
                                                     parameters_sim_II[i][0][2][0]), 
                    marker="D", linestyle="", color=colors[i][0], label="Before Sim II")
    axes[i][1].plot(test_param[i], np.array([tr[1] for tr in total_ram_key_sim_II[i]]) - 
                                    cubic_simple_fit(np.array(test_param[i]), 
                                                     parameters_sim_II[i][1][0][0],
                                                     parameters_sim_II[i][1][1][0],
                                                     parameters_sim_II[i][1][2][0]), 
                    marker="D", linestyle="", color=colors[i][1], label="After Sim II")
    
    axes[i][0].set_ylabel("Total RAM [GiB]")
    axes[i][1].set_ylabel("Residua [GiB]")
    
    axes[i][1].xaxis.set_ticks( [] )
    axes[i][1].xaxis.set_ticklabels( [] )
    axes[i][0].set_xlabel("Resolution")
    axes[i][0].xaxis.set_ticks( resolution[ np.argmax([len(res) for res in resolution]) ] )
    

    axes[i][0].legend()
    

axes[0][0].set_title("Vacuum")
axes[1][0].set_title("Water")

plt.suptitle("Fitted total RAM before and during simulations")

axes[1][0].yaxis.tick_right()
axes[1][1].yaxis.tick_right()
axes[1][0].yaxis.set_label_position("right")
axes[1][1].yaxis.set_label_position("right")

fig.set_size_inches([8.64, 4.8])
vs.saveplot(plot_file("FitAllKeyRAM2.png"), overwrite=True)

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

plt.plot(data[0][-1][:,0], theory[0][-1] / max(theory[0][-1]), 
         linestyle="dotted", color='red', label="Mie Theory Vacuum")
plt.plot(data[-1][-1][:,0], theory[-1][-1] / max(theory[-1][-1]), 
         linestyle="dotted", color='blue', label="Mie Theory Water")
plt.xlabel("Wavelength [nm]")
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
            
plt.plot(data[0][-1][:,0], theory[0][-1] * np.pi * (params[0][-1]["r"] * params[0][-1]["from_um_factor"] * 1e3)**2,
         linestyle="dotted", color='red', label="Mie Theory Vacuum")
plt.plot(data[-1][-1][:,0], theory[-1][-1] * np.pi * (params[-1][-1]["r"] * params[-1][-1]["from_um_factor"] * 1e3)**2,
         linestyle="dotted", color='blue', label="Mie Theory Water")
plt.xlabel("Wavelength [nm]")
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
            
plt.xlabel("Wavelength [nm]")
plt.ylabel(r"Difference in Scattering Cross Section [nm$^2$]")
plt.legend()
if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    del mng
vs.saveplot(plot_file("AllScattDiff.png"), overwrite=True)

