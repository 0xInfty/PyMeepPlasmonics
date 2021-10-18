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
import matplotlib.gridspec as gridspec
from matplotlib import use as use_backend
import os
import v_analysis as va
import v_materials as vmt
import v_plot as vp
import v_save as vs
import v_utilities as vu

english = False
trs = vu.BilingualManager(english=english)
vp.set_style()

#%% PARAMETERS <<

# Saving directories
folder = ["Scattering/AuSphere/AllWatTest/9)BoxDimensions/Courant/500to650"]
home = vs.get_home()

# Parameter for the test
test_param_string = "courant"
test_param_calculation = True
test_param_in_params = True
test_param_in_series = True
test_param_position = 0
test_param_ij_expression = "courant[i][j] * from_um_factor[i][j] * 1e6 / ( resolution[i][j] * 299.792458 )" # Leave "" by default
test_param_name = trs.choose("Minimum time division", "Mínima división temporal")
test_param_units = "as" # Leave "" by default

# Sorting and labelling data series
sorting_function = [lambda l : vu.sort_by_number(l, test_param_position)]*2
series_label = [lambda s : rf"Courant {vu.find_numbers(s)[test_param_position]:.2f}"]*2
series_must = [""] # leave "" per default
series_mustnt = ["Failed"]*2 # leave "" per default
series_column = [1]*2

# Scattering plot options
plot_title_ending = trs.choose("Au 103 nm NP in water", "NP de Au de 103 nm en agua")
series_legend = trs.choose(["Data"], ["Datos"])
series_colormaps = [plab.cm.summer.reversed()]
series_ind_colors = [["C0", "C2", "C3"]]*2
series_colors = ["k"]
series_markers = ["o","o"]
series_markersizes = [8,8]
series_linestyles = ["solid"]*2
theory_linestyles = ["dashed"]*2
plot_make_big = True
plot_for_display = False
plot_folder = "DataAnalysis/Scattering/AuSphere/AllWaterTest/Courant"

#%% LOAD DATA <<

# First some useful definitions regarding directories
def file_definer(path): return lambda s, n : os.path.join(path, s, n)
path = [os.path.join(home, fold) for fold in folder]
file = [file_definer(pa) for pa in path]

# Now look for the series of data inside of each folder
series = [[]] * len(path)
for i in range(len(folder)):
    series[i] = os.listdir(path[i])
    series[i] = vu.filter_by_string_must(series[i], series_must[i])
    if series_mustnt[i]!="": 
        series[i] = vu.filter_by_string_must(series[i], series_mustnt[i], False)
    series[i] = sorting_function[i](series[i])
del i

# Get the corresponding data
data = [[np.loadtxt(file[i](series[i][j], "Results.txt")) for j in range(len(series[i]))] for i in range(len(series))]

# Get the parameters of the simulations
params = [[vs.retrieve_footer(file[i](series[i][j], "Results.txt")) for j in range(len(series[i]))] for i in range(len(series))]
for i in range(len(series)):
    for j in range(len(series[i])):
        if not isinstance(params[i][j], dict): 
            params[i][j] = vu.fix_params_dict(params[i][j])

# Get the test parameter if no other data is needed
if test_param_in_series and not test_param_calculation:
    test_param = [[vu.find_numbers(series[i][j])[test_param_position] for j in range(len(series[i]))] for i in range(len(series))]
elif test_param_in_params and not test_param_calculation:
    test_param = [[params[i][j][test_param_string] for j in range(len(series[i]))] for i in range(len(series))]
elif not test_param_calculation:
    raise ValueError("Test parameter is nowhere to be found")

#%% LOAD PARAMETERS AND CONTROL VARIABLES <<

# Get the RAM and elapsed time data
loaded_ram = True
for i in range(len(series)):
    for j in range(len(series[i])):
        try:
            f = h5.File(file[i](series[i][j], "Resources.h5"))
            params[i][j]["used_ram"] = np.array(f["RAM"])
            params[i][j]["used_swap"] = np.array(f["SWAP"])
            params[i][j]["elapsed"] = np.array(f["ElapsedTime"])
        except FileNotFoundError:
            try:
                f = h5.File(file[i](series[i][j], "RAM.h5"))
                params[i][j]["used_ram"] = np.array(f["RAM"])
                params[i][j]["used_swap"] = np.array(f["SWAP"])
            except:
                print(f"No RAM data found for {series[i][j]}")
                loaded_ram = False
del i, j

# Extract some other parameters from the parameters dict
needs_fixing = False
from_um_factor = [[params[i][j]["from_um_factor"] for j in range(len(series[i]))] for i in range(len(series))]
resolution = [[params[i][j]["resolution"] for j in range(len(series[i]))] for i in range(len(series))]
try: courant = [[params[i][j]["courant"] for j in range(len(series[i]))] for i in range(len(series))]
except: courant = [[0.5 for j in range(len(series[i]))] for i in range(len(series))]
r = [[params[i][j]["r"] for j in range(len(series[i]))] for i in range(len(series))]
try: material = [[params[i][j]["material"] for j in range(len(series[i]))] for i in range(len(series))]
except: material = [["Au" for j in range(len(series[i]))] for i in range(len(series))]
try: paper = [[params[i][j]["paper"] for j in range(len(series[i]))] for i in range(len(series))]
except: paper = [["R" for j in range(len(series[i]))] for i in range(len(series))]
try: index = [[params[i][j]["submerged_index"] for j in range(len(series[i]))] for i in range(len(series))]
except: 
    try: index = [[params[i][j]["index"] for j in range(len(series[i]))] for i in range(len(series))]
    except: print("Index needs manual assignment"); needs_fixing = True
try: empty_width = [[params[i][j]["empty_width"] for j in range(len(series[i]))] for i in range(len(series))]
except: 
    try: empty_width = [[params[i][j]["air_width"] for j in range(len(series[i]))] for i in range(len(series))]
    except: print("Empty width needs manual assignment"); needs_fixing = True
wlen_range = [[params[i][j]["wlen_range"] for j in range(len(series[i]))] for i in range(len(series))]
pml_width = [[params[i][j]["pml_width"] for j in range(len(series[i]))] for i in range(len(series))]
source_center = [[params[i][j]["source_center"] for j in range(len(series[i]))] for i in range(len(series))]
until_after_sources = [[params[i][j]["until_after_sources"] for j in range(len(series[i]))] for i in range(len(series))]
time_factor_cell = [[params[i][j]["time_factor_cell"] for j in range(len(series[i]))] for i in range(len(series))]
second_time_factor = [[params[i][j]["second_time_factor"] for j in range(len(series[i]))] for i in range(len(series))]
try: sysname = [[params[i][j]["sysname"] for j in range(len(series[i]))] for i in range(len(series))]
except: print("Sysname needs manual assignment"); needs_fixing = True

#%% FIX PARAMETERS IF NEEDED BY TOUCHING THIS BLOCK <<

if test_param_string=="flux_r_factor":
    for i in range(len(series)):
        for j in range(len(series[i])):
            if "0.05" not in series[i][j]:
                data[i][j][:,1] = data[i][j][:,1] / (1 + vu.find_numbers(series[i][j])[test_param_position])**2

if needs_fixing:
    
    if test_param_string=="r":
        index = [[1]*len(series[0]), [1.33]*len(series[1])]
        sysname = [["SC"]*len(series[0]), ["SC"]*len(series[1])]
    else:
        empty_width = []
        for i in range(len(series)):
            empty_width.append([])
            for j in range(len(series[i])):
                try:
                    empty_width[i].append(params[i][j]["empty_width"])
                except:
                    empty_width[i].append(params[i][j]["air_width"])

#%% EXTRACT SOME MORE PARAMETERS <<

# Determine some other parameters, calculating them from others
try: cell_width = [[params[i][j]["cell_width"] for j in range(len(series[i]))] for i in range(len(series))]
except: cell_width = [[2*(pml_width[i][j] + empty_width[i][j] + r[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
empty_r_factor = [[empty_width[i][j] / r[i][j] for j in range(len(series[i]))] for i in range(len(series))]

# Guess some more parameters, calculating them from others
minor_division = [[from_um_factor[i][j] * 1e3 / resolution[i][j] for j in range(len(series[i]))] for i in range(len(series))]
mindiv_diameter_factor = [[minor_division[i][j] / (2 * 1e3 * from_um_factor[i][j] * r[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
width_points = [[int(cell_width[i][j] * resolution[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
inner_width_points = [[int( (cell_width[i][j] - 2*pml_width[i][j]) * resolution[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
grid_points = [[width_points[i][j]**3 for j in range(len(series[i]))] for i in range(len(series))]
inner_grid_points = [[inner_width_points[i][j]**3 for j in range(len(series[i]))] for i in range(len(series))]
memory_B = [[2 * 12 * grid_points[i][j] * 32 for j in range(len(series[i]))] for i in range(len(series))]

# Calculate test parameter if needed
if test_param_calculation:
    test_param = []
    for i in range(len(series)):
        test_param.append([])
        for j in range(len(series[i])):
            test_param[i].append( eval(test_param_ij_expression) )

#%% GENERAL PLOT CONFIGURATION <<

n = len(series)
m = max([len(s) for s in series])

if test_param_units != "":
    test_param_label = test_param_name + f" [{test_param_units}]"
else: test_param_label = test_param_name
    
vertical_plot = False

colors = [sc(np.linspace(0,1,len(s)+2))[2:] 
          for sc, s in zip(series_colormaps, series)]

if not os.path.isdir(os.path.join(home, plot_folder)):
    os.mkdir(os.path.join(home, plot_folder))
plot_file = lambda n : os.path.join(home, plot_folder, n)

#%% CALCULATE MIE DATA <<

theory = [[vmt.sigma_scatt_meep(r[i][j] * from_um_factor[i][j] * 1e3, # Radius [nm]
                                material[i][j], 
                                paper[i][j], 
                                data[i][j][:,0], # Wavelength [nm]
                                surrounding_index=index[i][j],
                                asEffiency=True)
                          for j in range(len(series[i]))] for i in range(len(series))]
        

wlen_plot = np.linspace(450, 650, 200)
theory_plot = [[vmt.sigma_scatt_meep(r[i][j] * from_um_factor[i][j] * 1e3, 
                                     material[i][j], 
                                     paper[i][j], 
                                     wlen_plot,
                                     surrounding_index=index[i][j],
                                     asEffiency=True)
                          for j in range(len(series[i]))] for i in range(len(series))]

#%% QUANTIFY CONTRAST WITH MIE THEORY <<

max_wlen = [[data[i][j][ np.argmax(data[i][j][:,series_column[i]]) , 0] 
             for j in range(len(series[i]))] for i in range(len(series))]

max_wlen_theory = [[data[i][j][ np.argmax(theory[i][j]) , 0] 
                    for j in range(len(series[i]))] for i in range(len(series))]

max_wlen_diff = [[max_wlen[i][j] - max_wlen_theory[i][j] 
                  for j in range(len(series[i]))] for i in range(len(series))]

msq_diff = [[np.mean(np.square(data[i][j][:,series_column[i]] - theory[i][j])) 
             for j in range(len(series[i]))] for i in range(len(series))]

# Careful! Wavelength is sorted from highest to lowest (Freq from lowest to highest)
msq_diff_left = [[np.mean(np.square(data[i][j][np.argmax(theory[i][j])+1:, series_column[i]] - 
                                    theory[i][j][np.argmax(theory[i][j])+1:])) 
                  for j in range(len(series[i]))] for i in range(len(series))]

msq_diff_right = [[np.mean(np.square(data[i][j][:np.argmax(theory[i][j]), series_column[i]] - 
                                     theory[i][j][:np.argmax(theory[i][j])])) 
                  for j in range(len(series[i]))] for i in range(len(series))]

#%% DIFFERENCE IN MAXIMUM WAVELENGTH PLOT <<

if plot_for_display: use_backend("Agg")

fig = plt.figure()
if plot_for_display: fig.dpi = 200
if not plot_for_display:
    plt.suptitle(trs.choose("Difference in scattering for ", 
                            "Diferencia en dispersión en ") + plot_title_ending)

for i in range(len(series)):
    plt.plot(test_param[i], 
             max_wlen_diff[i], 
             color=series_colors[i], 
             marker=series_markers[i], markersize=series_markersizes[i], 
             alpha=0.4, markeredgewidth=0, zorder=10)
if max(fig.axes[0].get_ylim()) >= 0 >= min(fig.axes[0].get_ylim()):
    plt.axhline(color="k", linewidth=.5, zorder=0)
plt.legend(fig.axes[0].lines, series_legend)
plt.xlabel(test_param_label)
if max([len(tp) for tp in test_param])<=4: 
    plt.xticks( test_param[ np.argmax([len(data) for data in test_param]) ] )
plt.ylabel(trs.choose("Difference in wavelength ", 
                      "Diferencia en longitud de onda ") + 
           "$\lambda_{max}^{MEEP}-\lambda_{max}^{MIE}$ [nm]")
fig.set_size_inches([6 , 4.32])
fig.tight_layout()
vs.saveplot(plot_file("MaxWLenDiff.png"), overwrite=True)

if plot_for_display: use_backend("Qt5Agg")

#%% DIFFERENCE IN SCATTERING MAXIMUM PLOT <<

alternate_axis = True

if plot_for_display: use_backend("Agg")

fig = plt.figure()
if plot_for_display: fig.dpi = 200
if not plot_for_display:
    plt.suptitle(trs.choose("Difference in scattering for ", 
                            "Diferencia en dispersión en ") + plot_title_ending)
for i in range(len(series)):
    plt.plot(test_param[i], 
             [np.max(data[i][j][:,series_column[i]]) - np.max(theory_plot[i][j]) for j in range(len(series[i]))], 
             color=series_colors[i], 
             marker=series_markers[i], markersize=series_markersizes[i], 
             alpha=0.4, markeredgewidth=0, zorder=10)
if max(fig.axes[0].get_ylim()) >= 0 >= min(fig.axes[0].get_ylim()):
    plt.axhline(color="k", linewidth=.5, zorder=0)
if plot_for_display and alternate_axis:
    fig.axes[0].yaxis.tick_right()
    fig.axes[0].yaxis.set_label_position("right")
plt.legend(fig.axes[0].lines, series_legend)
plt.xlabel(test_param_label)
if max([len(tp) for tp in test_param])<=4: 
    plt.xticks( test_param[ np.argmax([len(data) for data in test_param]) ] )
plt.ylabel(trs.choose("Difference in scattering efficiency ", 
                      "Diferencia en eficiencia de dispersión ") + 
           "$C_{max}^{MEEP}-C_{max}^{MIE}$")
plt.ylabel(trs.choose("Difference in scattering efficiency ", 
                  "Diferencia en eficiencia de dispersión ") + 
       "$C_{max}^{MEEP}-C_{max}^{MIE}$")
fig.set_size_inches([6 , 4.32])
fig.tight_layout()
vs.saveplot(plot_file("MaxScattDiff.png"), overwrite=True)

if plot_for_display: use_backend("Qt5Agg")

#%% DIFFERENCE IN SCATTERING MAXIMUM AND WAVELENGTH PLOT

if len(series)==1:
    
    with_legend = False
    
    if plot_for_display: use_backend("Agg")
    
    fig = plt.figure()
    ax = plt.subplot()
    ax2 = ax.twinx()
    if plot_for_display: fig.dpi = 200
    if not plot_for_display:
        plt.suptitle(trs.choose("Difference in scattering for ", 
                                "Diferencia en dispersión en ") + plot_title_ending)
        
    for i in range(len(series)):
        l, = ax.plot(test_param[i], 
                     max_wlen_diff[i], "o-",
                     color=series_colors[i], markersize=8, 
                     alpha=0.6, markeredgewidth=0, zorder=10)
        l2, = ax2.plot(test_param[i], 
                       [np.max(data[i][j][:,series_column[i]]) - np.max(theory_plot[i][j]) for j in range(len(series[i]))], 
                       "s-", color=series_colors[i], markersize=7, 
                       alpha=0.6, markeredgewidth=0, zorder=10)
    if with_legend or not plot_for_display:
        plt.legend([l, l2], [r"$\lambda_{max}$", r"$C_{max}^{MEEP}$"], loc="center right", framealpha=1, frameon=True)
    plt.xlabel(test_param_label)
    if max([len(tp) for tp in test_param])<=4: 
        plt.xticks( test_param[ np.argmax([len(data) for data in test_param]) ] )
    if plot_for_display:
        ax.set_ylabel(trs.choose("Difference\nin wavelength\n", 
                                 "Diferencia\nen longitud de onda\n") + 
                      "$\lambda_{max}^{MEEP}-\lambda_{max}^{MIE}$ [nm]")
        ax2.set_ylabel(trs.choose("Difference in\nscattering efficiency\n", 
                                  "Diferencia en\neficiencia de dispersión\n") + 
                       "$C_{max}^{MEEP}-C_{max}^{MIE}$")
    else:
        ax.set_ylabel(trs.choose("Difference in wavelength ", 
                                      "Diferencia en longitud de onda ") + 
                           "$\lambda_{max}^{MEEP}-\lambda_{max}^{MIE}$ [nm]")
        ax2.set_ylabel(trs.choose("Difference in scattering efficiency ", 
                                  "Diferencia en eficiencia de dispersión ") + 
                       "$C_{max}^{MEEP}-C_{max}^{MIE}$")
    if plot_for_display: fig.set_size_inches([9.84, 2.37])
    else: fig.set_size_inches([6 , 4.32])
    fig.tight_layout()
    vs.saveplot(plot_file("MaxScattDiff.png"), overwrite=True)
    
    if plot_for_display: use_backend("Qt5Agg")

#%% MEAN RESIDUAL LEFT AND RIGHT PLOT <<

if plot_for_display: use_backend("Agg")

fig = plt.figure()
if plot_for_display: fig.dpi = 200
if not plot_for_display:
    plt.suptitle(trs.choose("Difference in scattering for ", 
                            "Diferencia en dispersión en ") + plot_title_ending)
lines = []
medium_lines = []
lines_legend = []
medium_lines_legend = []
for i in range(len(test_param)):
    l1, = plt.plot(test_param[i], msq_diff_left[i], 'o-', 
                   color=series_colors[i],
                   markerfacecolor="mediumorchid", linewidth=1,
                   marker=series_markers[i], markersize=series_markersizes[i], 
                   alpha=0.6, markeredgewidth=0, zorder=10)
    if True:#i == 0:
        lines.append(l1)
        lines_legend.append(series_legend[i]+trs.choose(" left", " izquierda"))
    l2, = plt.plot(test_param[i], msq_diff_right[i], 'o-', 
                   color=series_colors[i],
                   markerfacecolor="red", linewidth=1,
                   marker=series_markers[i], markersize=series_markersizes[i],
                   alpha=0.6, markeredgewidth=0, zorder=10)
    if True:#i == 0:
        lines.append(l2)
        lines_legend.append(series_legend[i]+trs.choose(" right", " derecha"))
    l3, = plt.plot(test_param[i], msq_diff[i], 'o-', linewidth=1,
                   color=series_colors[i], markerfacecolor="k",
                   marker=series_markers[i], markersize=series_markersizes[i],
                   alpha=0.6, markeredgewidth=0, zorder=10)
    if True:#i == 0:
        lines.append(l3)
        lines_legend.append(series_legend[i]+trs.choose(" all", " completo"))
    medium_lines.append(l3)
    medium_lines_legend.append(series_legend[i])
if max(fig.axes[0].get_ylim()) >= 0 >= min(fig.axes[0].get_ylim()):
    plt.axhline(color="k", linewidth=.5, zorder=0)
# plt.legend(lines, lines_legend)
first_legend = plt.legend(lines, lines_legend, ncol=len(series))
# second_legend = plt.legend(medium_lines, medium_lines_legend, loc="center left")
# plt.gca().add_artist(first_legend)
plt.xlabel(test_param_label)
plt.ylabel(trs.choose("Mean squared difference ",
                      "Diferencia cuadrática media ") 
           + "MSD( $C^{MEEP} - C^{MIE}$ )")
fig.set_size_inches([6 , 4.32])
fig.tight_layout()
vs.saveplot(plot_file("QuaDiff.png"), overwrite=True)

if plot_for_display: use_backend("Qt5Agg")

#%% QUANTIFICATION OF DIFFERENCE PLOT ==

if len(series)==1:
    
    with_legend = False
    needs_plot_fixing = False
    
    if plot_for_display: use_backend("Agg")
    
    fig = plt.figure()
    if plot_for_display: fig.dpi = 200
    if not plot_for_display:
        plt.suptitle(trs.choose("Difference in scattering for ", 
                                "Diferencia en dispersión en ") + plot_title_ending)
    
    ax, axmsq = fig.subplots(nrows=2, sharex=True, gridspec_kw={"hspace":0})
    ax2 = ax.twinx()
    
    lines = []
    medium_lines = []
    lines_legend = []
    medium_lines_legend = []
    for i in range(len(series)):
        l, = ax.plot(test_param[i], 
                     max_wlen_diff[i], "o-",
                     color=series_colors[i], markersize=8, 
                     alpha=0.6, markeredgewidth=0, zorder=10)
        l2, = ax2.plot(test_param[i], 
                       [np.max(data[i][j][:,series_column[i]]) - np.max(theory_plot[i][j]) for j in range(len(series[i]))], 
                       "s-", color=series_colors[i], markersize=7, 
                       alpha=0.6, markeredgewidth=0, zorder=10)
        lm1, = axmsq.plot(test_param[i], msq_diff_left[i], 'o-', 
                          color=series_colors[i],
                          markerfacecolor="mediumorchid", linewidth=1,
                          marker=series_markers[i], markersize=series_markersizes[i], 
                          alpha=0.6, markeredgewidth=0, zorder=10)
        lines.append(lm1)
        lines_legend.append(series_legend[i]+trs.choose(" left", " izquierda"))
        lm2, = axmsq.plot(test_param[i], msq_diff_right[i], 'o-', 
                          color=series_colors[i],
                          markerfacecolor="red", linewidth=1,
                          marker=series_markers[i], markersize=series_markersizes[i],
                          alpha=0.6, markeredgewidth=0, zorder=10)
        lines.append(lm2)
        lines_legend.append(series_legend[i]+trs.choose(" right", " derecha"))
        lm3, = axmsq.plot(test_param[i], msq_diff[i], 'o-', linewidth=1,
                       color=series_colors[i], markerfacecolor="k",
                       marker=series_markers[i], markersize=series_markersizes[i],
                       alpha=0.6, markeredgewidth=0, zorder=10)
        lines.append(lm3)
        lines_legend.append(series_legend[i]+trs.choose(" all", " completo"))
        
    if with_legend or not plot_for_display:
        ax.legend([l, l2], [r"$\lambda_{max}$", r"$C_{max}^{MEEP}$"], loc="center right", framealpha=1, frameon=True)
        axmsq.legend(lines, lines_legend, ncol=len(series), framealpha=1, frameon=True)
    
    axmsq.set_xlabel(test_param_label)
    if max([len(tp) for tp in test_param])<=4: 
        axmsq.xticks( test_param[ np.argmax([len(data) for data in test_param]) ] )
        
    ax.set_ylabel(trs.choose("Difference in\nwavelength\n", 
                             "Diferencia en\nlongitud de onda\n") + 
                  "$\lambda_{max}^{MEEP}-\lambda_{max}^{MIE}$ [nm]")
    ax2.set_ylabel(trs.choose("Difference in\nscattering efficiency\n", 
                              "Diferencia en\neficiencia de dispersión\n") + 
                   "$C_{max}^{MEEP}-C_{max}^{MIE}$")
    axmsq.set_ylabel(trs.choose("Mean squared\ndifference\n",
                                "Diferencia\ncuadrática media\n")
                     + "MSD( $C^{MEEP} - C^{MIE}$ )")
    
    if needs_plot_fixing:
        if test_param_string=='max_wlen_range':
            ax2.set_ylim(-.47, -.46)
        elif False:
            ax2.set_ylim(-.5, -.4)
            
    if plot_for_display: fig.set_size_inches([9.84, 4.01])
    else: fig.set_size_inches([8.49, 4.55])
    fig.tight_layout()
    vs.saveplot(plot_file("Quantified.png"), overwrite=True)
    
    if plot_for_display: use_backend("Qt5Agg")

#%% DIAMETER DISCRETIZATION

if plot_for_display: use_backend("Agg")

fig = plt.figure()
if plot_for_display: fig.dpi = 200
if not plot_for_display:
    plt.suptitle(trs.choose("Difference in scattering for ", 
                            "Diferencia en dispersión en ") + plot_title_ending)
for i in range(len(series)):
    plt.plot(test_param[i], 
             mindiv_diameter_factor[i], 
             color=series_colors[i], 
             marker=series_markers[i], markersize=series_markersizes[i], 
             alpha=0.4, markeredgewidth=0)
if plot_for_display:
    fig.axes[0].yaxis.tick_right()
    fig.axes[0].yaxis.set_label_position("right")
plt.legend(series_legend)
plt.xlabel(test_param_label)
if max([len(tp) for tp in test_param])<=4: 
    plt.xticks( test_param[ np.argmax([len(data) for data in test_param]) ] )
plt.ylabel(trs.choose(r"Minimum spatial division $\Delta r$ [$d$]",
                      r"Mínima división espacial $\Delta r$ [$d$]"))
fig.set_size_inches([6 , 4.32])
fig.tight_layout()
vs.saveplot(plot_file("MinDivDiam.png"), overwrite=True)

if plot_for_display: use_backend("Qt5Agg")

#%% RELATIONS WITH DIAMETER DISCRETIZATION

fig = plt.figure()
plt.suptitle(trs.choose("Difference in scattering for ", 
                        "Diferencia en dispersión en ") + plot_title_ending)
plt.axhline(color="k", linewidth=.5)
for i in range(len(series)):
    plt.plot(mindiv_diameter_factor[i], max_wlen_diff[i], color=series_colors[i], 
             marker=series_markers[i], markersize=series_markersizes[i], 
             alpha=0.4, markeredgewidth=0)
plt.legend(fig.axes[0].lines[1:], series_legend)
plt.xlabel(trs.choose(r"Minimum spatial division $\Delta r$ [$d$]",
                      r"Mínima división espacial $\Delta r$ [$d$]"))
plt.ylabel(trs.choose("Difference in wavelength ", 
                      "Diferencia en longitud de onda ") + 
           "$\lambda_{max}^{MEEP}-\lambda_{max}^{MIE}$ [nm]")
fig.set_size_inches([6 , 4.32])
fig.tight_layout()
vs.saveplot(plot_file("WLenDiffFactor.png"), overwrite=True)

fig = plt.figure()
plt.suptitle(trs.choose("Difference in scattering for ", 
                        "Diferencia en dispersión en ") + plot_title_ending)
plt.axhline(color="k", linewidth=.5)
for i in range(len(series)):
    plt.plot(mindiv_diameter_factor[i], 
             [np.max(data[i][j][:,series_column[i]]) - np.max(theory_plot[i][j]) for j in range(len(series[i]))], 
             color=series_colors[i], 
             marker=series_markers[i], markersize=series_markersizes[i], 
             alpha=0.4, markeredgewidth=0)
plt.legend(fig.axes[0].lines[1:], series_legend)
plt.xlabel(trs.choose(r"Minimum spatial division $\Delta r$ [$d$]",
                      r"Mínima división espacial $\Delta r$ [$d$]"))
plt.ylabel(trs.choose("Difference in scattering efficiency ", 
                      "Diferencia en eficiencia de dispersión ") + 
           "$C_{max}^{MEEP}-C_{max}^{MIE}$")
fig.set_size_inches([6 , 4.32])
fig.tight_layout()
vs.saveplot(plot_file("MaxScattDiffFactor.png"), overwrite=True)

fig, axes = plt.subplots(2, sharex=True, gridspec_kw={"hspace":0})
plt.suptitle(trs.choose("Difference in scattering for ", 
                        "Diferencia en dispersión en ") + plot_title_ending)
if plot_for_display: fig.dpi = 200

for i in range(len(axes)):
    axes[i].axhline(color="k", linewidth=.5)
    axes[i].plot(mindiv_diameter_factor[i], msq_diff_right[i], "red", 
                 marker=series_markers[i], markersize=series_markersizes[i], 
                 alpha=0.4, markeredgewidth=0)
    axes[i].plot(mindiv_diameter_factor[i], msq_diff_left[i], "darkviolet", 
                 marker=series_markers[i], markersize=series_markersizes[i], 
                 alpha=0.4, markeredgewidth=0)
    axes[i].plot(mindiv_diameter_factor[i], msq_diff[i], "k", 
                 marker=series_markers[i], markersize=series_markersizes[i], 
                 alpha=0.4, markeredgewidth=0)
    axes[i].legend(axes[i].lines[1:],
                   [series_legend[i] + trs.choose(" left", " izquierda"),
                    series_legend[i] + trs.choose(" right", " derecha"),
                    series_legend[i] + trs.choose(" all", " completo")])
    axes[i].set_ylabel(trs.choose("Mean squared difference\n",
                                  "Diferencia cuadrática media\n") 
                       + "MSD( $C^{MEEP} - C^{MIE}$ )")
axes[1].yaxis.tick_right()
axes[1].yaxis.set_label_position("right")
plt.xlabel(trs.choose(r"Minimum spatial division $\Delta r$ [$d$]",
                      r"Mínima división espacial $\Delta r$ [$d$]"))
plt.tight_layout()
vs.saveplot(plot_file("QuaDiffFactor.png"), overwrite=True)

#%% GET ELAPSED TIME COMPARED <<

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
        try:
            if p["split_chunks_evenly"]:
                total_elapsed_time[-1].append(sum(p["elapsed"]))
            else:
                if len(e)==5:
                    total_elapsed_time[-1].append(sum(p["elapsed"]) + e[2])
                else:
                    total_elapsed_time[-1].append(sum(p["elapsed"]) + e[0])
        except:
            if len(e)==5:
                total_elapsed_time[-1].append(sum(p["elapsed"]) + e[2])
            else:
                total_elapsed_time[-1].append(sum(p["elapsed"]) + e[0])
                
#%% PLOT ELAPSED TIME IN DIFFERENT STAGES <<

these_markers = ["o", "o", "D", "D"]

if len(series)>1: these_colors = [*["darkgrey", "k"]*2]
else: these_colors = [*["k"]*2]
plt.figure()
plt.suptitle(trs.choose("Elapsed total time for ", 
                        "Tiempo transcurrido total en ") + plot_title_ending)
for i in range(len(series)):
    plt.plot(test_param[i], total_elapsed_time[i],
             color=these_colors[i], marker=these_markers[i],
             markersize=series_markersizes[i])
plt.legend(series_legend)
plt.xlabel(test_param_label)
if max([len(tp) for tp in test_param])<=4: 
    plt.xticks( test_param[ np.argmax([len(data) for data in test_param]) ] )
plt.ylabel(trs.choose("Elapsed time [s]", "Tiempo transcurrido [s]"))
vs.saveplot(plot_file("ComparedTotTime.png"), overwrite=True)
        
these_colors = ["r", "maroon", "darkorange", "orangered"]
fig = plt.figure()
plt.suptitle(trs.choose("Elapsed time for simulation of ", 
                        "Tiempo transcurrido para simulación de ") + plot_title_ending)
for i in range(len(series)):
    plt.plot(first_test_param[i], first_sim_time[i], 
             'D-', color=these_colors[i], label=series_legend[i] + " Sim I")
    plt.plot(second_test_param[i], second_sim_time[i], 
             's-', color=these_colors[i], label=series_legend[i] + " Sim I")
plt.xlabel(test_param_label)
if max([len(tp) for tp in test_param])<=4: 
    plt.xticks( test_param[ np.argmax([len(data) for data in test_param]) ] )
plt.ylabel(trs.choose("Elapsed time in simulations [s]", 
                      "Tiempo transcurrido en simulaciones [s]"))
box = fig.axes[0].get_position()
box.y0 = box.y0 + .25 * (box.y1 - box.y0)
box.y1 = box.y1 + .10 * (box.y1 - box.y0)
box.x1 = box.x1 + .10 * (box.x1 - box.x0)
fig.axes[0].set_position(box)
leg = plt.legend(ncol=2, bbox_to_anchor=(.5, -.47), loc="lower center", frameon=False)
plt.savefig(plot_file("ComparedSimTime.png"), bbox_inches='tight')

these_colors = ["b", "navy", "cyan", "deepskyblue"]
fig = plt.figure()
plt.suptitle(trs.choose("Elapsed time for building of ", 
                        "Tiempo transcurrido para construcción de ") + plot_title_ending)
for i in range(len(series)):
    plt.plot(first_test_param[i], first_build_time[i], 
             'D-', color=these_colors[i], label=series_legend[i] + " Sim I")
    plt.plot(second_test_param[i], second_build_time[i], 
             's-', color=these_colors[i], label=series_legend[i] + " Sim I")
plt.xlabel(test_param_label)
if max([len(tp) for tp in test_param])<=4: 
    plt.xticks( test_param[ np.argmax([len(data) for data in test_param]) ] )
plt.ylabel(trs.choose("Elapsed time in building [s]", 
                      "Tiempo transcurrido en construcción [s]"))
box = fig.axes[0].get_position()
box.y0 = box.y0 + .25 * (box.y1 - box.y0)
box.y1 = box.y1 + .10 * (box.y1 - box.y0)
box.x1 = box.x1 + .10 * (box.x1 - box.x0)
fig.axes[0].set_position(box)
leg = plt.legend(ncol=2, bbox_to_anchor=(.5, -.47), loc="lower center", frameon=False)
plt.savefig(plot_file("ComparedBuildTime.png"), bbox_inches='tight')

these_colors = ["m", "darkmagenta", "blueviolet", "indigo"]
fig = plt.figure()
plt.suptitle(trs.choose("Elapsed time for loading flux of ", 
                        "Tiempo transcurrido para cargar flujo de ") + plot_title_ending)
for i in range(len(series)):
    plt.plot(second_test_param[i], second_flux_time[i], 
             's-', color=these_colors[i], label=series_legend[i] + " Sim II")
plt.xlabel(test_param_label)
if max([len(tp) for tp in test_param])<=4: 
    plt.xticks( test_param[ np.argmax([len(data) for data in test_param]) ] )
plt.ylabel(trs.choose("Elapsed time in loading flux [s]", 
                      "Tiempo transcurrido en carga de flujo [s]"))
box = fig.axes[0].get_position()
box.y0 = box.y0 + .15 * (box.y1 - box.y0)
box.y1 = box.y1 + .10 * (box.y1 - box.y0)
box.x1 = box.x1 + .10 * (box.x1 - box.x0)
fig.axes[0].set_position(box)
leg = plt.legend(ncol=2, bbox_to_anchor=(.5, -.3), loc="lower center", frameon=False)
plt.savefig(plot_file("ComparedLoadTime.png"), bbox_inches='tight')

#%% ALL RAM MEMORY <<

if loaded_ram:
    
    reference = trs.choose(["Modules",
                            "Parameters",
                            "Initial (Sim I)",
                            "mp.Simulation (Sim I)",
                            "Flux (Sim I)",
                            "sim.init_sim() (Sim I)",
                            "Beginnings (Sim I)",
                            "Middle (Sim I)",
                            "Ending (Sim I)",
                            "Initial (Sim II)",
                            r"mp.Simulation (Sim II)",
                            "Flux (Sim II)",
                            "sim.init_sim() (Sim II)",
                            "load_midflux() (Sim II)",
                            "Minus flux (Sim II)",
                            "Beginnings (Sim II)",
                            "Middle (Sim II)",
                            "Ending (Sim II)"],
                           ["Módulos",
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
                            "Final (Sim II)"])
    
    total_ram = [[np.array([sum(ur) for ur in p["used_ram"]]) for p in par] for par in params]
    max_ram = [[p["used_ram"][:, np.argmax([sum(ur) for ur in p["used_ram"].T]) ] for p in par] for par in params]
    min_ram = [[p["used_ram"][:, np.argmin([sum(ur) for ur in p["used_ram"].T]) ] for p in par] for par in params]
    mean_ram = [[np.array([np.mean(ur) for ur in p["used_ram"]]) for p in par] for par in params]
    
    total_ram = [[ tr / (1024)**2 for tr in tram] for tram in total_ram]
    max_ram = [[ tr / (1024)**2 for tr in tram] for tram in max_ram]
    min_ram = [[ tr / (1024)**2 for tr in tram] for tram in min_ram]
    mean_ram = [[ tr / (1024)**2 for tr in tram] for tram in mean_ram]

#%% RAM PER SUBPROCESS PLOT

if loaded_ram:
    
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
    patches.append( plt.axvspan(*trs.choose(["Initial (Sim I)", "Beginnings (Sim I)"],
                                            ["Inicial (Sim I)", "Principio (Sim I)"]), 
                                alpha=0.15, color='grey', zorder=1) )
    patches.append( plt.axvspan(*trs.choose(["Beginnings (Sim I)", "Ending (Sim I)"], 
                                            ["Principio (Sim I)", "Final (Sim I)"]),
                                alpha=0.3, color='grey', zorder=1) )
    patches.append( plt.axvspan(*trs.choose(["Initial (Sim II)", "Beginnings (Sim II)"],
                                            ["Inicial (Sim II)", "Principio (Sim II)"]), 
                                alpha=0.1, color='gold', zorder=1) )
    patches.append( plt.axvspan(*trs.choose(["Beginnings (Sim II)", "Ending (Sim II)"], 
                                            ["Principio (Sim II)", "Final (Sim II)"]),
                                alpha=0.2, color='gold', zorder=1) )
    patches_legend = trs.choose(["Configuration Sim I", "Running Sim I",
                                 "Configuration Sim II", "Running Sim II"],
                                ["Configuración Sim I", "Corrida Sim I",
                                 "Configuración Sim II", "Corrida Sim II"])
            
    plt.xticks(rotation=-30, ha="left")
    plt.grid(True)
    plt.ylabel(trs.choose("RAM Memory Per Subprocess [GiB]",
                          "Memoria RAM por subproceso [GiB]"))
    
    plt.legend(lines, lines_legend)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.tight_layout()
    
    # first_legend = plt.legend(lines, lines_legend)
    first_legend = plt.legend(lines, lines_legend, ncol=2)
    second_legend = plt.legend(patches, patches_legend, loc="center left")
    plt.gca().add_artist(first_legend)
    
    plt.savefig(plot_file("AllRAM.png"), bbox_inches='tight')

#%% RAM TOTAL PLOT

if loaded_ram:
    
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
    patches.append( plt.axvspan(reference.index(trs.choose("Initial (Sim I)",
                                                           "Inicial (Sim I)")), 
                                reference.index(trs.choose("Beginnings (Sim I)",
                                                           "Principio (Sim I)")), 
                                alpha=0.15, color='grey', zorder=1) )
    patches.append( plt.axvspan(reference.index(trs.choose("Beginnings (Sim I)",
                                                           "Principio (Sim I)")), 
                                reference.index(trs.choose("Ending (Sim I)",
                                                           "Final (Sim I)")), 
                                alpha=0.3, color='grey', zorder=1) )
    patches.append( plt.axvspan(reference.index(trs.choose("Initial (Sim II)",
                                                           "Inicial (Sim II)")), 
                                reference.index(trs.choose("Beginnings (Sim II)",
                                                           "Principio (Sim II)")), 
                                alpha=0.1, color='gold', zorder=1) )
    patches.append( plt.axvspan(reference.index(trs.choose("Beginnings (Sim II)",
                                                           "Principio (Sim II)")), 
                                reference.index(trs.choose("Ending (Sim II)",
                                                           "Final (Sim II)")), 
                                alpha=0.2, color='gold', zorder=1) )
    patches_legend = trs.choose(["Configuration Sim I", "Running Sim I",
                                 "Configuration Sim II", "Running Sim II"],
                                ["Configuración Sim I", "Corrida Sim I",
                                 "Configuración Sim II", "Corrida Sim II"])
    
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
    plt.ylabel(trs.choose("Total RAM Memory [GiB]", "Memoria RAM Total [GiB]"))
    
    plt.legend(lines, lines_legend)
    # first_legend = plt.legend(lines, lines_legend)
    first_legend = plt.legend(lines, lines_legend, ncol=2)
    second_legend = plt.legend(patches, patches_legend, loc="center left")
    plt.gca().add_artist(first_legend)
    
    second_ax = plt.twinx()
    second_ax.set_ylabel(trs.choose("Total RAM Memory [%]", "Memoria RAM Total [%]"))
    second_ax.set_ylim(0, 100)
    second_ax.set_zorder(0)
    
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.tight_layout()
    
    plt.savefig(plot_file("AllTotalRAM.png"), bbox_inches='tight')

#%% COMPARE RAM IN SIMULATIONS I AND II <<

if loaded_ram:
    
    crossed_reference = trs.choose(["Initial",
                                    "mp.Simulation",
                                    "Flux",
                                    "sim.init_sim()",
                                    "load_midflux()",
                                    "Minus flux",
                                    "Beginnings",
                                    "Middle",
                                    "Ending"],
                                   ["Inicial",
                                    "mp.Simulation",
                                    "Flujo",
                                    "sim.init_sim()",
                                    "load_midflux()",
                                    "Flujo negado",
                                    "Principio",
                                    "Mitad",
                                    "Final"])
    
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

#%% RAM SIM I AND II PLOT

if loaded_ram:
    
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
    patches.append( plt.axvspan(*trs.choose(["Initial", "Beginnings"],
                                            ["Inicial", "Principio"]), 
                                alpha=0.15, color='peru', zorder=1) )
    patches.append( plt.axvspan(*trs.choose(["Beginnings", "Ending"],
                                            ["Principio", "Final"]), 
                                alpha=0.3, color='peru', zorder=1) )
    patches_legend = trs.choose(["Configuration", "Running"],
                                ["Configuración", "Corrida"])
            
    plt.xticks(rotation=-30, ha="left")
    plt.grid(True)
    plt.ylabel(trs.choose("Dedicated RAM Memory Per Subprocess [GiB]",
                          "Memoria RAM dedicada por subproceso [GiB]"))
    
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

#%% KEY POINTS OF COMPARISON FOR RAM <<

if loaded_ram:
    
    key_reference = trs.choose(["Beginnings", "Ending"], ["Principio", "Mitad"])
    key_mask = trs.choose(["Before", "During"], ["Antes", "Durante"])
    
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

#%% PLOT TOTAL RAM PER STAGE PLOT

if loaded_ram:
    
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
    plt.ylabel(trs.choose("Total RAM Memory [GiB]", "Memoria RAM Total [GiB]"))
    
    first_legend = plt.legend(lines, lines_legend, loc="upper left", ncol=2)
    second_legend = plt.legend(second_lines, second_lines_legend, loc="center left")
    plt.gca().add_artist(first_legend)
    
    second_ax = plt.twinx()
    second_ax.set_ylabel(trs.choose("Total RAM Memory [%]", "Memoria RAM Total [%]"))
    second_ax.set_ylim(0, 100)
    second_ax.set_zorder(0)
    
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.tight_layout()
    
    plt.savefig(plot_file("AllKeyRAMStage.png"), bbox_inches='tight')

#%% PLOT TOTAL RAM PER SIMULATION PLOT

if loaded_ram:
    
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
    
    axes[0].set_title(trs.choose("Simulation I", "Simulación I"))
    axes[1].set_title(trs.choose("Simulation II", "Simulación II"))
    
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
    axes[0].set_ylabel(trs.choose("Total RAM Memory [GiB]", "Memoria RAM Total [GiB]"))
    
    # axes[0].legend(lines, lines_legend, loc="upper left")
    axes[0].legend(lines, lines_legend, loc="upper left", ncol=2)
    
    second_ax = axes[1].twinx()
    second_ax.set_ylabel(trs.choose("Total RAM Memory [%]", "Memoria RAM Total [%]"))
    second_ax.set_ylim(0, 100)
    second_ax.set_zorder(0)
    
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.tight_layout()
    
    plt.savefig(plot_file("AllKeyRAMSim.png"), bbox_inches='tight')

#%% FIT TOTAL RAM VS RESOLUTION <<

if loaded_ram and test_param_string=="resolution":
    
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

#%% PLOT TOTAL RAM FITS VS RESOLUTION <<
 
if loaded_ram and test_param_string=="resolution":
       
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
                        marker="o", linestyle="", color=colors[i][0], 
                        label=trs.choose("Before Sim I", "Antes de Sim I"))
        axes[i][0].plot(test_param[i], [tr[1] for tr in total_ram_key_sim_I[i]], 
                        marker="o", linestyle="", color=colors[i][1], 
                        label=trs.choose("During Sim I", "Durante Sim I"))
        axes[i][0].plot(test_param[i], [tr[0] for tr in total_ram_key_sim_II[i]], 
                        marker="D", linestyle="", color=colors[i][0], 
                        label=trs.choose("Before Sim II", "Antes de Sim II"))
        axes[i][0].plot(test_param[i], [tr[1] for tr in total_ram_key_sim_II[i]], 
                        marker="D", linestyle="", color=colors[i][1], 
                        label=trs.choose("During Sim II", "Durante Sim II"))
        
        # Residuum
        axes[i][1].plot(test_param[i], np.array([tr[0] for tr in total_ram_key_sim_I[i]]) - 
                                        cubic_simple_fit(np.array(test_param[i]), 
                                                         parameters_sim_I[i][0][0][0],
                                                         parameters_sim_I[i][0][1][0],
                                                         parameters_sim_I[i][0][2][0]), 
                        marker="o", linestyle="", color=colors[i][0], 
                        label=trs.choose("Before Sim I", "Antes de Sim I"))
        axes[i][1].plot(test_param[i], np.array([tr[1] for tr in total_ram_key_sim_I[i]]) - 
                                        cubic_simple_fit(np.array(test_param[i]), 
                                                         parameters_sim_I[i][1][0][0],
                                                         parameters_sim_I[i][1][1][0],
                                                         parameters_sim_I[i][1][2][0]), 
                        marker="o", linestyle="", color=colors[i][1], 
                        label=trs.choose("During Sim I", "Durante Sim I"))
        axes[i][1].plot(test_param[i], np.array([tr[0] for tr in total_ram_key_sim_II[i]]) - 
                                        cubic_simple_fit(np.array(test_param[i]), 
                                                         parameters_sim_II[i][0][0][0],
                                                         parameters_sim_II[i][0][1][0],
                                                         parameters_sim_II[i][0][2][0]), 
                        marker="D", linestyle="", color=colors[i][0], 
                        label=trs.choose("Before Sim II", "Antes de Sim II"))
        axes[i][1].plot(test_param[i], np.array([tr[1] for tr in total_ram_key_sim_II[i]]) - 
                                        cubic_simple_fit(np.array(test_param[i]), 
                                                         parameters_sim_II[i][1][0][0],
                                                         parameters_sim_II[i][1][1][0],
                                                         parameters_sim_II[i][1][2][0]), 
                        marker="D", linestyle="", color=colors[i][1], 
                        label=trs.choose("During Sim II", "Durante Sim II"))
        
        axes[i][0].set_ylabel(trs.choose("Total RAM [GiB]", "RAM Total [GiB]"))
        axes[i][1].set_ylabel(trs.choose("Residua [GiB]", "Residuos [GiB]"))
        
        axes[i][1].xaxis.set_ticks( [] )
        axes[i][1].xaxis.set_ticklabels( [] )
        axes[i][0].set_xlabel(trs.choose("Resolution", "Resolución"))
        axes[i][0].xaxis.set_ticks( resolution[ np.argmax([len(res) for res in resolution]) ] )
    
        axes[i][0].legend()
    
    axes[0][0].set_title(trs.choose("Vacuum", "Vacío"))
    axes[1][0].set_title(trs.choose("Water", "Agua"))
    
    plt.suptitle(trs.choose("Fitted total RAM before and during simulations",
                            "RAM total ajustada antes y durante la simulación"))
    
    axes[1][0].yaxis.tick_right()
    axes[1][1].yaxis.tick_right()
    axes[1][0].yaxis.set_label_position("right")
    axes[1][1].yaxis.set_label_position("right")
    
    fig.set_size_inches([8.64, 4.8])
    vs.saveplot(plot_file("FitAllKeyRAM2.png"), overwrite=True)

#%% PLOT SCATTERING NORMALIZED <<

fig = plt.figure()
plt.title(trs.choose("Normalized Scattering for ",
                     "Dispersión normalizada en ") + plot_title_ending)

series_lines = []
inside_series_lines = []
for i in range(len(series)):
    for j in range(len(series[i])):
        l, = plt.plot(data[i][j][:,0], 
                      data[i][j][:,series_column[i]] / np.max(data[i][j][:, series_column[i]]), 
                      linestyle=series_linestyles[i], color=colors[i][j])
                      # label="MEEP " + series_legend[i] + series_label[i](series[i][j]))
        lt, = plt.plot(data[i][j][:,0], theory[i][j] / np.max(theory[i][j]), 
                       linestyle=theory_linestyles[i], color=colors[i][j])
                       # label="Mie " + series_legend[i] + series_label[i](series[i][j]))
        if i==len(series)-1:
            l.set_label(series_label[i](series[i][j]))
        else:
            l.set_label("")
        if j==int(len(series[i])/2):
            series_lines = [*series_lines, l, lt]
        inside_series_lines.append(l)

plt.xlabel(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
plt.ylabel(trs.choose("Normalized Scattering Cross Section " + 
                      r"$\sigma_{scatt}/\sigma_{scatt}^{max}$",
                      "Sección eficaz de dispersión normalizada " +
                      r"$\sigma_{disp}/\sigma_{scatt}^{disp}$"))

box = fig.axes[0].get_position()
width = box.x1 - box.x0
box.x0 = box.x0 - .07 * width
box.x1 = box.x1 - .15 * width
fig.axes[0].set_position(box)

first_legend = fig.axes[0].legend(series_lines, 
                                  [s1 + s2 for s1, s2 in zip(
                                      sum([[series_legend[i]]*2 for i in range(len(series))], []), 
                                      [" MEEP", " Mie"]*len(series))],
                                   loc="center", frameon=False, 
                                   bbox_to_anchor=(1.15, .2),
                                   bbox_transform=fig.axes[0].transAxes)
second_legend = fig.axes[0].legend(
    inside_series_lines, 
    [l.get_label() for l in inside_series_lines],
    bbox_to_anchor=(1.15, .6), columnspacing=-.5,
    loc="center", ncol=len(series), frameon=False)
fig.axes[0].add_artist(first_legend)
# leg = plt.legend(bbox_to_anchor=(1.40, .5), loc="center right", bbox_transform=fig.axes[0].transAxes)

if plot_make_big: fig.set_size_inches([10.6,  5.5])
vs.saveplot(plot_file("AllScattNorm.png"), overwrite=True)

#%% PLOT SCATTERING EFFICIENCY ==

if plot_for_display: use_backend("Agg")

fig = plt.figure()
plt.title(trs.choose("Scattering Efficiency for ",
                     "Eficiencia de dispersión en ") + plot_title_ending)
if plot_for_display: fig.dpi=200

series_lines = []
inside_series_lines = []
for i in range(len(series)):
    for j in range(len(series[i])):
        l, = plt.plot(data[i][j][:,0], 
                      data[i][j][:,series_column[i]], 
                      linestyle=series_linestyles[i], color=colors[i][j])
                      # label="MEEP " + series_legend[i] + series_label[i](series[i][j]))
        lt, = plt.plot(data[i][j][:,0], theory[i][j],
                       linestyle=theory_linestyles[i], color=colors[i][j])
                       # label="Mie " + series_legend[i] + series_label[i](series[i][j]))
        if i==len(series)-1:
            l.set_label(series_label[i](series[i][j]))
        else:
            l.set_label("")
        if j==int(len(series[i])/2):
            series_lines = [*series_lines, l, lt]
        inside_series_lines.append(l)

plt.xlabel(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
plt.ylabel(trs.choose(r"Scattering Efficiency $C_{scatt}$", 
                      "Eficiencia de dispersión $C_{disp}$"))

box = fig.axes[0].get_position()
width = box.x1 - box.x0
box.x0 = box.x0 - .07 * width
box.x1 = box.x1 - .15 * width
fig.axes[0].set_position(box)

first_legend = fig.axes[0].legend(series_lines, 
                                  [s1 + s2 for s1, s2 in zip(
                                      sum([[series_legend[i]]*2 for i in range(len(series))], []), 
                                      [" MEEP", " Mie"]*len(series))],
                                   loc="center", frameon=False, 
                                   bbox_to_anchor=(1.15, .2),
                                   bbox_transform=fig.axes[0].transAxes)
second_legend = fig.axes[0].legend(
    inside_series_lines, 
    [l.get_label() for l in inside_series_lines],
    bbox_to_anchor=(1.15, .65), columnspacing=-.5,
    loc="center", ncol=len(series), frameon=False)
fig.axes[0].add_artist(first_legend)
# leg = plt.legend(bbox_to_anchor=(1.40, .5), loc="center right", bbox_transform=fig.axes[0].transAxes)

if plot_make_big: fig.set_size_inches([10.6,  5.5])
if plot_for_display: fig.set_size_inches([10.6 ,  4.12])
vs.saveplot(plot_file("AllScattEff.png"), overwrite=True)

if plot_for_display: use_backend("Qt5Agg")

#%% PLOT SCATTERING IN UNITS

fig = plt.figure()
plt.title(trs.choose("Scattering for ",
                     "Dispersión en ") + plot_title_ending)

series_lines = []
inside_series_lines = []
for i in range(len(series)):
    for j in range(len(series[i])):
        l, = plt.plot(data[i][j][:,0], 
                      data[i][j][:,series_column[i]] * np.pi * (r[i][j] * from_um_factor[i][j] * 1e3)**2, 
                      linestyle=series_linestyles[i], color=colors[i][j])
                      # label="MEEP " + series_legend[i] + series_label[i](series[i][j]))
        lt, = plt.plot(data[i][j][:,0], theory[i][j] * np.pi * (r[i][j] * from_um_factor[i][j] * 1e3)**2,
                       linestyle=theory_linestyles[i], color=colors[i][j])
                       # label="Mie " + series_legend[i] + series_label[i](series[i][j]))
        if i==len(series)-1:
            l.set_label(series_label[i](series[i][j]))
        else:
            l.set_label("")
        if j==int(len(series[i])/2):
            series_lines = [*series_lines, l, lt]
        inside_series_lines.append(l)

plt.xlabel(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
plt.ylabel(trs.choose(r"Scattering Cross Section $\sigma_{scatt}$ [nm$^2$]",
                      r"Sección eficaz de dispersión $\sigma_{disp}$ [nm$^2$]"))

box = fig.axes[0].get_position()
width = box.x1 - box.x0
box.x0 = box.x0 - .07 * width
box.x1 = box.x1 - .15 * width
fig.axes[0].set_position(box)

first_legend = fig.axes[0].legend(series_lines, 
                                  [s1 + s2 for s1, s2 in zip(
                                      sum([[series_legend[i]]*2 for i in range(len(series))], []), 
                                      [" MEEP", " Mie"]*len(series))],
                                   loc="center", frameon=False, 
                                   bbox_to_anchor=(1.15, .2),
                                   bbox_transform=fig.axes[0].transAxes)
second_legend = fig.axes[0].legend(
    inside_series_lines, 
    [l.get_label() for l in inside_series_lines],
    bbox_to_anchor=(1.15, .6), columnspacing=-.5,
    loc="center", ncol=len(series), frameon=False)
fig.axes[0].add_artist(first_legend)
# leg = plt.legend(bbox_to_anchor=(1.40, .5), loc="center right", bbox_transform=fig.axes[0].transAxes)

if plot_make_big: fig.set_size_inches([10.6,  5.5])
vs.saveplot(plot_file("AllScatt.png"), overwrite=True)

#%% PLOT SCATTERING EFFICIENCY DIFFERENCE IN UNITS

fig = plt.figure()
plt.title(trs.choose("Scattering Efficiency for ",
                     "Eficiencia de dispersión en ") + plot_title_ending)

series_lines = []
inside_series_lines = []
for i in range(len(series)):
    for j in range(len(series[i])):
        l, = plt.plot(data[i][j][:,0], 
                      data[i][j][:,series_column[i]] - theory[i][j], 
                      linestyle=series_linestyles[i], color=colors[i][j])
                      # label="MEEP " + series_legend[i] + series_label[i](series[i][j]))
        if i==len(series)-1:
            l.set_label(series_label[i](series[i][j]))
        else:
            l.set_label("")
        if j==int(len(series[i])/2):
            series_lines.append(l)
        inside_series_lines.append(l)

plt.xlabel(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
plt.ylabel(trs.choose(r"Difference in Scattering Efficiency",
                      r"Diferencia en eficineica de dispersión"))

box = fig.axes[0].get_position()
width = box.x1 - box.x0
box.x0 = box.x0 - .07 * width
box.x1 = box.x1 - .15 * width
fig.axes[0].set_position(box)

first_legend = fig.axes[0].legend(series_lines, 
                                  [s1 + s2 for s1, s2 in zip(
                                      sum([[series_legend[i]]*2 for i in range(len(series))], []), 
                                      [" MEEP", " Mie"]*len(series))],
                                   loc="center", frameon=False, 
                                   bbox_to_anchor=(1.15, .2),
                                   bbox_transform=fig.axes[0].transAxes)
second_legend = fig.axes[0].legend(
    inside_series_lines, 
    [l.get_label() for l in inside_series_lines],
    bbox_to_anchor=(1.15, .6), columnspacing=-.5,
    loc="center", ncol=len(series), frameon=False)
fig.axes[0].add_artist(first_legend)
# leg = plt.legend(bbox_to_anchor=(1.40, .5), loc="center right", bbox_transform=fig.axes[0].transAxes)

if plot_make_big: fig.set_size_inches([10.6,  5.5])
vs.saveplot(plot_file("AllScattDiff.png"), overwrite=True)

#%% ONE HUGE SCATTERING PLOT

if plot_for_display: use_backend("Agg")

fig = plt.figure()
plot_grid = gridspec.GridSpec(ncols=4, nrows=2, hspace=0.4, wspace=0.5, figure=fig)
if plot_for_display: fig.dpi = 200

main_ax = fig.add_subplot(plot_grid[:,0:2])
main_ax.set_title(trs.choose("All Diameters", "Todos los diámetros"))
main_ax.xaxis.set_label_text(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
main_ax.yaxis.set_label_text(trs.choose(
    "Normalized Scattering Cross Section\n"+r"$\sigma_{scatt}/\sigma_{scatt}^{max}$",
    "Sección eficaz de dispersión normalizada\n"+r"$\sigma_{disp}/\sigma_{disp}^{max}$"))
        
lines_origin = []
lines_series = []
for i in range(len(series)):
    for j in range(len(series[i])):
        l_meep, = main_ax.plot(data[i][j][:,0], 
                               data[i][j][:,series_column[i]] / max(data[i][j][:,series_column[i]]), 
                               linestyle=series_linestyles[i], color=colors[i][j], 
                               label=series_label[i](series[i][j]))
        l_theory, = main_ax.plot(wlen_plot, 
                                 theory_plot[i][j] / max(theory_plot[i][j]), 
                                 linestyle=theory_linestyles[i], color=colors[i][j], 
                                 label=series_label[i](series[i][j]))
        if i==0 and j==len(series[0])-1:
            lines_origin = [l_meep, l_theory]
        lines_series.append(l_meep)
# main_ax.legend()

first_legend = main_ax.legend(lines_origin, trs.choose(["MEEP Data", "Mie Theory"],
                                                       ["Datos MEEP", "Teoría Mie"]),
                          loc="upper left")
second_legend = plt.legend(
    lines_series, 
    [l.get_label() for l in lines_series],
    loc="lower center",
    ncol=2)
main_ax.add_artist(first_legend)

plot_list = [plot_grid[0,2], plot_grid[0,3], plot_grid[1,2], plot_grid[1,3]]
axes_list = [fig.add_subplot(pl) for pl in plot_list]
right_axes_list = [ ax.twinx() for ax in axes_list ]
axes_list = [axes_list]*2
right_axes_list = [right_axes_list]*2

for i in range(len(series)):
    for j in range(len(series[i])):
        if test_param_units != "":
            axes_list[i][j].set_title(f"{test_param_name} {test_param[i][j]} {test_param_units}")
        else:
            axes_list[i][j].set_title(f"{test_param_name} {test_param[i][j]}")
        if i == 0:
            ax = axes_list[i][j]
        else:
            ax = right_axes_list[i][j]
        ax.plot(data[i][j][:,0], 
                data[i][j][:,series_column[i]] * np.pi * (r[i][j] * from_um_factor[i][j] * 1e3)**2,
                linestyle=series_linestyles[i], color=colors[i][j], 
                label=series_label[i](series[i][j]))
        ax.plot(wlen_plot, 
                theory_plot[i][j] * np.pi * (r[i][j] * from_um_factor[i][j] * 1e3)**2,
                linestyle=theory_linestyles[i], color=colors[i][j], 
                label=series_label[i](series[i][j]))
        ax.xaxis.set_label_text(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
        if i == 0 and (j == 0 or j == 2):
            ax.yaxis.set_label_text(trs.choose("Vacuum \n Scattering Cross Section\n"+r"$\sigma_{scatt}$ [nm$^2$]",
                                               "Vacío \n Sección eficaz de dispersión\n"+r"$\sigma_{disp}$ [nm$^2$]"))
        if i == 1 and (j == 1 or j == 3):
            ax.yaxis.set_label_text(trs.choose("Scattering Cross Section\n"+r"$\sigma_{scatt}$ [nm$^2$]"+"\n Water",
                                               "Sección eficaz de dispersión\n"+r"$\sigma_{disp}$ [nm$^2$]"+"\n Agua"))

fig.set_size_inches([18.45,  6.74])
vs.saveplot(plot_file("AllScattBig.png"), overwrite=True)

if plot_for_display: use_backend("Qt5Agg")

#%% TWO NICE SCATTERING PLOTS

if plot_for_display: use_backend("Agg")

fig = plt.figure()
if plot_for_display: fig.dpi = 200

plt.title(trs.choose("All Diameters", "Todos los diámetros"))
plt.xlabel(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
plt.ylabel(trs.choose(
    "Normalized Scattering Cross Section\n"+r"$\sigma_{scatt}/\sigma_{scatt}^{max}$",
    "Sección eficaz de dispersión normalizada\n"+r"$\sigma_{disp}/\sigma_{disp}^{max}$"))
        
lines_origin = []
lines_series = []
for i in range(len(series)):
    for j in range(len(series[i])):
        l_meep, = plt.plot(data[i][j][:,0], 
                           data[i][j][:,series_column[i]] / max(data[i][j][:,series_column[i]]), 
                           linestyle=series_linestyles[i], color=colors[i][j], 
                           label=series_label[i](series[i][j]))
        l_theory, = plt.plot(wlen_plot, 
                             theory_plot[i][j] / max(theory_plot[i][j]), 
                             linestyle=theory_linestyles[i], color=colors[i][j], 
                             label=series_label[i](series[i][j]))
        if i==0 and j==len(series[0])-1:
            lines_origin = [l_meep, l_theory]
        lines_series.append(l_meep)
# main_ax.legend()

first_legend = plt.legend(lines_origin, trs.choose(["MEEP Data", "Mie Theory"],
                                                   ["Datos MEEP", "Teoría Mie"]),
                          loc="upper left")
second_legend = plt.legend(
    lines_series, 
    [l.get_label() for l in lines_series],
    loc="lower center",
    ncol=2)
fig.axes[0].add_artist(first_legend)
plt.xlim(min(wlen_plot), max(wlen_plot))

fig.set_size_inches([10.6,  5.5])
vs.saveplot(plot_file("AllScattBig1.png"), overwrite=True)

fig = plt.figure()
plot_grid = gridspec.GridSpec(ncols=2, nrows=2, hspace=0.6, wspace=0.3, figure=fig)

plot_list = [plot_grid[0,0], plot_grid[0,1], plot_grid[1,0], plot_grid[1,1]]
axes_list = [fig.add_subplot(pl) for pl in plot_list]
right_axes_list = [ ax.twinx() for ax in axes_list ]
axes_list = [axes_list]*2
right_axes_list = [right_axes_list]*2

for i in range(len(series)):
    for j in range(len(series[i])):
        if test_param_units != "":
            axes_list[i][j].set_title(f"{test_param_name} {test_param[i][j]} {test_param_units}")
        else:
            axes_list[i][j].set_title(f"{test_param_name} {test_param[i][j]}")
        if i == 0:
            ax = axes_list[i][j]
        else:
            ax = right_axes_list[i][j]
        ax.plot(data[i][j][:,0], 
                data[i][j][:,series_column[i]] * np.pi * (r[i][j] * from_um_factor[i][j] * 1e3)**2,
                linestyle=series_linestyles[i], color=colors[i][j], 
                label=series_label[i](series[i][j]))
        ax.plot(wlen_plot, 
                theory_plot[i][j] * np.pi * (r[i][j] * from_um_factor[i][j] * 1e3)**2,
                linestyle=theory_linestyles[i], color=colors[i][j], 
                label=series_label[i](series[i][j]))
        ax.xaxis.set_label_text(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
        if i == 0 and (j == 0 or j == 2):
            ax.yaxis.set_label_text(trs.choose("Vacuum \n Scattering Cross Section\n"+r"$\sigma_{scatt}$ [nm$^2$]",
                                               "Vacío \n Sección eficaz de dispersión\n"+r"$\sigma_{disp}$ [nm$^2$]"))
        if i == 1 and (j == 1 or j == 3):
            ax.yaxis.set_label_text(trs.choose("Scattering Cross Section\n"+r"$\sigma_{scatt}$ [nm$^2$]"+"\n Water",
                                               "Sección eficaz de dispersión\n"+r"$\sigma_{disp}$ [nm$^2$]"+"\n Agua"))
        ax.set_xlim(min(wlen_plot), max(wlen_plot))

for ax in axes_list[0]:
    box = ax.get_position()
    width = box.x1 - box.x0
    box.x0 = box.x0 - .07 * width
    box.x1 = box.x1 - .07 * width
    ax.set_position(box)

fig.set_size_inches([10.6,  5.5])
vs.saveplot(plot_file("AllScattBig2.png"), overwrite=True)

if plot_for_display: use_backend("Qt5Agg")