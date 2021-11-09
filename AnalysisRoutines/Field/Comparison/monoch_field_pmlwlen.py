#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:45:00 2020

@author: vall
"""

# Monochromatic planewave field

from socket import gethostname
if "Nano" in gethostname():
    syshome = "/home/nanofisica/Documents/Vale/NanoMeepPlasmonics"
elif "vall" in gethostname():
    syshome = "/home/vall/Documents/Thesis/NanoMeepPlasmonics"
else:
    raise ValueError("Your PC must be registered at the top of this code")

import sys
sys.path.append(syshome)

# import imageio as mim
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plab
import matplotlib.gridspec as gridspec
from matplotlib.colors import colorConverter
import os
import v_analysis as va
import vmp_analysis as vma
import v_plot as vp
import v_save as vs
import v_utilities as vu

english = False
trs = vu.BilingualManager(english=english)
vp.set_style()

#%% PARAMETERS <<

# Saving directories
folder = ["Field/Sources/MonochPlanewave/TestPMLwlen/Not Centered/Vacuum",
          "Field/Sources/MonochPlanewave/TestPMLwlen/Not Centered/Water"]
home = vs.get_home()

# Parameter for the test
test_param_string = "pml_wlen_factor"
test_param_in_series = True
test_param_in_params = False
test_param_position = 0
test_param_label = trs.choose(r"PML $\lambda$ Factor", "Factor PML $\lambda$")

# Sorting and labelling data series
sorting_function = [lambda l : vu.sort_by_number(l, test_param_position)]*3
series_label = [lambda s : rf"PML {vu.find_numbers(s)[test_param_position]:.2f} $\lambda$"]*3
series_must = [""]*3 # leave "" per default
series_mustnt = [["Fail"]]*3 # leave "" per default

# Scattering plot options
plot_title_base = trs.choose('Dimnesionless monochromatic wave', 
                             "Onda monocromática adimensional")
series_legend = trs.choose(["Vacuum", r"Water"], ["Vacío", r"Agua"])
series_colormaps = [plab.cm.Reds, plab.cm.Blues, plab.cm.YlGn]
series_colors = ["red", "blue", "limegreen"]
series_markers = ["o", "o", "o"]
series_markersize = [8, 7, 6]
series_linestyles = ["solid"]*3
plot_make_big = False
plot_folder = "DataAnalysis/Field/Sources/MonochPlanewave/TestPMLwlen/Official"

force_normalization = False
periods_sensitivity = 0.05
amplitude_sensitivity = 0.05
peaks_sep_sensitivity = 0.2 # 0.1

# force_normalization = True
# periods_sensitivity = 0.12 # 0.05
# amplitude_sensitivity = 0.06 # 0.05
# peaks_sep_sensitivity = 0.2 # 0.1

#%% LOAD DATA <<

def file_definer(path): return lambda s, n : os.path.join(path, s, n)

path = [os.path.join(home, fold) for fold in folder]
file = [file_definer(pa) for pa in path]

series = [[]] * len(path)
for i in range(len(folder)):    
    series[i] = os.listdir(path[i])
    series[i] = vu.filter_by_string_must(series[i], series_must[i])
    if series_mustnt[i]!="": 
        series[i] = vu.filter_by_string_must(series[i], series_mustnt[i], False)
    series[i] = sorting_function[i](series[i])
del i
    
files_line = [[h5.File(file[i](series[i][j], "Field-Lines.h5"), "r") 
               for j in range(len(series[i]))] for i in range(len(series))]
files_plane = [[h5.File(file[i](series[i][j], "Field-Planes.h5"), "r") 
                for j in range(len(series[i]))] for i in range(len(series))]

results_line = [[files_line[i][j]["Ez"] 
                 for j in range(len(series[i]))] for i in range(len(series))]
results_plane = [[files_plane[i][j]["Ez"] 
                  for j in range(len(series[i]))] for i in range(len(series))]

params = [[dict(files_line[i][j]["Ez"].attrs)
           for j in range(len(series[i]))] for i in range(len(series))]

t_line = [[np.asarray(files_line[i][j]["T"])
           for j in range(len(series[i]))] for i in range(len(series))]
x_line = [[np.asarray(files_line[i][j]["X"])
           for j in range(len(series[i]))] for i in range(len(series))]

t_plane = [[np.asarray(files_plane[i][j]["T"])
            for j in range(len(series[i]))] for i in range(len(series))]
y_plane = [[np.asarray(files_plane[i][j]["Y"])
            for j in range(len(series[i]))] for i in range(len(series))]
z_plane = [[np.asarray(files_plane[i][j]["Z"])
            for j in range(len(series[i]))] for i in range(len(series))]

for i in range(len(series)):
    for j in range(len(series[i])):
        try:
            f = h5.File(file[i](series[i][j], "Resources.h5"))
            params[i][j]["used_ram"] = np.array(f["RAM"])
            params[i][j]["used_swap"] = np.array(f["SWAP"])
            params[i][j]["elapsed_time"] = np.array(f["ElapsedTime"])
        except FileNotFoundError:
            f = h5.File(file[i](series[i][j], "RAM.h5"))
            params[i][j]["used_ram"] = np.array(f["RAM"])
            params[i][j]["used_swap"] = np.array(f["SWAP"])
            params[i][j]["elapsed_time"] = params["elapsed"]
            params.pop("elapsed")
del i, j
            
requires_normalization = False
from_um_factor = []
resolution = []
resolution_wlen = []
units = []
index = []
wlen = []
cell_width = []
pml_width = []
source_center = []
period_plane = []
period_line = []
until_time = []
time_period_factor = []
norm_amplitude = []
norm_period = []
sysname = []
for p in params:
    from_um_factor.append( [pi["from_um_factor"] for pi in p] )
    resolution.append( [pi["resolution"] for pi in p] )
    resolution_wlen.append( [pi["resolution_wlen"] for pi in p] )
    units.append( [pi["units"] for pi in p] )
    index.append( [pi["submerged_index"] for pi in p] )
    wlen.append( [pi["wlen"] for pi in p] )
    cell_width.append( [pi["cell_width"] for pi in p] )
    pml_width.append( [pi["pml_width"] for pi in p] )
    source_center.append( [pi["source_center"] for pi in p] )
    period_plane.append( [pi["period_plane"] for pi in p] )
    period_line.append( [pi["period_line"] for pi in p] )
    until_time.append( [pi["until_time"] for pi in p] )
    time_period_factor.append( [pi["time_period_factor"] for pi in p] )
    if not requires_normalization:
        try:
            norm_amplitude.append( [pi["norm_amplitude"] for pi in p] )    
            norm_period.append( [pi["norm_period"] for pi in p] )
            requires_normalization = False
        except:
            norm_amplitude = []
            norm_period = []
            requires_normalization = True
    sysname.append( [pi["sysname"] for pi in p] )
del p

if test_param_in_params:
    test_param = [[p[test_param_string] for p in par] for par in params]
else:
    test_param = [[vu.find_numbers(s)[test_param_position] for s in ser] for ser in series]
    
use_units = True in [True in units[i] for i in range(len(series))]

minor_division = [[fum * 1e3 / res for fum, res in zip(frum, reso)] for frum, reso in zip(from_um_factor, resolution)]
try:
    width_points = [[int(p["cell_width"] * p["resolution"]) for p in par] for par in params] 
    effective_width_points = [[(p["cell_width"] - 2 * params["pml_width"]) * p["resolution"] for p in par] for par in params]
except:
    width_points = [[2*int((p["pml_width"] + p["empty_width"]) * p["resolution"]) for p in par] for par in params] 
    effective_width_points = [[2*int(p["empty_width"] * p["resolution"]) for p in par] for par in params]
grid_points = [[wp**3 for wp in wpoints] for wpoints in width_points]
memory_B = [[2 * 12 * gp * 32 for p, gp in zip(par, gpoints)] for par, gpoints in zip(params, grid_points)] # in bytes

elapsed_time = [[p["elapsed_time"] for p in par] for par in params]
total_elapsed_time = [[sum(p["elapsed_time"]) for p in par] for par in params]

used_ram = [[np.array(p["used_ram"])/(1024)**2 for p in par] for par in params]
total_used_ram = [[np.sum(used_ram[i][j], axis=1) for j in range(len(series[i]))] for i in range(len(series))]
used_swap = [[p["used_swap"] for p in par] for par in params]

#%% POSITION RECONSTRUCTION <<

t_line_index = [[vma.def_index_function(t_line[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
x_line_index = [[vma.def_index_function(x_line[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

t_plane_index = [[vma.def_index_function(t_plane[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
y_plane_index = [[vma.def_index_function(y_plane[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
z_plane_index = [[vma.def_index_function(z_plane[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

x_line_cropped = [[x_line[i][j][:x_line_index[i][j](cell_width[i][j]/2 - pml_width[i][j])+1] for j in range(len(series[i]))] for i in range(len(series))]
x_line_cropped = [[x_line_cropped[i][j][x_line_index[i][j](-cell_width[i][j]/2 + pml_width[i][j]):] for j in range(len(series[i]))] for i in range(len(series))]

y_plane_cropped = [[y_plane[i][j][:y_plane_index[i][j](cell_width[i][j]/2 - pml_width[i][j])+1] for j in range(len(series[i]))] for i in range(len(series))]
y_plane_cropped = [[y_plane_cropped[i][j][y_plane_index[i][j](-cell_width[i][j]/2 + pml_width[i][j]):] for j in range(len(series[i]))] for i in range(len(series))]

z_plane_cropped = [[z_plane[i][j][:z_plane_index[i][j](cell_width[i][j]/2 - pml_width[i][j])+1] for j in range(len(series[i]))] for i in range(len(series))]
z_plane_cropped = [[z_plane_cropped[i][j][z_plane_index[i][j](-cell_width[i][j]/2 + pml_width[i][j]):] for j in range(len(series[i]))] for i in range(len(series))]

#%% DATA EXTRACTION <<

source_results = [[vma.get_source_from_line(results_line[i][j], x_line_index[i][j], source_center[i][j]) 
                   for j in range(len(series[i]))] for i in range(len(series))]

if not requires_normalization and not force_normalization:
    
    period_results, amplitude_results = norm_period, norm_amplitude
    
else:
    
    period_results = [[vma.get_period_from_source(source_results[i][j], t_line[i][j],
                                                  peaks_sep_sensitivity=peaks_sep_sensitivity,
                                                  periods_sensitivity=periods_sensitivity)[-1] 
                       for j in range(len(series[i]))] for i in range(len(series))]
    amplitude_results = [[vma.get_amplitude_from_source(source_results[i][j],
                                                        amplitude_sensitivity=amplitude_sensitivity,
                                                        peaks_sep_sensitivity=peaks_sep_sensitivity)[-1] 
                          for j in range(len(series[i]))] for i in range(len(series))]
    
    source_results = [[source_results[i][j] / amplitude_results[i][j] 
                       for j in range(len(series[i]))] for i in range(len(series))]
    results_plane = [[np.asarray(results_plane[i][j]) / amplitude_results[i][j] 
                      for j in range(len(series[i]))] for i in range(len(series))]
    results_line = [[np.asarray(results_line[i][j]) / amplitude_results[i][j] 
                     for j in range(len(series[i]))] for i in range(len(series))]
    
    norm_period, norm_amplitude = period_results, amplitude_results

cropped_line = [[vma.crop_field_xprofile(results_line[i][j], x_line_index[i][j], 
                                          cell_width[i][j], pml_width[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

cropped_plane = [[vma.crop_field_yzplane(results_plane[i][j], y_plane_index[i][j], z_plane_index[i][j],
                                          cell_width[i][j], pml_width[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

#%% GENERAL PLOT CONFIGURATION <<

if not os.path.isdir(os.path.join(home, plot_folder)):
    os.mkdir(os.path.join(home, plot_folder))
plot_file = lambda n : os.path.join(home, plot_folder, n)

colors = [sc(np.linspace(0,1,len(s)+6))[6:] 
          for sc, s in zip(series_colormaps, series)]

#%% BASIC CONTROL: DIMENSIONS PLOTS

fig = plt.figure()
ax = plt.subplot()
ax2 = plt.twinx()
lines, lines2, lines3 = [], [], []
for i in range(len(series)):
    l, = ax.plot(test_param[i],
                 [params[i][j]["courant"]/params[i][j]["resolution"] for j in range(len(series[i]))], 
                 "o", color=series_colors[i], alpha=0.5, markersize=series_markersize[i])
    l2, = ax.plot(test_param[i],
                  [1/params[i][j]["resolution"] for j in range(len(series[i]))], 
                  "o", color=series_colors[i], fillstyle="none", markersize=series_markersize[i]+1)
    l3, = ax.plot(test_param[i],
                 [period_line[i][j] for j in range(len(series[i]))], 
                 "o", color=series_colors[i], alpha=0.5, fillstyle="top", markersize=series_markersize[i]+2)
    lines.append(l)
    lines2.append(l2)
    lines3.append(l3)
plt.xlabel(test_param_label)
ax.set_ylabel(trs.choose("Time Minimum Division [MPu]", "Mínima división del tiempo [uMP]"))
ax2.set_ylabel(trs.choose("Space Minimum Division [MPu]", "Space división del tiempo [uMP]"))
plt.legend([*lines, *lines3, *lines2], 
           [*[s + r" $\Delta t$" for s in series_legend], 
            *[s + r" $\Delta t_{line}$" for s in series_legend],
            *[s + r" $\Delta r$" for s in series_legend]])
plt.savefig(plot_file("MinimumDivision.png"))

plt.figure()
for i in range(len(series)):
    plt.plot(test_param[i],
             [params[i][j]["resolution"] for j in range(len(series[i]))], "o", color=series_colors[i], alpha=0.5)
plt.xlabel(test_param_label)
plt.ylabel(trs.choose(r"Resolution [points/$\Delta r$]", r"Resolución [puntos/$\Delta r$]"))
# plt.ylabel(trs.choose("Number of points in time", "Número de puntos en el tiempo"))
plt.legend(series_legend)
plt.savefig(plot_file("Resolution.png"))

plt.figure()
for i in range(len(series)):        
    plt.plot(test_param[i],
             [results_line[i][j].shape[0] for j in range(len(series[i]))], "o", color=series_colors[i], alpha=0.5)
plt.xlabel(test_param_label)
plt.ylabel(trs.choose("Number of points in whole cell", "Número de puntos en la celda completa"))
plt.legend(series_legend)

plt.savefig(plot_file("Points.png"))

plt.figure()
for i in range(len(series)):        
    plt.plot(test_param[i],
             [cropped_line[i][j].shape[0] for j in range(len(series[i]))], "o", color=series_colors[i], alpha=0.5)
plt.xlabel(test_param_label)
plt.ylabel(trs.choose("Number of points in real cell", "Número de puntos en la celda real"))
plt.legend(series_legend)

plt.savefig(plot_file("InnerPoints.png"))

# these_markersize = [6,7,8]
# plt.figure()
# lines, lines2 = [], []
# for i in range(len(series)):
#     l, = plt.plot(test_param[i],
#                   [until_time[i][j]/period_line[i][j] for j in range(len(series[i]))], "o", color=series_colors[i], alpha=0.5, markersize=these_markersize[i]+1, markeredgewidth=0)
#     l2, = plt.plot(test_param[i],
#                    [results_line[i][j].shape[-1] for j in range(len(series[i]))], "o", color=series_colors[i], alpha=1, fillstyle="none", markersize=these_markersize[i])
#     lines.append(l)
#     lines2.append(l2)
# plt.xlabel(test_param_label)
# plt.ylabel(trs.choose("Number of points in time", "Número de puntos en el tiempo"))
# plt.legend(series_legend)
# plt.legend([*lines, *lines2], 
#            [*[s + trs.choose(" Predicted Points", " predicción") for s in series_legend], 
#             *[s + trs.choose(" Actual Points", " realidad") for s in series_legend]])

plt.figure()
for i in range(len(series)):
    plt.plot(test_param[i],
                  [results_line[i][j].shape[-1] for j in range(len(series[i]))], "o", color=series_colors[i], alpha=0.5)
plt.xlabel(test_param_label)
plt.ylabel(trs.choose("Number of points in time", "Número de puntos en el tiempo"))
plt.legend(series_legend)

plt.savefig(plot_file("TimePoints.png"))

#%% SPECIFIC CONTROL: TIME VARIABLES PLOT

plt.figure()
for i in range(len(series)):
    plt.plot(test_param[i],
             [until_time[i][j] for j in range(len(series[i]))], "o", color=series_colors[i], alpha=0.5)
plt.xlabel(test_param_label)
plt.ylabel(trs.choose("Simulation time [MPu]", "Tiempo de simulación [uMP]"))
plt.legend(series_legend)

plt.savefig(plot_file("UntilTime.png"))

plt.figure()
for i in range(len(series)):
    l, = plt.plot(test_param[i],
                  [period_line[i][j] for j in range(len(series[i]))], "o", color=series_colors[i], alpha=0.5)
plt.xlabel(test_param_label)
plt.ylabel(trs.choose("Line period [MPu]", "Período de líneas [uMP]"))
plt.legend(series_legend)

plt.savefig(plot_file("PeriodLine.png"))

#%% MAKE FOURIER ANALYSIS FOR SOURCE <<

fourier = [[np.abs(np.fft.rfft(source_results[i][j])) for j in range(len(series[i]))] for i in range(len(series))]
fourier_freq = [[np.fft.rfftfreq(len(source_results[i][j]), d=period_line[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
if use_units:
    fourier_wlen = [[from_um_factor[i][j] * 1e3 / fourier_freq[i][j] / index[i][j] for j in range(len(series[i]))] for i in range(len(series))]
    fourier_best = [[wlen[i][j] * from_um_factor[i][j] * 1e3 / index[i][j] for j in range(len(series[i]))] for i in range(len(series))]
else:
    fourier_wlen = [[1 / fourier_freq[i][j] / index[i][j]  for j in range(len(series[i]))] for i in range(len(series))]
    fourier_best = [[wlen[i][j] / index[i][j] for j in range(len(series[i]))] for i in range(len(series))]

fourier_max_wlen = [[fourier_wlen[i][j][ np.argmax(fourier[i][j]) ]  for j in range(len(series[i]))] for i in range(len(series))]
fourier_max_best = [[fourier_wlen[i][j][ np.argmin(np.abs(fourier_wlen[i][j] - fourier_best[i][j])) ]  for j in range(len(series[i]))] for i in range(len(series))]

#%% SHOW SOURCE AND FOURIER USED FOR NORMALIZATION

fig = plt.figure()
plt.title(plot_title_base)

series_lines = []
for i in range(len(series)):
    for j in range(len(series[i])):
        l, = plt.plot(t_line[i][j] / period_results[i][j], 
                      source_results[i][j],
                      color=colors[i][j])            
        if i == len(series)-1:
            l.set_label(series_label[i](series[i][j]))
        else:
            l.set_label(" ")
        if j == int( 2 * len(series[i]) / 3 ):
            series_lines.append(l)
plt.xlabel(trs.choose("Time in multiples of period", "Tiempo en múltiplos del período"))
plt.ylabel(trs.choose(r"Electric Field $E_z(y=z=0)$",
                      r"Campo eléctrico $E_z(y=z=0)$"))

box = fig.axes[0].get_position()
box.x1 = box.x1 - .3 * (box.x1 - box.x0)
fig.axes[0].set_position(box)
leg = plt.legend(ncol=len(series), columnspacing=-0.5, 
                 bbox_to_anchor=(1.6, .5), loc="center right", frameon=False)

plt.savefig(plot_file("Source.png"))
        
fig = plt.figure()
plt.title(plot_title_base)

series_lines = []
for i in range(len(series)):
    for j in range(len(series[i])):
        l, = plt.plot(fourier_wlen[i][j], fourier[i][j],
                      color=colors[i][j])
        if i == len(series)-1:
            l.set_label(series_label[i](series[i][j]))
        else:
            l.set_label(" ")
        if j == int( 2 * len(series[i]) / 3 ):
            series_lines.append(l)
if use_units:
    plt.xlabel(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
else:
    plt.xlabel(trs.choose("Wavelength [MPu]", "Longitud de onda [uMP]"))
plt.ylabel(trs.choose(r"Electric Field Fourier $\mathcal{F}\;(E_z)$",
                      r"Transformada del campo eléctrico $\mathcal{F}\;(E_z)$"))
box = fig.axes[0].get_position()
box.x1 = box.x1 - .3 * (box.x1 - box.x0)
fig.axes[0].set_position(box)

first_legend = plt.legend(series_lines, series_legend)
second_legend = plt.legend(ncol=len(series), columnspacing=-0.5, bbox_to_anchor=(1.6, .5), 
                           loc="center right", frameon=False)
plt.gca().add_artist(first_legend)

# fig.axes[0].set_yscale("log")

plt.savefig(plot_file("SourceFFT.png"))

if use_units: plt.xlim([350, 850])
else: plt.xlim([0, 2])
# else: plt.xlim([0, 4])
        
plt.savefig(plot_file("SourceFFTZoom.png"))

#%% WAVELENGTH OPTIMIMUM VARIATION PLOT

plt.figure()
plt.title(plot_title_base)
for i in range(len(series)):
    plt.plot(test_param[i], 
             100 * ( np.array(fourier_max_wlen[i]) - np.array(fourier_max_best[i]) ) / np.array(fourier_max_best[i]), 
             color=series_colors[i], marker=series_markers[i], alpha=0.4,
             markersize=8, linestyle="", markeredgewidth=0)
plt.legend(series_legend)
plt.xlabel(test_param_label)
plt.ylabel(trs.choose("Maximum Wavelength Percentual Variation", 
                      "Variación porcentual de la longitud de onda máxima\n") + 
           r"$\lambda_{max} = \argmax [\mathcal{F}\;(E_z)]$ [%]")
plt.tight_layout()
vs.saveplot(plot_file("LambdaVariation.png"), overwrite=True)

#%% GET AMPLITUDE AND PERIOD FROM SOURCE FULL SIGNAL <<

this_peaks_sep_sensitivity = .2

peaks_index = [[vma.get_peaks_from_source(source_results[i][j], peaks_sep_sensitivity=this_peaks_sep_sensitivity) for j in range(len(series[i]))] for i in range(len(series))]
peaks_heights = [[np.abs(source_results[i][j][peaks_index[i][j]]) for j in range(len(series[i]))] for i in range(len(series))]
peaks_times = [[t_line[i][j][peaks_index[i][j]] for j in range(len(series[i]))] for i in range(len(series))]
peaks_periods = [[np.diff(peaks_times[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

peaks_height_variation = [[100 * ( max(peaks_heights[i][j]) - min(peaks_heights[i][j]) ) / amplitude_results[i][j] for j in range(len(series[i]))] for i in range(len(series))]
peaks_period_variation = [[100 * ( max(peaks_periods[i][j]) - min(peaks_periods[i][j]) ) / period_results[i][j] for j in range(len(series[i]))] for i in range(len(series))]

#%% AMPLITUDE VARIATION PLOT

plt.figure()
plt.title(plot_title_base)
for i in range(len(series)):
    plt.plot(test_param[i], peaks_height_variation[i], 
             color=series_colors[i], marker=series_markers[i], alpha=0.4,
             markersize=8, linestyle="", markeredgewidth=0)
plt.legend(series_legend)
plt.xlabel(test_param_label)
plt.ylabel(trs.choose("Maximum percentual variation in amplitude\n", 
                      "Diferencia porcentual máxima en amplitud\n") + 
           r"$\max[ E_z(y=z=0) ]$ [%]")
           # r"$\frac{\max|E^{max}|_i - \min|E^{max}|_i}{\min|E^{max}|_i}$")
vs.saveplot(plot_file("AmpVariation.png"), overwrite=True)

#%% PERIOD VARIATION PLOT

plt.figure()
plt.title(plot_title_base)
for i in range(len(series)):
    plt.plot(test_param[i], peaks_period_variation[i], 
             color=series_colors[i], marker=series_markers[i], markersize=8, 
             alpha=0.4, linestyle="", markeredgewidth=0)
plt.legend(series_legend)
plt.xlabel(test_param_label)
plt.ylabel(trs.choose("Maximum percentual variation in period ", 
                      "Diferencia porcentual máxima en período ") + 
           r"$T$ [%]")
           # r"$\frac{\max T_i - \min T_i}{\min T_i}$")
vs.saveplot(plot_file("PerVariation.png"), overwrite=True)

#%% FULL SOURCE ANALYSIS <<

fig = plt.figure()

plot_grid = gridspec.GridSpec(ncols=6, nrows=6, hspace=0, wspace=0.2, figure=fig)
main_axes = [fig.add_subplot(plot_grid[:2,:3]), fig.add_subplot(plot_grid[2:4,:3])]
fourier_ax = fig.add_subplot(plot_grid[-2:,:3])
amp_ax = fig.add_subplot(plot_grid[:3,-3:])
per_ax = fig.add_subplot(plot_grid[-3:,-3:])

color_lines = []
series_lines = []
for i in range(len(series)):
    
    for j in range(len(series[i])):
        l, = main_axes[i].plot(t_line[i][j] / period_results[i][j], 
                               source_results[i][j],
                               color=colors[i][j])
        main_axes[i].axhline(color="k", linewidth=0.5)
        
        fourier_ax.plot(fourier_wlen[i][j], fourier[i][j], ".-",
                        color=colors[i][j])
        if i == len(series)-1:
            l.set_label(series_label[i](series[i][j]))
        else:
            l.set_label(r"1p$\lambda$")
        color_lines.append(l)
        if j == int( 2 * len(series[i]) / 3 ):
            series_lines.append(l)
    
    amp_ax.plot(test_param[i], peaks_height_variation[i], 
                color=series_colors[i], marker=series_markers[i], alpha=0.4,
                linestyle="", markeredgewidth=0)
    
    per_ax.plot(test_param[i], peaks_period_variation[i], 
                color=series_colors[i], marker=series_markers[i],
                alpha=0.4, linestyle="", markeredgewidth=0)

main_axes[0].xaxis.tick_top()
main_axes[0].xaxis.set_label_position("top")
main_axes[0].set_xlabel(trs.choose(r"Time $T$ [$\tau$]", 
                                   r"Tiempo $T$ [$\tau$]"))
main_axes[0].set_ylabel(trs.choose(r"Electric Field $E_z(y=z=0)$",
                                   r"Campo eléctrico $E_z(y=z=0)$"))

fourier_ax.set_yscale("log")
fourier_ax.set_ylim(10e-3, 10e3)
if use_units:
    fourier_ax.set_xlabel(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
    fourier_ax.set_xlim([350, 850])
else:
    fourier_ax.set_xlabel(trs.choose(r"Wavelength $\lambda/n$ in medium [$\lambda]", 
                                     r"Longitud de onda $\lambda/n$ en medio [$\lambda$]"))
    # fourier_ax.set_xlim([0.5, 3.5])
    fourier_ax.set_xlim(np.min([[1/index[i][j] for j in range(len(series[i]))] for i in range(len(series))])-.3,
                        np.max([[1/index[i][j] for j in range(len(series[i]))] for i in range(len(series))])+.3)
fourier_ax.set_ylabel(trs.choose("Electric Field Fourier\n" + r"$\mathcal{F}\;(E_z)$",
                                 "Transformada del\n" + "campo eléctrico " + r"$\mathcal{F}\;(E_z)$"))

amp_ax.xaxis.set_ticklabels([])
amp_ax.set_ylabel(trs.choose("Maximum percentual variation\n" + "in amplitud ", 
                             "Diferencia porcentual máxima\n" + "en amplitud ") + r"$|E_{z0}|$ [%]")

per_ax.set_xlabel(trs.choose(r"Resolution [points/$\lambda$]", 
                             r"Resolución [puntos/$\lambda$]"))
per_ax.set_ylabel(trs.choose("Maximum percentual variation\n" + "in period ", 
                             "Diferencia porcentual máxima\n" + "en período ") + r"$\tau$ [%]")

for ax in fig.axes:
    box = ax.get_position()
    width = box.x1 - box.x0
    box.x0 = box.x0 - .13 * width
    box.x1 = box.x0 + width
    ax.set_position(box)
    
for ax in [amp_ax, per_ax]:
    box = ax.get_position()
    box.x1 = box.x1 - .3 * (box.x1 - box.x0)
    ax.set_position(box)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
fig.set_size_inches([11.58,  7.12])

first_legend = plt.legend(series_lines, series_legend,
                          bbox_to_anchor=(2.4, 1.7), 
                          bbox_transform=main_axes[-1].axes.transAxes)

second_legend = main_axes[0].legend([*main_axes[0].lines, *main_axes[1].lines],
                                    [l.get_label() for l in [*main_axes[0].lines, *main_axes[1].lines]],
                                    ncol=len(series), columnspacing=0.3, 
                                    bbox_to_anchor=(2.48, 0), 
                                    bbox_transform=main_axes[-1].axes.transAxes,
                                    loc="center right", frameon=False)

third_legend = amp_ax.legend(amp_ax.lines, series_legend,
                             bbox_to_anchor=(2.4, 2.1), 
                             bbox_transform=main_axes[-1].axes.transAxes)

# plt.gca().add_artist(first_legend)
# plt.gca().add_artist(second_legend)

#%% ANALYSE X AXIS FOR DIFFERENT POSITIONS VIA FOURIER

n_probe = 3

x_probe_fourier_position = [[ [-cell_width[i][j]/2 + pml_width[i][j] + k * (cell_width[i][j] - 2*pml_width[i][j]) / (n_probe-1) 
                               for k in range(n_probe)] for j in range(len(series[i]))] for i in range(len(series))]
x_probe_fourier_position_factor = [[ [x_probe_fourier_position[i][j][k] / (cell_width[i][j] - 2*pml_width[i][j])
                                      for k in range(n_probe)] for j in range(len(series[i]))] for i in range(len(series))]

x_probe_fourier_field = [[ [results_line[i][j][
                        x_line_index[i][j](-cell_width[i][j]/2 + pml_width[i][j] + k * (cell_width[i][j] - 2*pml_width[i][j]) / (n_probe-1)), :
                    ] for k in range(n_probe)] for j in range(len(series[i]))] for i in range(len(series))]

x_probe_fourier_field = [[ [results_line[i][j][x_line_index[i][j](x_probe_fourier_position[i][j][k]), :]
                            for k in range(n_probe)] for j in range(len(series[i]))] for i in range(len(series))]
    
fourier_cropping_index = [[ [np.where(np.abs(x_probe_fourier_field[i][j][k]) > 0.05)[0][0] for k in range(n_probe)] 
                           for j in range(len(series[i]))] for i in range(len(series))]

x_probe_fourier_field = [[ [x_probe_fourier_field[i][j][k][ fourier_cropping_index[i][j][k] : ]  for k in range(n_probe)] 
                          for j in range(len(series[i]))] for i in range(len(series))]
x_probe_fourier_time = [[ [t_line[i][j][ fourier_cropping_index[i][j][k] : ]  for k in range(n_probe)] 
                         for j in range(len(series[i]))] for i in range(len(series))]

x_probe_fourier = [[ [np.abs(np.fft.rfft(x_probe_fourier_field[i][j][k], norm="ortho")) for k in range(n_probe)] 
                    for j in range(len(series[i]))] for i in range(len(series))]
x_probe_freqs = [[ [np.fft.rfftfreq(len(x_probe_fourier_field[i][j][k]), d=period_line[i][j]) for k in range(n_probe)] 
                  for j in range(len(series[i]))] for i in range(len(series))]
if use_units:
    x_probe_wlen = [[ [from_um_factor[i][j] * 1e3 / x_probe_freqs[i][j][k] for k in range(n_probe)] for j in range(len(series[i]))] for i in range(len(series))]
else:
    x_probe_wlen = [[ [1 / x_probe_freqs[i][j][k] for k in range(n_probe)] for j in range(len(series[i]))] for i in range(len(series))]

#%% X FOURIER PROBE PLOT

def see_x_probe_fourier(i,j):
    
    colors = plab.cm.winter(np.linspace(0,1,n_probe))
    
    plt.figure()
    
    plt.title(f"{series_legend[i]} {series_label[1](series[i][j])}")
    for k in range(n_probe):
        plt.plot( x_probe_fourier_time[i][j][k],
                  x_probe_fourier_field[i][j][k], 
                  label=f"x = {x_probe_fourier_position[i][j][k]:.2f} " + trs.choose("cell", "celda"), 
                  color=colors[k])
    plt.legend()

    plt.xlabel(trs.choose("Time T [Mp.u]", "Tiempo T [uMP]"))
    plt.ylabel(trs.choose("Electric Field\n" + r"$E_z(y=z=0)$",
                          "Campo eléctrico\n" + r"$E_z(y=z=0)$"))
        
    plt.figure()
    
    plt.title(f"{series_legend[i]} {series_label[1](series[i][j])}")
    for k in range(n_probe):
        plt.plot( x_probe_wlen[i][j][k], 
                  x_probe_fourier[i][j][k], 
                  label=f"x = {x_probe_fourier_position[i][j][k]:.2f} " + trs.choose("cell", "celda"), 
                  color=colors[k])
    plt.xlim(0,5)
    plt.legend()
    
    plt.xlabel(trs.choose("Frequency f [Mp.u]", "Frecuencia f [uMP]"))
    plt.ylabel(trs.choose("Electric Field\n" + r"$E_z(y=z=0)$",
                          "Campo eléctrico\n" + r"$E_z(y=z=0)$"))
    
#%% ANALYSE X AXIS FOR DIFFERENT POSITIONS VIA FIT AND RESIDUA <<

n_x_probe = 20

x_probe_position = [[ [-cell_width[i][j]/2 + pml_width[i][j] + k * (cell_width[i][j] - 2*pml_width[i][j]) / (n_x_probe-1) 
                       for k in range(n_x_probe)] for j in range(len(series[i]))] for i in range(len(series))]
x_probe_position_factor = [[ [x_probe_position[i][j][k] / (cell_width[i][j] - 2*pml_width[i][j])
                              for k in range(n_x_probe)] for j in range(len(series[i]))] for i in range(len(series))]

x_probe_field = [[ [results_line[i][j][x_line_index[i][j](x_probe_position[i][j][k]), :]
                    for k in range(n_x_probe)] for j in range(len(series[i]))] for i in range(len(series))]

cropping_index = [[ [np.where(np.abs(x_probe_field[i][j][k]) > 0.05)[0][0] for k in range(n_x_probe)] for j in range(len(series[i]))] for i in range(len(series))]

x_probe_field = [[ [x_probe_field[i][j][k][ cropping_index[i][j][k] : ]  for k in range(n_x_probe)] for j in range(len(series[i]))] for i in range(len(series))]
x_probe_time = [[ [t_line[i][j][ cropping_index[i][j][k] : ]  for k in range(n_x_probe)] for j in range(len(series[i]))] for i in range(len(series))]

# def fit_function_generator(norm_period):
#     def fit_function(time, amplitude, phase, offset):
#         omega = 2 * np.pi / norm_period
#         return amplitude * np.cos( omega * time + phase ) + offset
#     return fit_function
# x_probe_fit_functions = [[fit_function_generator(norm_period[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

def x_probe_fit_function(time, amplitude, omega, phase, offset):
    return amplitude * np.cos( omega * time + phase ) + offset

x_probe_fit_amplitude = []
x_probe_fit_omega = []
x_probe_fit_phase = []
x_probe_fit_offset = []
x_probe_fit_residua = []
for i in range(len(series)):
    x_probe_fit_amplitude.append([])
    x_probe_fit_omega.append([])
    x_probe_fit_phase.append([])
    x_probe_fit_offset.append([])
    x_probe_fit_residua.append([])
    for j in range(len(series[i])):
        x_probe_fit_amplitude[i].append([])
        x_probe_fit_omega[i].append([])
        x_probe_fit_phase[i].append([])
        x_probe_fit_offset[i].append([])
        x_probe_fit_residua[i].append([])
        for k in range(n_x_probe):
            rsq, fit_params = va.nonlinear_fit(
                x_probe_time[i][j][k],
                x_probe_field[i][j][k],
                x_probe_fit_function, # x_probe_fit_functions[i][j],
                initial_guess=(1, 2 * np.pi / norm_period[i][j], 0, 0),
                showplot=False)
            amplitude, omega, phase, offset = fit_params[0][0], fit_params[1][0], fit_params[2][0], fit_params[3][0]
            x_probe_fit_amplitude[i][j].append( amplitude )
            x_probe_fit_omega[i][j].append( omega )
            x_probe_fit_phase[i][j].append( phase )
            x_probe_fit_offset[i][j].append( offset )
            
            residua = np.array(x_probe_field[i][j][k]) - x_probe_fit_function( x_probe_time[i][j][k], amplitude, omega, phase, offset)
            
            x_probe_fit_residua[-1][-1].append( residua )

x_probe_fit_res_std = [[ [np.std(x_probe_fit_residua[i][j][k]) for k in range(n_x_probe)] for j in range(len(series[i]))] for i in range(len(series))]

#%% X PROBE PLOT <<

# see_x_probe(0,6) a ver, tirá fachaaa con n_x_period=6 demostrativo
def see_x_probe(i,j):
    
    colors = plab.cm.jet(np.linspace(0,1,n_x_probe+2)[1:-1])
    
    fig = plt.figure()
    
    plot_grid = gridspec.GridSpec(ncols=1, nrows=6, hspace=0, figure=fig)
    main_ax = fig.add_subplot(plot_grid[:4,:])
    res_ax = fig.add_subplot(plot_grid[-2:,:])
    
    plt.suptitle(f"{series_legend[i]} {series_label[1](series[i][j])}")
    for k in range(n_x_probe):        
        main_ax.plot(x_probe_time[i][j][k][:int(.35*len(x_probe_time[i][j][k]))],
                     x_probe_field[i][j][k][:int(.35*len(x_probe_time[i][j][k]))], 
                     color=colors[k],
                     label=fr"x = {x_probe_position_factor[i][j][k]:.2f} $\lambda/n$")
        main_ax.plot(x_probe_time[i][j][k][:int(.35*len(x_probe_time[i][j][k]))],
                     x_probe_fit_function( x_probe_time[i][j][k][:int(.35*len(x_probe_time[i][j][k]))],
                                           x_probe_fit_amplitude[i][j][k],
                                           x_probe_fit_omega[i][j][k],
                                           x_probe_fit_phase[i][j][k],
                                           x_probe_fit_offset[i][j][k] ),
                     linestyle="dashed", color="k", alpha=0.5)
    
   
    res_ax.set_xlabel(trs.choose("Time T [period]", "Tiempo T [período]"))
    res_ax.set_ylabel(trs.choose("Electric Field\n" + r"$E_z(y=z=0)$ Residua",
                                 "Residuos del\n" + "campo eléctrico\n" + "r""$E_z(y=z=0)$"))
    main_ax.set_ylabel(trs.choose("Electric Field\n" + r"$E_z(y=z=0)$",
                                  "Campo eléctrico\n" + r"$E_z(y=z=0)$"))
    
    for k in range(n_x_probe):
        res_ax.plot(x_probe_time[i][j][k][:int(.35*len(x_probe_time[i][j][k]))],
                    x_probe_fit_residua[i][j][k][:int(.35*len(x_probe_time[i][j][k]))],
                    "o", color=colors[k], markeredgewidth=0, alpha=0.5,
                    label=fr"x = {x_probe_position_factor[i][j][k]:.2f} $\lambda/n$")
    res_ax.axhline(0, color="k", linewidth=0.5)
    
    for ax in fig.axes:
        box = ax.get_position()
        box.x1 = box.x1 - .2 * (box.x1 - box.x0)
        ax.set_position(box)
    
    first_legend = main_ax.legend(bbox_to_anchor=(1.2, .65),  # (1.2, .65)
                                  bbox_transform=main_ax.axes.transAxes,
                                  loc="center right", frameon=False)
    res_ax.legend(res_ax.lines[:-1], [l.get_label() for l in res_ax.lines[:-1]],
                  bbox_to_anchor=(1.2, .8), # (1.2, .8)
                  bbox_transform=res_ax.axes.transAxes, # res_ax.axes.transAxes
                  loc="center right", frameon=False)
    
    main_ax.add_artist(first_legend)
    
    # fig.set_size_inches([16.06,  7.17]) #[14.22,  7.17]
    fig.set_size_inches([14,  4.98])
    
see_x_probe(0,6)
    
#%% NOISE VS X PLOT

fig, axes = plt.subplots(nrows=len(series), sharex=True, sharey=True, 
                         gridspec_kw={"hspace":0})

plt.suptitle(plot_title_base + trs.choose(r": Residua $\sigma$", ": $\sigma$ residuos"))

lines = []
for i in range(len(series)):
    for j in range(len(series[i])):
        l, = axes[i].plot(x_probe_position_factor[i][j], 
                          x_probe_fit_res_std[i][j], 
                          "o-", alpha=0.7, color=colors[i][j], markeredgewidth=0)
                          # label=series_label[1](series[i][j]))
        lines.append(l)
axes[-1].set_ylabel(trs.choose("Electric\n Field " + r"$E_z(y=z=0)$",
                               "Campo eléctrico \n " + r"$E_z(y=z=0)$"))

fig.set_size_inches([10.28,  4.8 ])

for ax in axes:
    box = ax.get_position()
    box.x1 = box.x1 - .2 * (box.x1 - box.x0)
    ax.set_position(box)
axes[-1].set_xlabel(trs.choose("Position $X$ [Cell]", "Posición $X$ [Celda]"))

legend_labels = []
for i in range(len(series)):
    for j in range(len(series[i])):
        if i!=len(series)-1:
            legend_labels.append(" ")
        else:
            legend_labels.append(series_label[i](series[i][j]))

plt.legend(lines, legend_labels,
           columnspacing=-0.5, ncol=len(series), bbox_to_anchor=(1.35, 1.5), 
           loc="center right", frameon=False)

vs.saveplot(plot_file("NoiseVsX.png"), overwrite=True)

#%% NOISE AT X0, XF VS RESOLUTION PLOT

plt.figure()

plt.title(plot_title_base + trs.choose(r": Residua $\sigma$", ": $\sigma$ residuos"))

for i in range(len(series)):
    plt.plot(test_param[i], 
              [x_probe_fit_res_std[i][j][0] for j in range(len(series[i]))],
              "o", color=colorConverter.to_rgba(series_colors[i], alpha=0.7), 
              fillstyle="none", markersize=8, markeredgewidth=1.5,
              label=fr"{series_legend[i]} x = {x_probe_position_factor[i][j][0]} $\lambda/n$")
    plt.plot(test_param[i], 
             [x_probe_fit_res_std[i][j][-1] for j in range(len(series[i]))],
             "o", color=series_colors[i], 
             alpha=0.4, markeredgewidth=0, markersize=8,
             label=fr"{series_legend[i]} x = {x_probe_position_factor[i][j][-1]} $\lambda/n$")

plt.legend()

plt.xlabel(test_param_label)
plt.ylabel(trs.choose("Electric Field Noise Amplitude\n", 
                      "Amplitud de ruido en campo eléctrico\n") + 
           trs.choose(r"${E_z}^{noise}(y=z=0)$", r"${E_z}^{ruido}(y=z=0)$") )
plt.tight_layout()

vs.saveplot(plot_file("NoiseVsXVsResolution.png"), overwrite=True)

#%% NOISE DIFFERENCE XF-X0 VS RESOLUTION PLOT

plt.figure()

plt.title(plot_title_base + trs.choose(r": Residua $\sigma$", ": $\sigma$ residuos"))

for i in range(len(series)):
    plt.plot(test_param[i], 
             [np.mean([x_probe_fit_res_std[i][j][-1] - x_probe_fit_res_std[i][j][0],
                       x_probe_fit_res_std[i][j][-1] - x_probe_fit_res_std[i][j][1],
                       x_probe_fit_res_std[i][j][-2] - x_probe_fit_res_std[i][j][1],
                       x_probe_fit_res_std[i][j][-2] - x_probe_fit_res_std[i][j][0]])
              for j in range(len(series[i]))],
             "o", color=series_colors[i], 
             alpha=0.4, markeredgewidth=0, markersize=8,
             label=series_legend[i])

plt.legend()

plt.xlabel(test_param_label)
plt.ylabel(trs.choose("Electric Field Noise Difference\n", 
                      "Diferencia de ruido en campo eléctrico\n") + 
           trs.choose(r"${E_z}^{noise}(x=x_f) - {E_z}^{noise}(x=x_0)$",
                      r"${E_z}^{ruido}(x=x_f) - {E_z}^{ruido}(x=x_0)$") )
plt.tight_layout()

vs.saveplot(plot_file("NoiseDifVsXVsResolution.png"), overwrite=True)

#%% FULL NOISE VS X ANALYSIS <<

fig = plt.figure()

plot_grid = gridspec.GridSpec(ncols=3, nrows=len(series), hspace=0, wspace=0.1, figure=fig)
axes = [fig.add_subplot(plot_grid[i,:2]) for i in range(len(series))]
sigma_ax = fig.add_subplot(plot_grid[:,-1:])

lines = [[]]*len(series)
for i in range(len(series)):
    
    for j in range(len(series[i])):
        l, = axes[i].plot(x_probe_position_factor[i][j], 
                          x_probe_fit_res_std[i][j], 
                          "o-", alpha=0.7, color=colors[i][j], markeredgewidth=0)
                          # label=series_label[1](series[i][j]))
        lines[i].append(l)
    axes[i].axvline(x_probe_position_factor[i][j][0], linestyle=(0,(5,10)), linewidth=0.5, color="k")
    axes[i].axvline(x_probe_position_factor[i][j][-1], linestyle=(0,(5,10)), linewidth=0.5, color="k")
    
    sigma_ax.plot(test_param[i], 
                  [x_probe_fit_res_std[i][j][0] for j in range(len(series[i]))],
                  "o", color=colorConverter.to_rgba(series_colors[i], alpha=0.7), 
                  fillstyle="none", markersize=8, markeredgewidth=1.5,
                  label=fr"x = {x_probe_position_factor[i][j][0]} $\lambda/n$")
    sigma_ax.plot(test_param[i], 
                  [x_probe_fit_res_std[i][j][-1] for j in range(len(series[i]))],
                  "o", color=series_colors[i], 
                  alpha=0.4, markeredgewidth=0, markersize=8,
                  label=fr"x = {x_probe_position_factor[i][j][-1]} $\lambda/n$")

for ax in fig.axes:
    box = ax.get_position()
    width = box.x1 - box.x0
    box.x0 = box.x0 + .1 * width
    box.x1 = box.x0 + width
    ax.set_position(box)    
for ax in axes:
    box = ax.get_position()
    box.x0 = box.x0 + .13 * (box.x1 - box.x0)
    ax.set_position(box)
axes[-1].set_xlabel(trs.choose(r"Position $X$ [$\lambda/n$]", r"Posición $X$ [$\lambda/n$]"))
for ax in axes:
    ax.set_ylabel(trs.choose("Electric\n Field " + r"$E_z(y=z=0)$",
                             "Campo eléctrico \n " + r"$E_z(y=z=0)$"))

sigma_ax.set_xlabel(test_param_label)
sigma_ax.set_ylabel(trs.choose("Electric Field Noise Amplitude\n", 
                               "Amplitud de ruido en campo eléctrico\n") + 
                    trs.choose(r"${E_z}^{noise}(y=z=0)$", r"${E_z}^{ruido}(y=z=0)$") )
sigma_ax.yaxis.tick_right()
sigma_ax.yaxis.set_label_position("right")

# axes[0].legend([*lines[0], *lines[1], *lines[2]], 
#                [*[" "]*(len(series[0])+len(series[1])), *[series_label[2](series[2][j]) for j in range(len(series[2]))]],
axes[0].legend([*lines[0], *lines[1]], 
               [*[r"1p$\lambda$"]*len(series[0]), *[series_label[1](series[1][j]) for j in range(len(series[1]))]],
               columnspacing=0.3, ncol=len(series), bbox_to_anchor=(-2.1, 0.5), loc="center right", frameon=False,
               bbox_transform=sigma_ax.axes.transAxes)
sigma_ax.legend(frameon=True, facecolor="white", edgecolor="black",
                bbox_to_anchor=(0.4, 0.52), bbox_transform=sigma_ax.axes.transAxes)

fig.set_size_inches([13.76,  4.66])

#%% ANALYSE X AXIS FOR DIFFERENT TIMES VIA FIT AND RESIDUA <<

n_t_probe = 20

start_time = 1
start_point = [[np.where(t_line[i][j] / period_results[i][j] >= start_time)[0][0] for j in range(len(series[i]))] for i in range(len(series))]
end_time = [[start_time + (time_period_factor[i][j]-1) for j in range(len(series[i]))] for i in range(len(series))]
end_point = [[np.where(t_line[i][j] / period_results[i][j] <= end_time[i][j])[0][-1] for j in range(len(series[i]))] for i in range(len(series))]

probe_index = [[ [int(k) for k in np.linspace(start_point[i][j], end_point[i][j], n_t_probe+1)][:-1] for j in range(len(series[i]))] for i in range(len(series))]
# probe_index = [[ [int(k) for k in np.linspace(start_point[i][j], end_point[i][j], n_t_probe+1)][:-1] for j in range(len(series[i]))] for i in range(len(series))]

# zero_time = []
# for i in range(len(series)):
#     zero_time.append([])
#     for j in range(len(series[i])):
#         if wlen_in_vacuum[i][j]:
#             zero_time[i].append(index[i][j])
#         else:
#             zero_time[i].append(1)
# zero_point = [[np.where(t_line[i][j] / period_results[i][j] >= zero_time[i][j])[0][0] for j in range(len(series[i]))] for i in range(len(series))]

# probe_index = [[ [zero_point[i][j],
#                  *[int(k) for k in np.linspace(start_point[i][j], len(t_line[i][j])-1, n_t_probe-1)]] 
#                  for j in range(len(series[i]))] for i in range(len(series))]

t_probe_field = [[ [cropped_line[i][j][:, k] for k in probe_index[i][j]] for j in range(len(series[i]))] for i in range(len(series))]

t_probe_time = [[ [t_line[i][j][k] for k in probe_index[i][j]] for j in range(len(series[i]))] for i in range(len(series))]

# def fit_function_generator(wavelength, index):
#     def fit_function(position, amplitude, phase, offset):
#         medium_wavelength = wavelength / index
#         medium_wave_number = 2 * np.pi / medium_wavelength
#         return amplitude * np.cos( medium_wave_number * position + phase ) + offset
#     return fit_function
# t_probe_fit_functions = [[fit_function_generator(wlen[i][j], index[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

def t_probe_fit_function(position, amplitude, wave_number, phase, offset):
    return amplitude * np.cos( wave_number * position + phase ) + offset

t_probe_fit_amplitude = []
t_probe_fit_wave_number = []
t_probe_fit_phase = []
t_probe_fit_offset = []
t_probe_fit_residua = []
for i in range(len(series)):
    t_probe_fit_amplitude.append([])
    t_probe_fit_wave_number.append([])
    t_probe_fit_phase.append([])
    t_probe_fit_offset.append([])
    t_probe_fit_residua.append([])
    for j in range(len(series[i])):
        t_probe_fit_amplitude[i].append([])
        t_probe_fit_phase[i].append([])
        t_probe_fit_wave_number[i].append([])
        t_probe_fit_offset[i].append([])
        t_probe_fit_residua[i].append([])
        for k in range(n_t_probe):
            rsq, fit_params = va.nonlinear_fit(x_line_cropped[i][j],
                                           t_probe_field[i][j][k],
                                           t_probe_fit_function,
                                           initial_guess=(1, 2 * np.pi * index[i][j] / wlen[i][j], 0, 0),
                                           showplot=False)
            amplitude, wave_number, phase, offset = fit_params[0][0], fit_params[1][0], fit_params[2][0], fit_params[3][0]
            t_probe_fit_amplitude[i][j].append( amplitude )
            t_probe_fit_wave_number[i][j].append( wave_number )
            t_probe_fit_phase[i][j].append( phase )
            t_probe_fit_offset[i][j].append( offset )
            
            residua = np.array(t_probe_field[i][j][k]) - t_probe_fit_function( x_line_cropped[i][j], amplitude, wave_number, phase, offset)
            
            t_probe_fit_residua[i][j].append( residua )

t_probe_fit_res_std = [[ [np.std(t_probe_fit_residua[i][j][k]) for k in range(n_t_probe)] for j in range(len(series[i]))] for i in range(len(series))]

#%% T PROBE PLOT <<

# see_t_probe(0,6) a ver, tirá fachaaa con n_t_period=8 demostrativo, con nt+1[:-1]
def see_t_probe(i,j):
    
    colors = plab.cm.jet(np.linspace(0,1,n_t_probe+2)[1:-1]) #jet
    
    fig = plt.figure()
    
    plot_grid = gridspec.GridSpec(ncols=1, nrows=6, hspace=0, figure=fig)
    main_ax = fig.add_subplot(plot_grid[:4,:])
    res_ax = fig.add_subplot(plot_grid[-2:,:])
    
    plt.suptitle(f"{series_legend[i]} {series_label[1](series[i][j])}")
    for k in range(n_t_probe):
        main_ax.plot(x_line_cropped[i][j] / (cell_width[i][j] - 2*pml_width[i][j]),
                     t_probe_field[i][j][k], 
                     color=colors[k],
                     label=f"t = {t_probe_time[i][j][k] / norm_period[i][j]:.2f}" + r" $\tau$")
        main_ax.plot(x_line_cropped[i][j] / (cell_width[i][j] - 2*pml_width[i][j]),
                    t_probe_fit_function( x_line_cropped[i][j],
                                          t_probe_fit_amplitude[i][j][k],
                                          t_probe_fit_wave_number[i][j][k],
                                          t_probe_fit_phase[i][j][k],
                                          t_probe_fit_offset[i][j][k] ),
                    linestyle="dashed", color="k", alpha=0.5,)
    
    res_ax.set_xlabel(trs.choose("Position $X$ [$\lambda/n$]", "Posición $X$ [$\lambda/n$]"))
    main_ax.set_ylabel(trs.choose("Electric\n Field " + r"$E_z(y=z=0)$",
                                  "Campo eléctrico \n " + r"$E_z(y=z=0)$"))
    res_ax.set_ylabel(trs.choose("Electric Field\n" + r"$E_z(y=z=0)$ Residua",
                                 "Residuos del\n" + "campo eléctrico\n" + "r""$E_z(y=z=0)$"))
    
    for k in range(n_t_probe):
        res_ax.plot(x_line_cropped[i][j] / (cell_width[i][j] - 2*pml_width[i][j]),
                    t_probe_fit_residua[i][j][k],
                    "o", color=colors[k], markeredgewidth=0, alpha=0.5,
                    # ".", color=colors[k], markeredgewidth=0,
                    label=f"t = {t_probe_time[i][j][k] / norm_period[i][j]:.2f}" + r" $\tau$")
    res_ax.axhline(0, color="k", linewidth=0.5)
    
    for ax in fig.axes:
        box = ax.get_position()
        box.x1 = box.x1 - .2 * (box.x1 - box.x0)
        ax.set_position(box)
    
    first_legend = main_ax.legend(bbox_to_anchor=(1.17, .65), # (1.17, .35)
                                  bbox_transform=main_ax.axes.transAxes,
                                  loc="center right", frameon=False)
    res_ax.legend(res_ax.lines[:-1], [l.get_label() for l in res_ax.lines[:-1]],
                  bbox_to_anchor=(1.17, -.15),  # (1.3, .35)
                  bbox_transform=main_ax.axes.transAxes, # res_ax.axes.transAxes
                  loc="center right", frameon=False)
    
    main_ax.add_artist(first_legend)
    
    # fig.set_size_inches([16.06,  7.17]) #[14.22,  7.17]
    fig.set_size_inches([14.83,  4.98])
    
see_t_probe(0,6)
 
#%% NOISE VS T PLOT

fig, axes = plt.subplots(nrows=len(series), sharex=True, sharey=True, 
                         gridspec_kw={"hspace":0})

plt.suptitle(plot_title_base + trs.choose(r": Residua $\sigma$", ": $\sigma$ residuos"))

lines = [[]]*len(series)
for i in range(len(series)):
    for j in range(len(series[i])):
        l, = axes[i].plot(t_probe_time[i][j] / norm_period[i][j], 
                          t_probe_fit_res_std[i][j], 
                          "o-", alpha=0.7, color=colors[i][j], markeredgewidth=0)
                          # label=series_label[1](series[i][j]))
        lines[i].append(l)

fig.set_size_inches([10.28,  4.8 ])

for ax in axes:
    box = ax.get_position()
    box.x1 = box.x1 - .2 * (box.x1 - box.x0)
    ax.set_position(box)
axes[-1].set_xlabel(trs.choose("Time T [MPu]", "Tiempo T [uMP]"))
axes[-1].set_ylabel(trs.choose("Electric\n Field " + r"$E_z(y=z=0)$",
                               "Campo eléctrico \n " + r"$E_z(y=z=0)$"))

plt.legend([*lines[0], *lines[1], *lines[2]], 
           [*[" "]*(len(series[0])+len(series[1])), *[series_label[2](series[2][j]) for j in range(len(series[2]))]],
    columnspacing=-0.5, ncol=len(series), bbox_to_anchor=(1.35, 1), loc="center right", frameon=False)

vs.saveplot(plot_file("NoiseVsT.png"), overwrite=True)


#%% NOISE AT T0, TF VS RESOLUTION PLOT

plt.figure()

plt.title(plot_title_base + trs.choose(r": Residua $\sigma$", ": $\sigma$ residuos"))

for i in range(len(series)):
    plt.plot(test_param[i], 
              [t_probe_fit_res_std[i][j][0] for j in range(len(series[i]))],
              "o", color=colorConverter.to_rgba(series_colors[i], alpha=0.7), 
              fillstyle="none", markersize=8, markeredgewidth=1.5,
              label=f"{series_legend[i]} t = {t_probe_time[i][j][0] / norm_period[i][j] :.2f}" + r" $\tau$")
    plt.plot(test_param[i], 
             [t_probe_fit_res_std[i][j][-1] for j in range(len(series[i]))],
             "o", color=series_colors[i], 
             alpha=0.4, markeredgewidth=0, markersize=8,
             label=f"{series_legend[i]} t = {t_probe_time[i][j][-1] / norm_period[i][j] :.2f}" + r" $\tau$")

plt.legend()

plt.xlabel(test_param_label)
plt.ylabel(trs.choose("Electric Field Noise Amplitude\n", 
                      "Amplitud de ruido en campo eléctrico\n") + 
           trs.choose(r"${E_z}^{noise}(y=z=0)$", r"${E_z}^{ruido}(y=z=0)$") )
plt.tight_layout()

vs.saveplot(plot_file("NoiseVsTVsResolution.png"), overwrite=True)

#%% FULL NOISE VS T ANALYSIS <<

fig = plt.figure()

plot_grid = gridspec.GridSpec(ncols=3, nrows=len(series), hspace=0, wspace=0.1, figure=fig)
axes = [fig.add_subplot(plot_grid[i,:2]) for i in range(len(series))]
sigma_ax = fig.add_subplot(plot_grid[:,-1:])

lines = [[]]*len(series)
for i in range(len(series)):
    
    for j in range(len(series[i])):
        
        l, = axes[i].plot(t_probe_time[i][j] / norm_period[i][j], 
                          t_probe_fit_res_std[i][j], 
                          "o-", alpha=0.7, color=colors[i][j], markeredgewidth=0)
                          # label=series_label[1](series[i][j]))
        lines[i].append(l)
    axes[i].axvline(t_probe_time[i][j][0] / norm_period[i][j], linestyle=(0,(5,10)), linewidth=0.5, color="k")
    axes[i].axvline(t_probe_time[i][j][-1] / norm_period[i][j], linestyle=(0,(5,10)), linewidth=0.5, color="k")
    
    sigma_ax.plot(test_param[i], 
                  [t_probe_fit_res_std[i][j][0] for j in range(len(series[i]))],
                  "o", color=colorConverter.to_rgba(series_colors[i], alpha=0.7), 
                  fillstyle="none", markersize=8, markeredgewidth=1.5,
                  label=f"t = {t_probe_time[i][j][0] / norm_period[i][j] :.2f}" + r" $\tau$")
    sigma_ax.plot(test_param[i], 
                  [t_probe_fit_res_std[i][j][-1] for j in range(len(series[i]))],
                  "o", color=series_colors[i], 
                  alpha=0.4, markeredgewidth=0, markersize=8,
                  label=f"t = {t_probe_time[i][j][-1] / norm_period[i][j] :.2f}" + r" $\tau$")

for ax in fig.axes:
    box = ax.get_position()
    width = box.x1 - box.x0
    box.x0 = box.x0 + .1 * width
    box.x1 = box.x0 + width
    ax.set_position(box)    
for ax in axes:
    box = ax.get_position()
    box.x0 = box.x0 + .13 * (box.x1 - box.x0)
    ax.set_position(box)
axes[-1].set_xlabel(trs.choose("Time T [MPu]", "Tiempo T [uMP]"))
for ax in axes:
    ax.set_ylabel(trs.choose("Electric\n Field " + r"$E_z(y=z=0)$",
                             "Campo eléctrico \n " + r"$E_z(y=z=0)$"))

sigma_ax.set_xlabel(test_param_label)
sigma_ax.set_ylabel(trs.choose("Electric Field Noise Amplitude\n", 
                               "Amplitud de ruido en campo eléctrico\n") + 
                    trs.choose(r"${E_z}^{noise}(y=z=0)$", r"${E_z}^{ruido}(y=z=0)$") )
sigma_ax.yaxis.tick_right()
sigma_ax.yaxis.set_label_position("right")

# axes[0].legend([*lines[0], *lines[1], *lines[2]], 
#                [*[" "]*(len(series[0])+len(series[1])), *[series_label[2](series[2][j]) for j in range(len(series[2]))]],
axes[0].legend([*lines[0], *lines[1]], 
               [*[r"1p$\lambda$"]*len(series[0]), *[series_label[1](series[1][j]) for j in range(len(series[1]))]],
               columnspacing=0.3, ncol=len(series), bbox_to_anchor=(-2.1, 0.5), loc="center right", frameon=False,
               bbox_transform=sigma_ax.axes.transAxes)
sigma_ax.legend(frameon=True, facecolor="white", edgecolor="black",
                bbox_to_anchor=(0.4, 0.52), bbox_transform=sigma_ax.axes.transAxes)

fig.set_size_inches([13.76,  4.66])

#%%

# x_field_integral = np.sum(results_line[i][j], axis=-1) * np.mean(np.diff(t_line[i][j]))
# x_field_integral_left = x_field_integral[:x_line_index[i][j](-cell_width[i][j]/2 + pml_width[i][j])][::-1]
# x_field_integral_right = x_field_integral[x_line_index[i][j](+cell_width[i][j]/2 - pml_width[i][j])+1:]

# find_peaks( np.abs( x_field_integral_left - norm_amplitude[i][j] / np.e ) )

zcropped_planes = [[results_plane[i][j][:, : z_plane_index[i][j](cell_width[i][j]/2 - pml_width[i][j]) + 1, ...] for j in range(len(series[i]))] for i in range(len(series))]
zcropped_planes = [[zcropped_planes[i][j][:, z_plane_index[i][j](-cell_width[i][j]/2 + pml_width[i][j]) :, ...] for j in range(len(series[i]))] for i in range(len(series))]

y_field_integral = np.mean(np.abs(zcropped_planes[i][j]), axis=-1) #* np.mean(np.diff(t_line[i][j]))
y_field_integral = np.mean(np.abs(y_field_integral), axis=-1) #* np.mean(np.diff(z_line[i][j]))
# x_field_integral_left = x_field_integral[:x_line_index[i][j](-cell_width[i][j]/2 + pml_width[i][j])][::-1]
# x_field_integral_right = x_field_integral[x_line_index[i][j](+cell_width[i][j]/2 - pml_width[i][j])+1:]

# find_peaks( np.abs( x_field_integral_left - norm_amplitude[i][j] / np.e ) )

#%%

def see_xt_axis(i, j):
    
    plt.figure()
    plt.title(f"{series_legend[i]} {series_label[1](series[i][j])}")
    
    T, X = np.meshgrid(t_line[i][j], x_line_cropped[i][j])
    plt.contourf(T, X, cropped_line[i][j], 100, cmap='RdBu')
    plt.xlabel(trs.choose("Time [MPu]", "Tiempo [uMP]"))
    plt.ylabel(trs.choose("Position $X$ [MPu]", "Posición $X$ [uMP]"))
    
    plt.figure()    
    plt.title(f"{series_legend[i]} {series_label[1](series[i][j])}")    
    
    T, X = np.meshgrid(t_line[i][j], x_line[i][j])
    plt.contourf(T, X, results_line[i][j], 100, cmap='RdBu')
    xlims = plt.xlim()
    plt.hlines(-cell_width[i][j]/2 + pml_width[i][j], *xlims, 
               color="k", linestyle="dashed")
    plt.hlines(cell_width[i][j]/2 - pml_width[i][j], *xlims, 
               color="k", linestyle="dashed")
    plt.xlim(*xlims)
    plt.xlabel(trs.choose("Time [MPu]", "Tiempo [uMP]"))
    plt.ylabel(trs.choose("Position $X$ [MPu]", "Posición $X$ [uMP]"))
    
#%%

def see_x_axis_in_t(i, j, t):
    
    k = np.argmin( np.abs(t_line[i][j] - t) )
    
    plt.figure()
    plt.title(f"{series_legend[i]} {series_label[1](series[i][j])}")
    plt.axhline(color="k", linewidth=0.5)
    plt.axvline(color="k", linewidth=0.5)
    plt.plot(x_line[i][j], results_line[i][j][..., k], linewidth=1.5)
    plt.xlim(min(x_line[i][j]), max(x_line[i][j]))
    plt.xlabel(trs.choose("Position $X$ [MPu]", "Posición $X$ [uMP]"))
    plt.ylabel(trs.choose("Electric field $E_z$", "Campo eléctrico $E_z$"))
    
    plt.axvline(-cell_width[i][j]/2 + pml_width[i][j], 
                color="k", linestyle="dashed", linewidth=1)
    plt.axvline(cell_width[i][j]/2 - pml_width[i][j], 
                color="k", linestyle="dashed", linewidth=1)
        
    plt.figure()
    plt.title(f"{series_legend[i]} {series_label[1](series[i][j])}")
    plt.axhline(color="k", linewidth=0.5)
    plt.axvline(color="k", linewidth=0.5)
    plt.plot(x_line_cropped[i][j], cropped_line[i][j][..., k], linewidth=1.5)
    plt.xlim(min(x_line_cropped[i][j]), max(x_line_cropped[i][j]))
    plt.xlabel(trs.choose("Position $X$ [MPu]", "Posición $X$ [uMP]"))
    plt.ylabel(trs.choose("Electric field $E_z$", "Campo eléctrico $E_z$"))

#%%

def see_yz_plane_in_t(i, j, t):
    
    k = np.argmin( np.abs(t_plane[i][j] - t) )
    
    fig = plt.figure()
    plt.title(f"{series_legend[i]} {series_label[1](series[i][j])}")
    ax = plt.subplot()
    ax.set_aspect('equal')
    lims = (np.min(cropped_plane[i][j]), np.max(cropped_plane[i][j]))
    lims = max([abs(l) for l in lims])
    lims = [-lims, lims]

    ims = ax.imshow(cropped_plane[i][j][...,k].T,
                    cmap='RdBu', #interpolation='spline36', 
                    vmin=lims[0], vmax=lims[1],
                    extent=[min(y_plane_cropped[i][j]), max(y_plane_cropped[i][j]),
                            min(z_plane_cropped[i][j]), max(z_plane_cropped[i][j])])
    plt.grid(False)
    
    ax.text(-.1, -.105, f"Time t = {t:.2f}" + r" $\tau$", transform=ax.transAxes)
    plt.show()
    plt.xlabel(trs.choose("Position $Y$ [MPu]", "Posición $Y$ [uMP]"))
    plt.ylabel(trs.choose("Position $Z$ [MPu]", "Posición $Z$ [uMP]"))
    if use_units:
        plt.annotate(trs.choose(f"1 Meep Unit = {from_um_factor * 1e3:.0f} nm",
                                f"1 Unidad de Meep = {from_um_factor * 1e3:.0f} nm"),
                     (300, 11), xycoords='figure points') # 50, 300
    else:
        plt.annotate(trs.choose(r"1 Meep Unit = $\lambda$",
                                r"1 Unidad de Meep = $\lambda$"),
                     (300, 11), xycoords='figure points') # 50, 310
    
    cax = ax.inset_axes([1.04, 0, 0.07, 1], #[1.04, 0.2, 0.05, 0.6], 
                        transform=ax.transAxes)
    cbar = fig.colorbar(ims, ax=ax, cax=cax)
    cbar.set_label(trs.choose("Electric field $E_z$",
                              "Campo eléctrico $E_z$"))

    fig = plt.figure()
    plt.title(f"{series_legend[i]} {series_label[1](series[i][j])}")
    ax = plt.subplot()
    ax.set_aspect('equal')
    lims = (np.min(results_plane[i][j]), np.max(results_plane[i][j]))
    lims = max([abs(l) for l in lims])
    lims = [-lims, lims]  
    
    k = np.argmin( np.abs(t_plane[i][j] - t) )

    ims = ax.imshow(results_plane[i][j][...,k].T,
                    cmap='RdBu', #interpolation='spline36', 
                    vmin=lims[0], vmax=lims[1],
                    extent=[min(y_plane[i][j]), max(y_plane[i][j]),
                            min(z_plane[i][j]), max(z_plane[i][j])])

    plt.axvline(-cell_width[i][j]/2 + pml_width[i][j], 
                color="k", linestyle="dashed", linewidth=1)
    plt.axvline(cell_width[i][j]/2 - pml_width[i][j], 
                color="k", linestyle="dashed", linewidth=1)
    plt.grid(False)
    
    ax.text(-.1, -.105, f"Time t = {t:.2f}" + r" $\tau$", transform=ax.transAxes)
    plt.show()
    plt.xlabel(trs.choose("Position $Y$ [MPu]", "Posición $Y$ [uMP]"))
    plt.ylabel(trs.choose("Position $Z$ [MPu]", "Posición $Z$ [uMP]"))
    if use_units:
        plt.annotate(trs.choose(f"1 Meep Unit = {from_um_factor * 1e3:.0f} nm",
                                f"1 Unidad de Meep = {from_um_factor * 1e3:.0f} nm"),
                     (300, 11), xycoords='figure points') # 50, 300
    else:
        plt.annotate(trs.choose(r"1 Meep Unit = $\lambda$",
                                r"1 Unidad de Meep = $\lambda$"),
                     (300, 11), xycoords='figure points') # 50, 310
    
    cax = ax.inset_axes([1.04, 0, 0.07, 1], #[1.04, 0.2, 0.05, 0.6], 
                        transform=ax.transAxes)
    cbar = fig.colorbar(ims, ax=ax, cax=cax)
    cbar.set_label(trs.choose("Electric field $E_z$",
                              "Campo eléctrico $E_z$"))
