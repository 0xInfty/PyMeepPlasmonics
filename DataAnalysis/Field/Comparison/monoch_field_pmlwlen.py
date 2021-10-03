#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:45:00 2020

@author: vall
"""

# Monochromatic planewave field

from socket import gethostname
if "Nano" in gethostname():
    syshome = "/home/nanofisica/Documents/Vale/ThesisPython"
elif "vall" in gethostname():
    syshome = "/home/vall/Documents/Thesis/ThesisPython"
else:
    raise ValueError("Your PC must be registered at the top of this code")

import sys
sys.path.append(syshome)

import imageio as mim
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plab
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import colorConverter
import os
import v_analysis as va
import v_materials as vmt
import v_meep as vm
import v_meep_analysis as vma
import v_plot as vp
import v_theory as vt
import v_save as vs
import v_utilities as vu

english = False
trs = vu.BilingualManager(english=english)
vp.set_style()

#%% PARAMETERS

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
sorting_function = [lambda l : vu.sort_by_number(l, test_param_position)]*2
series_label = [lambda s : r" ",
                lambda s : rf"PML {vu.find_numbers(s)[test_param_position]:.2f} $\lambda$"]
series_must = [""]*2 # leave "" per default
series_mustnt = ["Fail"]*2 # leave "" per default

# Scattering plot options
plot_title_base = trs.choose('Dimnesionless monochromatic wave', 
                             "Onda monocromática adimensional")
series_legend = trs.choose(["Vacuum", "Water"], ["Vacío", "Agua"])
series_colormaps = [plab.cm.Reds, plab.cm.Blues, plab.cm.YlGn]
series_colors = ["red", "blue", "limegreen"]
series_markers = ["o", "o", "o"]
series_linestyles = ["solid"]*3
plot_make_big = False
plot_file = lambda n : os.path.join(home, "DataAnalysis/Field/Sources/MonochPlanewave/TestPMLwlen/TestPMLwlen" + n)

#%% LOAD DATA

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
files_plane = [[h5.File(file[i](series[i][j], "Field-Lines.h5"), "r") 
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

#%% POSITION RECONSTRUCTION

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

#%% DATA EXTRACTION

source_results = [[vma.get_source_from_line(results_line[i][j], x_line_index[i][j], source_center[i][j]) 
                   for j in range(len(series[i]))] for i in range(len(series))]

if not requires_normalization:
    
    period_results, amplitude_results = norm_period, norm_amplitude
    
else:
    
    period_results = [[vma.get_period_from_source(source_results[i][j], t_line[i][j],
                                                  periods_sensitivity=0.06) 
                       for j in range(len(series[i]))] for i in range(len(series))]
    amplitude_results = [[vma.get_amplitude_from_source(source_results[i][j],
                                                        amplitude_sensitivity=0.05) 
                          for j in range(len(series[i]))] for i in range(len(series))]
    
    source_results = [[source_results[i][j] / amplitude_results[i][j] 
                       for j in range(len(series[i]))] for i in range(len(series))]
    results_plane = [[np.asarray(results_plane[i][j]) / amplitude_results[i][j] 
                      for j in range(len(series[i]))] for i in range(len(series))]
    results_line = [[np.asarray(results_line[i][j]) / amplitude_results[i][j] 
                     for j in range(len(series[i]))] for i in range(len(series))]
    
    norm_period, norm_amplitude = period_results, amplitude_results

#%% GENERAL PLOT CONFIGURATION

colors = [sc(np.linspace(0,1,len(s)+2))[2:] 
          for sc, s in zip(series_colormaps, series)]

#%% BASIC CONTROL

plt.figure()
for i in range(len(series)):        
    plt.plot(test_param[i],
             [results_line[i][j].shape[0] for j in range(len(series[i]))], "o", color=series_colors[i], alpha=0.5)
plt.xlabel(test_param_label)
plt.ylabel(trs.choose("Number of points in whole cell", "Número de puntos en la celda completa"))
plt.legend(series_legend)

plt.savefig(plot_file("Points.png"))

cropped_line = [[vma.crop_field_xprofile(results_line[i][j], x_line_index[i][j], 
                                         cell_width[i][j], pml_width[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

plt.figure()
for i in range(len(series)):        
    plt.plot(test_param[i],
             [cropped_line[i][j].shape[0] for j in range(len(series[i]))], "o", color=series_colors[i], alpha=0.5)
plt.xlabel(test_param_label)
plt.ylabel(trs.choose("Number of points in real cell", "Número de puntos en la celda real"))
plt.legend(series_legend)

plt.savefig(plot_file("InnerPoints.png"))

plt.figure()
for i in range(len(series)):
    plt.plot(test_param[i],
             [results_line[i][j].shape[-1] for j in range(len(series[i]))], "o", color=series_colors[i], alpha=0.5)
plt.xlabel(test_param_label)
plt.ylabel(trs.choose("Number of points in time", "Número de puntos en el tiempo"))
plt.legend(series_legend)

plt.savefig(plot_file("TimePoints.png"))

fig = plt.figure()
ax = plt.subplot()
ax2 = plt.twinx()
lines, lines2 = [], []
for i in range(len(series)):
    l, = ax.plot(test_param[i],
                 [params[i][j]["courant"]/params[i][j]["resolution"] for j in range(len(series[i]))], "o", color=series_colors[i], alpha=0.5)
    l2, = ax.plot(test_param[i],
                  [1/params[i][j]["resolution"] for j in range(len(series[i]))], "o", color=series_colors[i], fillstyle="none")
    lines.append(l)
    lines2.append(l2)
plt.xlabel(test_param_label)
ax.set_ylabel(trs.choose("Time Minimum Division [MPu]", "Mínima división del tiempo [uMP]"))
ax2.set_ylabel(trs.choose("Space Minimum Division [MPu]", "Space división del tiempo [uMP]"))
plt.legend([*lines, *lines2], [*[s + r" $\Delta t$" for s in series_legend], 
                               *[s + r" $\Delta r$" for s in series_legend]])
plt.savefig(plot_file("MinimumDivision.png"))

plt.figure()
for i in range(len(series)):
    plt.plot(test_param[i],
             [params[i][j]["resolution"] for j in range(len(series[i]))], "o", color=series_colors[i], alpha=0.5)
plt.xlabel(test_param_label)
plt.ylabel(trs.choose("Resolution", "Resolución") + r" [points/$\Delta r$]")
# plt.ylabel(trs.choose("Number of points in time", "Número de puntos en el tiempo"))
plt.legend(series_legend)

#%% SHOW SOURCE AND FOURIER USED FOR NORMALIZATION

fig = plt.figure()
plt.title(plot_title_base)

series_lines = []
for i in range(len(series)):
    for j in range(len(series[i])):
        l, = plt.plot(t_line[i][j] / period_results[i][j], 
                      source_results[i][j],
                      label=series_label[i](series[i][j]),
                      color=colors[i][j])            
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
        
fourier = [[np.abs(np.fft.rfft(source_results[i][j])) for j in range(len(series[i]))] for i in range(len(series))]
fourier_freq = [[np.fft.rfftfreq(len(source_results[i][j]), d=period_line[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
if use_units:
    fourier_wlen = [[from_um_factor[i][j] * 1e3 / fourier_freq[i][j] for j in range(len(series[i]))] for i in range(len(series))]
    fourier_best = [[wlen[i][j] * from_um_factor[i][j] * 1e3 for j in range(len(series[i]))] for i in range(len(series))]
else:
    fourier_wlen = [[1 / fourier_freq[i][j]  for j in range(len(series[i]))] for i in range(len(series))]
    fourier_best = [[wlen[i][j] for j in range(len(series[i]))] for i in range(len(series))]

fig = plt.figure()
plt.title(plot_title_base)

series_lines = []
for i in range(len(series)):
    for j in range(len(series[i])):
        l, = plt.plot(fourier_wlen[i][j], fourier[i][j],
                      label=series_label[i](series[i][j]),
                      color=colors[i][j])
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

plt.savefig(plot_file("SourceFFT.png"))

if use_units: plt.xlim([350, 850])
else: plt.xlim([0, 2])
        
plt.savefig(plot_file("SourceFFTZoom.png"))

#%%

fourier_max_wlen = [[fourier_wlen[i][j][ np.argmax(fourier[i][j]) ]  for j in range(len(series[i]))] for i in range(len(series))]
fourier_max_best = [[fourier_wlen[i][j][ np.argmin(np.abs(fourier_wlen[i][j] - fourier_best[i][j])) ]  for j in range(len(series[i]))] for i in range(len(series))]

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

#%%

peaks_index = [[vma.get_peaks_from_source(source_results[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
peaks_heights = [[source_results[i][j][peaks_index[i][j]] for j in range(len(series[i]))] for i in range(len(series))]
peaks_times = [[t_line[i][j][peaks_index[i][j]] for j in range(len(series[i]))] for i in range(len(series))]
peaks_periods = [[np.diff(peaks_times[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

# plt.figure()
# plt.suptitle(plot_title_base)

# lines = []
# for i in range(len(series)):
#     for j in range(len(series[i])):
#         l, = plt.plot(t_line[i][j] / period_results[i][j], 
#                       source_results[i][j],
#                       label=series_label[i](series[i][j]),
#                       color=colors[i][j])            
#         l, = plt.plot(t_line[i][j][peaks_index[i][j]] / period_results[i][j], 
#                       source_results[i][j][peaks_index[i][j]], "o",
#                       label=series_label[i](series[i][j]),
#                       color=colors[i][j])       
#         lines.append(l)
# plt.xlabel(trs.choose("Time in multiples of period", "Tiempo en múltiplos del período"))
# plt.ylabel(trs.choose(r"Electric Field $E_z(y=z=0)$",
#                       r"Campo eléctrico $E_z(y=z=0)$"))
# plt.legend(ncol=2)

#%%

peaks_height_variation = [[100 * ( max(peaks_heights[i][j]) - min(peaks_heights[i][j]) ) / min(peaks_heights[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

plt.figure()
plt.title(plot_title_base)
for i in range(len(series)):
    plt.plot(test_param[i], peaks_height_variation[i], 
             color=series_colors[i], marker=series_markers[i], alpha=0.4,
             markersize=8, linestyle="", markeredgewidth=0)
plt.legend(series_legend)
plt.xlabel(test_param_label)
plt.ylabel(trs.choose("Maximum percentual variation in amplitude ", 
                      "Diferencia porcentual máxima en amplitud ") + 
           r"$\max[ E_z(y=z=0) ]$ [%]")
           # r"$\frac{\max|E^{max}|_i - \min|E^{max}|_i}{\min|E^{max}|_i}$")
vs.saveplot(plot_file("AmpVariation.png"), overwrite=True)

#%%

peaks_period_variation = [[100 * ( max(peaks_periods[i][j]) - min(peaks_periods[i][j]) ) / min(peaks_periods[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

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

#%%

x_probe_fourier = [[ [np.abs(np.fft.rfft(x_probe_fourier_field[i][j][k], norm="ortho")) for k in range(n_probe)] 
                    for j in range(len(series[i]))] for i in range(len(series))]
x_probe_freqs = [[ [np.fft.rfftfreq(len(x_probe_fourier_field[i][j][k]), d=period_line[i][j]) for k in range(n_probe)] 
                  for j in range(len(series[i]))] for i in range(len(series))]
if use_units:
    x_probe_wlen = [[ [from_um_factor[i][j] * 1e3 / x_probe_freqs[i][j][k] for k in range(n_probe)] for j in range(len(series[i]))] for i in range(len(series))]
else:
    x_probe_wlen = [[ [1 / x_probe_freqs[i][j][k] for k in range(n_probe)] for j in range(len(series[i]))] for i in range(len(series))]

#%%

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
    
#%% ANALYSE X AXIS FOR DIFFERENT POSITIONS VIA FIT AND RESIDUA

n_x_probe = 10

x_probe_position = [[ [-cell_width[i][j]/2 + pml_width[i][j] + k * (cell_width[i][j] - 2*pml_width[i][j]) / (n_x_probe-1) 
                       for k in range(n_x_probe)] for j in range(len(series[i]))] for i in range(len(series))]
x_probe_position_factor = [[ [x_probe_position[i][j][k] / (cell_width[i][j] - 2*pml_width[i][j])
                              for k in range(n_x_probe)] for j in range(len(series[i]))] for i in range(len(series))]
                       
x_probe_field = [[ [results_line[i][j][x_line_index[i][j](x_probe_position[i][j][k]), :]
                    for k in range(n_x_probe)] for j in range(len(series[i]))] for i in range(len(series))]

cropping_index = [[ [np.where(np.abs(x_probe_field[i][j][k]) > 0.05)[0][0] for k in range(n_x_probe)] for j in range(len(series[i]))] for i in range(len(series))]

x_probe_field = [[ [x_probe_field[i][j][k][ cropping_index[i][j][k] : ]  for k in range(n_x_probe)] for j in range(len(series[i]))] for i in range(len(series))]
x_probe_time = [[ [t_line[i][j][ cropping_index[i][j][k] : ]  for k in range(n_x_probe)] for j in range(len(series[i]))] for i in range(len(series))]

#%%

def fit_function_generator(norm_period):
    
    def fit_function(time, amplitude, phase, offset):
        
        omega = 2 * np.pi / norm_period
        
        return amplitude * np.cos( omega * time + phase ) + offset
    
    return fit_function

x_probe_fit_functions = [[fit_function_generator(norm_period[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

x_probe_fit_amplitude = []
x_probe_fit_phase = []
x_probe_fit_offset = []
x_probe_fit_residua = []
for i in range(len(series)):
    x_probe_fit_amplitude.append([])
    x_probe_fit_phase.append([])
    x_probe_fit_offset.append([])
    x_probe_fit_residua.append([])
    for j in range(len(series[i])):
        x_probe_fit_amplitude[-1].append([])
        x_probe_fit_phase[-1].append([])
        x_probe_fit_offset[-1].append([])
        x_probe_fit_residua[-1].append([])
        for k in range(n_x_probe):
            rsq, fit_params = va.nonlinear_fit(x_probe_time[i][j][k],
                                           x_probe_field[i][j][k],
                                           x_probe_fit_functions[i][j],
                                           initial_guess=(1, 0, 0),
                                           showplot=False)
            amplitude, phase, offset = fit_params[0][0], fit_params[1][0], fit_params[2][0]
            x_probe_fit_amplitude[-1][-1].append( amplitude )
            x_probe_fit_phase[-1][-1].append( phase )
            x_probe_fit_offset[-1][-1].append( offset )
            
            residua = np.array(x_probe_field[i][j][k]) - x_probe_fit_functions[i][j]( x_probe_time[i][j][k], amplitude, phase, offset)
            
            x_probe_fit_residua[-1][-1].append( residua )

x_probe_fit_res_std = [[ [np.std(x_probe_fit_residua[i][j][k][x_probe_time[i][j][k]>2]) for k in range(n_x_probe)] for j in range(len(series[i]))] for i in range(len(series))]

#%%

def see_x_probe(i,j):
    
    colors = plab.cm.jet(np.linspace(0,1,n_x_probe))
    
    fig = plt.figure()
    
    plot_grid = gridspec.GridSpec(ncols=1, nrows=5, hspace=0, figure=fig)
    main_ax = fig.add_subplot(plot_grid[:3,:])
    res_ax = fig.add_subplot(plot_grid[-2:,:])
    
    plt.suptitle(f"{series_legend[i]} {series_label[1](series[i][j])}")
    for k in range(n_x_probe):
        main_ax.plot(x_probe_time[i][j][k],
                     x_probe_field[i][j][k], 
                     color=colors[k],
                     label=f"x = {x_probe_position_factor[i][j][k]:.2f} " + trs.choose("cell", "celda"))    
        main_ax.plot(x_probe_time[i][j][k],
                     x_probe_fit_functions[i][j]( x_probe_time[i][j][k],
                                                  x_probe_fit_amplitude[i][j][k],
                                                  x_probe_fit_phase[i][j][k],
                                                  x_probe_fit_offset[i][j][k] ),
                     linestyle="dashed", color="k", alpha=0.5)
    
    for k in range(n_x_probe):
        res_ax.plot(x_probe_time[i][j][k],
                    x_probe_fit_residua[i][j][k],
                    ".", color=colors[k], markeredgewidth=0)
    res_ax.axhline(0, color="k", linewidth=0.5)
    
    main_ax.legend()
    
    res_ax.set_xlabel(trs.choose("Time T [Mp.u]", "Tiempo T [uMP]"))
    res_ax.set_ylabel(trs.choose("Electric Field\n" + r"$E_z(y=z=0)$",
                                 "Campo eléctrico\n" + r"$E_z(y=z=0)$"))
    main_ax.set_ylabel(trs.choose("Electric Field\n" + r"$E_z(y=z=0)$",
                                  "Campo eléctrico\n" + r"$E_z(y=z=0)$"))
        
#%%

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

#%%

plt.figure()

plt.title(plot_title_base + trs.choose(r": Residua $\sigma$", ": $\sigma$ residuos"))

for i in range(len(series)):
    plt.plot(test_param[i], 
              [x_probe_fit_res_std[i][j][0] for j in range(len(series[i]))],
              "o", color=colorConverter.to_rgba(series_colors[i], alpha=0.7), 
              fillstyle="none", markersize=8, markeredgewidth=1.5,
              label=f"{series_legend[i]} x = {x_probe_position_factor[i][j][0]} " + trs.choose("cell", "celda"))
    plt.plot(test_param[i], 
             [x_probe_fit_res_std[i][j][-1] for j in range(len(series[i]))],
             "o", color=series_colors[i], 
             alpha=0.4, markeredgewidth=0, markersize=8,
             label=f"{series_legend[i]} x = {x_probe_position_factor[i][j][-1]} " + trs.choose("cell", "celda"))

plt.legend()

plt.xlabel(test_param_label)
plt.ylabel(trs.choose("Electric Field Noise Amplitude\n", 
                      "Amplitud de ruido en campo eléctrico\n") + 
           trs.choose(r"${E_z}^{noise}(y=z=0)$", r"${E_z}^{ruido}(y=z=0)$") )
plt.tight_layout()

vs.saveplot(plot_file("NoiseVsXVsResolution.png"), overwrite=True)

#%%

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

#%% ANALYSE X AXIS FOR DIFFERENT TIMES VIA FIT AND RESIDUA

n_t_probe = 10

cropped_line = [[vma.crop_field_xprofile(results_line[i][j], x_line_index[i][j], 
                                         cell_width[i][j], pml_width[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

time_one_cell = [[(cell_width[i][j] - 2*pml_width[i][j]) * index[i][j] for j in range(len(series[i]))] for i in range(len(series))]
start_point = [[t_line_index[i][j](time_one_cell[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

probe_index = [[ [int(k) for k in np.linspace(start_point[i][j], len(t_line[i][j])-1, n_t_probe)] for j in range(len(series[i]))] for i in range(len(series))]

t_probe_field = [[ [cropped_line[i][j][:, k] for k in probe_index[i][j]] for j in range(len(series[i]))] for i in range(len(series))]

t_probe_time = [[ [t_line[i][j][k] for k in probe_index[i][j]] for j in range(len(series[i]))] for i in range(len(series))]

#%%

def fit_function_generator(wavelength, index):
    
    def fit_function(position, amplitude, phase, offset):
        
        medium_wavelength = wavelength / index
        medium_wave_number = 2 * np.pi / medium_wavelength
        
        return amplitude * np.cos( medium_wave_number * position + phase ) + offset
    
    return fit_function

t_probe_fit_functions = [[fit_function_generator(wlen[i][j], index[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

t_probe_fit_amplitude = []
t_probe_fit_phase = []
t_probe_fit_offset = []
t_probe_fit_residua = []
for i in range(len(series)):
    t_probe_fit_amplitude.append([])
    t_probe_fit_phase.append([])
    t_probe_fit_offset.append([])
    t_probe_fit_residua.append([])
    for j in range(len(series[i])):
        t_probe_fit_amplitude[-1].append([])
        t_probe_fit_phase[-1].append([])
        t_probe_fit_offset[-1].append([])
        t_probe_fit_residua[-1].append([])
        for k in range(n_t_probe):
            rsq, fit_params = va.nonlinear_fit(x_line_cropped[i][j],
                                           t_probe_field[i][j][k],
                                           t_probe_fit_functions[i][j],
                                           initial_guess=(1, 0, 0),
                                           showplot=False)
            amplitude, phase, offset = fit_params[0][0], fit_params[1][0], fit_params[2][0]
            t_probe_fit_amplitude[-1][-1].append( amplitude )
            t_probe_fit_phase[-1][-1].append( phase )
            t_probe_fit_offset[-1][-1].append( offset )
            
            residua = np.array(t_probe_field[i][j][k]) - t_probe_fit_functions[i][j]( x_line_cropped[i][j], amplitude, phase, offset)
            
            t_probe_fit_residua[-1][-1].append( residua )

t_probe_fit_res_std = [[ [np.std(t_probe_fit_residua[i][j][k]) for k in range(n_t_probe)] for j in range(len(series[i]))] for i in range(len(series))]

#%%

def see_t_probe(i,j):
    
    colors = plab.cm.jet(np.linspace(0,1,n_t_probe))
    
    fig = plt.figure()
    
    plot_grid = gridspec.GridSpec(ncols=1, nrows=5, hspace=0, figure=fig)
    main_ax = fig.add_subplot(plot_grid[:3,:])
    res_ax = fig.add_subplot(plot_grid[-2:,:])
    
    plt.suptitle(f"{series_legend[i]} {series_label[1](series[i][j])}")
    for k in range(n_t_probe):
        main_ax.plot(x_line_cropped[i][j] / (cell_width[i][j] - 2*pml_width[i][j]),
                     t_probe_field[i][j][k], 
                     color=colors[k],
                     label=f"t = {t_probe_time[i][j][k] / norm_period[i][j]:.2f} T")      
        main_ax.plot(x_line_cropped[i][j] / (cell_width[i][j] - 2*pml_width[i][j]),
                    t_probe_fit_functions[i][j]( x_line_cropped[i][j],
                                                t_probe_fit_amplitude[i][j][k],
                                                t_probe_fit_phase[i][j][k],
                                                t_probe_fit_offset[i][j][k] ),
                    linestyle="dashed", color="k", alpha=0.5)
    
    res_ax.set_xlabel(trs.choose("Position $X$ [cell]", "Posición $X$ [celda]"))
    
    for k in range(n_t_probe):
        res_ax.plot(x_line_cropped[i][j] / (cell_width[i][j] - 2*pml_width[i][j]),
                    t_probe_fit_residua[i][j][k],
                    ".", color=colors[k], markeredgewidth=0)
    res_ax.axhline(0, color="k", linewidth=0.5)
    
    main_ax.legend()
    
#%%

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


#%%

plt.figure()

plt.title(plot_title_base + trs.choose(r": Residua $\sigma$", ": $\sigma$ residuos"))

for i in range(len(series)):
    plt.plot(test_param[i], 
              [t_probe_fit_res_std[i][j][0] for j in range(len(series[i]))],
              "o", color=colorConverter.to_rgba(series_colors[i], alpha=0.7), 
              fillstyle="none", markersize=8, markeredgewidth=1.5,
              label=f"{series_legend[i]} t = {t_probe_time[i][j][0] / norm_period[i][j] :.2f} T")
    plt.plot(test_param[i], 
             [t_probe_fit_res_std[i][j][-1] for j in range(len(series[i]))],
             "o", color=series_colors[i], 
             alpha=0.4, markeredgewidth=0, markersize=8,
             label=f"{series_legend[i]} t = {t_probe_time[i][j][-1] / norm_period[i][j] :.2f} T")

plt.legend()

plt.xlabel(test_param_label)
plt.ylabel(trs.choose("Electric Field Noise Amplitude\n", 
                      "Amplitud de ruido en campo eléctrico\n") + 
           trs.choose(r"${E_z}^{noise}(y=z=0)$", r"${E_z}^{ruido}(y=z=0)$") )
plt.tight_layout()

vs.saveplot(plot_file("NoiseVsTVsResolution.png"), overwrite=True)