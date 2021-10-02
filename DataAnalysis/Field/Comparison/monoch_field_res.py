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

#%% PARAMETERS

# Saving directories
folder = ["Field/Sources/MonochPlanewave/TestRes/Not Centered/Vacuum",
          "Field/Sources/MonochPlanewave/TestRes/Not Centered/Water"]
home = vs.get_home()

# Parameter for the test
test_param_string = "resolution_wlen"
test_param_in_series = True
test_param_in_params = True
test_param_position = 0
test_param_label = trs.choose(r"Resolution [points/$\lambda$]", 
                              r"Resolución [points/$\lambda$]")

# Sorting and labelling data series
sorting_function = [lambda l : vu.sort_by_number(l, test_param_position)]*2
series_label = [lambda s : " ",
                lambda s : trs.choose("Resolution ", "Resolución ") + rf"{vu.find_numbers(s)[test_param_position]:.0f}"]
series_must = [""]*2 # leave "" per default
series_mustnt = ["Weird"]*2 # leave "" per default

# Scattering plot options
plot_title_base = trs.choose('Dimnesionless monochromatic wave', 
                             "Onda monocromática adimensional")
series_legend = trs.choose(["Vacuum", "Water"], ["Vacío", "Agua"])
series_colors = [plab.cm.Reds, plab.cm.Blues]
series_linestyles = ["solid", "solid"]
plot_make_big = False
plot_file = lambda n : os.path.join(home, "DataAnalysis/Field/Sources/MonochPlanewave/TestRes/TestRes" + n)

#%% LOAD DATA

def file_definer(path):
    return lambda s, n : os.path.join(path, s, n)

path = [os.path.join(home, fold) for fold in folder]
file = [file_definer(pa) for pa in path]

series = []
files_line = []
files_plane = []
results_line = []
results_plane = []
t_line = []
x_line = []
t_plane = []
y_plane = []
z_plane = []
params = []

# for f, sf, sm, smn in zip(folder, sorting_function, series_must, series_mustnt):
for i in range(len(folder)):

    # path.append( os.path.join(home, f) )
    # file.append( lambda f, s : os.path.join(path[-1], f, s) )
    
    series.append( os.listdir(path[i]) )
    series[-1] = vu.filter_by_string_must(series[-1], series_must[i])
    if series_mustnt[i]!="": 
        series[-1] = vu.filter_by_string_must(series[-1], series_mustnt[i], False)
    series[-1] = sorting_function[i](series[-1])
    
    files_line.append( [] )
    files_plane.append( [] )
    for s in series[-1]:
        files_line[-1].append( h5.File(file[i](s, "Field-Lines.h5"), "r") )
        files_plane[-1].append( h5.File(file[i](s, "Field-Planes.h5"), "r") )
    del s
    
    results_line.append( [fi["Ez"] for fi in files_line[-1]] )
    results_plane.append( [fi["Ez"] for fi in files_plane[-1]] )
    params.append( [dict(fi["Ez"].attrs) for fi in files_line[-1]] )
    
    t_line.append( [np.asarray(fi["T"]) for fi in files_line[-1]] )
    x_line.append( [np.asarray(fi["X"]) for fi in files_line[-1]] )
    
    t_plane.append( [np.asarray(fi["T"]) for fi in files_plane[-1]] )
    y_plane.append( [np.asarray(fi["Y"]) for fi in files_plane[-1]] )
    z_plane.append( [np.asarray(fi["Z"]) for fi in files_plane[-1]] )
    
    for s, p in zip(series[-1], params[-1]):
        try:
            f = h5.File(file[-1](s, "Resources.h5"))
            p["used_ram"] = np.array(f["RAM"])
            p["used_swap"] = np.array(f["SWAP"])
            p["elapsed_time"] = np.array(f["ElapsedTime"])
        except FileNotFoundError:
            f = h5.File(file[-1](s, "RAM.h5"))
            p["used_ram"] = np.array(f["RAM"])
            p["used_swap"] = np.array(f["SWAP"])
            p["elapsed_time"] = p["elapsed"]
            del p["elapsed"]
    # del s, p
del i
            
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
    
use_units = np.array(units).any()

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

source_results = [[vma.get_source_from_line(results_line[i][j], x_line_index[i][j], source_center[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

if not requires_normalization:
    
    period_results, amplitude_results = norm_period, norm_amplitude
    
else:
    
    period_results = [[vma.get_period_from_source(source_results[i][j], t_line[i][j],
                                                  periods_sensitivity=0.06) for j in range(len(series[i]))] for i in range(len(series))]
    amplitude_results = [[vma.get_amplitude_from_source(source_results[i][j],
                                                        amplitude_sensitivity=0.05) for j in range(len(series[i]))] for i in range(len(series))]
    
    source_results = [[source_results[i][j] / amplitude_results[i][j] for j in range(len(series[i]))] for i in range(len(series))]
    results_plane = [[np.asarray(results_plane[i][j]) / amplitude_results[i][j] for j in range(len(series[i]))] for i in range(len(series))]
    results_line = [[np.asarray(results_line[i][j]) / amplitude_results[i][j] for j in range(len(series[i]))] for i in range(len(series))]
    
    norm_period, norm_amplitude = period_results, amplitude_results

#%% SHOW SOURCE AND FOURIER USED FOR NORMALIZATION

# colors = [["r"], ["maroon"], ["b"], ["navy"]]
colors = [sc(np.linspace(0,1,len(s)+2))[2:] 
          for sc, s in zip(series_colors, series)]

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
plt.ylabel(trs.choose(r"Normalized Electric Field $E_z(y=z=0)$",
                      r"Campo eléctrico normalizado $E_z(y=z=0)$"))

box = fig.axes[0].get_position()
box.x1 = box.x1 - .3 * (box.x1 - box.x0)
fig.axes[0].set_position(box)
leg = plt.legend(ncol=2, columnspacing=-0.5, 
                 bbox_to_anchor=(1.6, .5), loc="center right", frameon=False)

plt.savefig(plot_file("Source.png"))
        
fourier = [[np.abs(np.fft.rfft(source_results[i][j])) for j in range(len(series[i]))] for i in range(len(series))]
fourier_freq = [[np.fft.rfftfreq(len(source_results[i][j]), d=period_line[i][j])  for j in range(len(series[i]))] for i in range(len(series))]
if use_units:
    fourier_wlen = [[from_um_factor[i][j] * 1e3 / fourier_freq[i][j]  for j in range(len(series[i]))] for i in range(len(series))]
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
second_legend = plt.legend(ncol=2, columnspacing=-0.5, bbox_to_anchor=(1.6, .5), 
                           loc="center right", frameon=False)
plt.gca().add_artist(first_legend)

plt.savefig(plot_file("SourceFFT.png"))

if use_units: plt.xlim([350, 850])
else: plt.xlim([0, 2])
        
plt.savefig(plot_file("SourceFFTZoom.png"))

#%%

fourier_max_wlen = [[fourier_wlen[i][j][ np.argmax(fourier[i][j]) ]  for j in range(len(series[i]))] for i in range(len(series))]
fourier_max_best = [[fourier_wlen[i][j][ np.argmin(np.abs(fourier_wlen[i][j] - fourier_best[i][j])) ]  for j in range(len(series[i]))] for i in range(len(series))]

colors = [sc(np.linspace(0,1,len(s)+2))[2:] 
          for sc, s in zip(series_colors, series)]

if len(series)>1:
    colors = [*["red", "blue"]*2]
else:
    colors = ["k"]

markers = ["o", "o"]
markers_alpha = [0.4, 0.4]

plt.figure()
plt.title(plot_title_base)
for i in range(len(series)):
    plt.plot(test_param[i], 
             100 * ( np.array(fourier_max_wlen[i]) - np.array(fourier_max_best[i]) ) / np.array(fourier_max_best[i]), 
             color=colors[i], marker=markers[i], alpha=markers_alpha[i],
             markersize=8, linestyle="", markeredgewidth=0)
plt.grid(True)
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

# colors = [sc(np.linspace(0,1,len(s)+2))[2:] 
#           for sc, s in zip(series_colors, series)]

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
# plt.ylabel(trs.choose(r"Normalized Electric Field $E_z(y=z=0)$",
#                       r"Campo eléctrico normalizado $E_z(y=z=0)$"))
# plt.legend(ncol=2)

#%%

peaks_height_variation = [[100 * ( max(peaks_heights[i][j]) - min(peaks_heights[i][j]) ) / min(peaks_heights[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

if len(series)>1:
    colors = [*["red", "blue"]*2]
else:
    colors = ["k"]

markers = ["o", "o"]
markers_alpha = [0.4, 0.4]

plt.figure()
plt.title(plot_title_base)
for i in range(len(series)):
    plt.plot(test_param[i], peaks_height_variation[i], 
             color=colors[i], marker=markers[i], alpha=markers_alpha[i],
             markersize=8, linestyle="", markeredgewidth=0)
plt.grid(True)
plt.legend(series_legend)
plt.xlabel(test_param_label)
plt.ylabel(trs.choose("Maximum percentual variation in amplitude ", 
                      "Diferencia porcentual máxima en amplitud ") + 
           r"$\max[ E_z(y=z=0) ]$ [%]")
           # r"$\frac{\max|E^{max}|_i - \min|E^{max}|_i}{\min|E^{max}|_i}$")
vs.saveplot(plot_file("AmpVariation.png"), overwrite=True)

#%%

peaks_period_variation = [[100 * ( max(peaks_periods[i][j]) - min(peaks_periods[i][j]) ) / min(peaks_periods[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

if len(series)>1:
    colors = [*["red", "blue"]*2]
else:
    colors = ["k"]

markers = ["o", "o"]
markers_alpha = [0.4, 0.4]

plt.figure()
plt.title(plot_title_base)
for i in range(len(series)):
    plt.plot(test_param[i], peaks_period_variation[i], 
             color=colors[i], marker=markers[i], markersize=8, 
             alpha=markers_alpha[i], linestyle="", markeredgewidth=0)
plt.grid(True)
plt.legend(series_legend)
plt.xlabel(test_param_label)
plt.ylabel(trs.choose("Maximum percentual variation in period ", 
                      "Diferencia porcentual máxima en período ") + 
           r"$T$ [%]")
           # r"$\frac{\max T_i - \min T_i}{\min T_i}$")
vs.saveplot(plot_file("PerVariation.png"), overwrite=True)

#%% ANALYSE X AXIS FOR DIFFERENT POSITIONS VIA FOURIER

n_probe = 3

x_probe_position = [[ [-cell_width[i][j]/2 + pml_width[i][j] + k * (cell_width[i][j] - 2*pml_width[i][j]) / (n_probe-1) 
                       for k in range(n_probe)] for j in range(len(series[i]))] for i in range(len(series))]
x_probe_position_factor = [[ [x_probe_position[i][j][k] / (cell_width[i][j] - 2*pml_width[i][j])
                              for k in range(n_probe)] for j in range(len(series[i]))] for i in range(len(series))]

x_probe_field = [[ [results_line[i][j][
                        x_line_index[i][j](-cell_width[i][j]/2 + pml_width[i][j] + k * (cell_width[i][j] - 2*pml_width[i][j]) / (n_probe-1)), :
                    ] for k in range(n_probe)] for j in range(len(series[i]))] for i in range(len(series))]

cropping_index = [[ [np.where(np.abs(x_probe_field[i][j][k]) > 0.05)[0][0] for k in range(n_probe)] for j in range(len(series[i]))] for i in range(len(series))]

x_probe_field = [[ [x_probe_field[i][j][k][ cropping_index[i][j][k] : ]  for k in range(n_probe)] for j in range(len(series[i]))] for i in range(len(series))]
x_probe_time = [[ [t_line[i][j][ cropping_index[i][j][k] : ]  for k in range(n_probe)] for j in range(len(series[i]))] for i in range(len(series))]

#%%

x_probe_fourier = [[ [np.abs(np.fft.rfft(x_probe_field[i][j][k], norm="ortho")) for k in range(n_probe)] for j in range(len(series[i]))] for i in range(len(series))]
x_probe_freqs = [[ [np.fft.rfftfreq(len(x_probe_field[i][j][k]), d=period_line[i][j]) for k in range(n_probe)] for j in range(len(series[i]))] for i in range(len(series))]
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
        plt.plot( x_probe_time[i][j][k],
                  x_probe_field[i][j][k], 
                  label=f"x = {x_probe_position[i][j][k]:.2f} " + trs.choose("cell", "celda"), 
                  color=colors[k])
    plt.legend()

    plt.xlabel(trs.choose("Time T [Mp.u]", "Tiempo T [uMP]"))
    plt.ylabel(trs.choose("Normalized Electric Field\n" + r"$E_z(y=z=0)$",
                          "Campo eléctrico normalizado\n" + r"$E_z(y=z=0)$"))
        
    plt.figure()
    
    plt.title(f"{series_legend[i]} {series_label[1](series[i][j])}")
    for k in range(n_probe):
        plt.plot( x_probe_wlen[i][j][k], 
                  x_probe_fourier[i][j][k], 
                  label=f"x = {x_probe_position[i][j][k]:.2f} " + trs.choose("cell", "celda"), 
                  color=colors[k])
    plt.xlim(0,5)
    plt.legend()
    
    plt.xlabel(trs.choose("Frequency f [Mp.u]", "Frecuencia f [uMP]"))
    plt.ylabel(trs.choose("Normalized Electric Field\n" + r"$E_z(y=z=0)$",
                          "Campo eléctrico normalizado\n" + r"$E_z(y=z=0)$"))
    
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
            rsq, params = va.nonlinear_fit(x_probe_time[i][j][k],
                                           x_probe_field[i][j][k],
                                           x_probe_fit_functions[i][j],
                                           initial_guess=(1, 0, 0),
                                           showplot=False)
            amplitude, phase, offset = params[0][0], params[1][0], params[2][0]
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
    res_ax.set_ylabel(trs.choose("Normalized Electric Field\n" + r"$E_z(y=z=0)$",
                                 "Campo eléctrico normalizado\n" + r"$E_z(y=z=0)$"))
    main_ax.set_ylabel(trs.choose("Normalized Electric Field\n" + r"$E_z(y=z=0)$",
                                  "Campo eléctrico normalizado\n" + r"$E_z(y=z=0)$"))
        
#%%

fig, axes = plt.subplots(nrows=len(series), sharex=True, sharey=True, 
                         gridspec_kw={"hspace":0})

colors = [sc(np.linspace(0,1,len(s)+2))[2:] 
          for sc, s in zip(series_colors, series)]

plt.suptitle(plot_title_base + trs.choose(r": Residua $\sigma$", ": $\sigma$ residuos"))

lines = [[]]*len(series)
for i in range(len(series)):
    for j in range(len(series[i])):
        l, = axes[i].plot(x_probe_position_factor[i][j], 
                          x_probe_fit_res_std[i][j], 
                          "o-", alpha=0.7, color=colors[i][j], markeredgewidth=0)
                          # label=series_label[1](series[i][j]))
        lines[i].append(l)
        axes[i].set_ylabel(trs.choose("Normalized Electric\n Field " + r"$E_z(y=z=0)$",
                                      "Campo eléctrico \n normalizado " + r"$E_z(y=z=0)$"))

fig.set_size_inches([10.28,  4.8 ])

for ax in axes:
    box = ax.get_position()
    box.x1 = box.x1 - .2 * (box.x1 - box.x0)
    ax.set_position(box)
axes[-1].set_xlabel(trs.choose("Position X [Cell]", "Posición X [Celda]"))


plt.legend([*lines[0], *lines[1]], 
           [*[" "]*len(series[0]), *[series_label[1](series[1][j]) for j in range(len(series[1]))]],
    columnspacing=-0.5, ncol=2, bbox_to_anchor=(1.35, 1), loc="center right", frameon=False)

vs.saveplot(plot_file("NoiseVsX.png"), overwrite=True)

#%%

if len(series)>1:
    colors = [*["red", "blue"]*2]
else:
    colors = ["k"]

markers = ["o", "o"]
markers_alpha = [0.4, 0.4]

plt.figure()

plt.title(plot_title_base + trs.choose(r": Residua $\sigma$", ": $\sigma$ residuos"))

for i in range(len(series)):
    plt.plot(test_param[i], 
              [x_probe_fit_res_std[i][j][0] for j in range(len(series[i]))],
              "o", color=colorConverter.to_rgba(colors[i], alpha=0.7), 
              fillstyle="none", markersize=8, markeredgewidth=1.5,
              label=f"{series_legend[i]} x = {x_probe_position_factor[i][j][0]} " + trs.choose("cell", "celda"))
    plt.plot(test_param[i], 
             [x_probe_fit_res_std[i][j][-1] for j in range(len(series[i]))],
             "o", color=colors[i], 
             alpha=markers_alpha[i], markeredgewidth=0, markersize=8,
             label=f"{series_legend[i]} x = {x_probe_position_factor[i][j][-1]} " + trs.choose("cell", "celda"))

plt.legend()

plt.xlabel(test_param_label)
plt.ylabel(trs.choose("Normalized Electric Field Noise Amplitude\n", 
                      "Amplitud de ruido en campo eléctrico normalizado\n") + 
           trs.choose(r"${E_z}^{noise}(y=z=0)$", r"${E_z}^{ruido}(y=z=0)$") )
plt.tight_layout()

vs.saveplot(plot_file("NoiseVsXVsResolution.png"), overwrite=True)

#%%

if len(series)>1:
    colors = [*["red", "blue"]*2]
else:
    colors = ["k"]

markers = ["o", "o"]
markers_alpha = [0.4, 0.4]

plt.figure()

plt.title(plot_title_base + trs.choose(r": Residua $\sigma$", ": $\sigma$ residuos"))

for i in range(len(series)):
    plt.plot(test_param[i], 
             [np.mean([x_probe_fit_res_std[i][j][-1] - x_probe_fit_res_std[i][j][0],
                       x_probe_fit_res_std[i][j][-1] - x_probe_fit_res_std[i][j][1],
                       x_probe_fit_res_std[i][j][-2] - x_probe_fit_res_std[i][j][1],
                       x_probe_fit_res_std[i][j][-2] - x_probe_fit_res_std[i][j][0]])
              for j in range(len(series[i]))],
             "o", color=colors[i], 
             alpha=markers_alpha[i], markeredgewidth=0, markersize=8,
             label=series_legend[i])

plt.legend()

plt.xlabel(test_param_label)
plt.ylabel(trs.choose("Normalized Electric Field Noise Difference\n", 
                      "Diferencia de ruido en campo eléctrico normalizado\n") + 
           trs.choose(r"${E_z}^{noise}(x=\Delta X/2) - {E_z}^{noise}(x=-\Delta X/2)$",
                      r"${E_z}^{ruido}(x=\Delta X/2) - {E_z}^{ruido}(x=-\Delta X/2)$") )
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
            rsq, params = va.nonlinear_fit(x_line_cropped[i][j],
                                           t_probe_field[i][j][k],
                                           t_probe_fit_functions[i][j],
                                           initial_guess=(1, 0, 0),
                                           showplot=False)
            amplitude, phase, offset = params[0][0], params[1][0], params[2][0]
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
    
    res_ax.set_xlabel(trs.choose("Position X [cell]", "Posición X [celda]"))
    
    for k in range(n_t_probe):
        res_ax.plot(x_line_cropped[i][j] / (cell_width[i][j] - 2*pml_width[i][j]),
                    t_probe_fit_residua[i][j][k],
                    ".", color=colors[k], markeredgewidth=0)
    res_ax.axhline(0, color="k", linewidth=0.5)
    
    main_ax.legend()
    
#%%

fig, axes = plt.subplots(nrows=len(series), sharex=True, sharey=True, 
                         gridspec_kw={"hspace":0})

colors = [sc(np.linspace(0,1,len(s)+2))[2:] 
          for sc, s in zip(series_colors, series)]

plt.suptitle(plot_title_base + trs.choose(r": Residua $\sigma$", ": $\sigma$ residuos"))

lines = [[]]*len(series)
for i in range(len(series)):
    for j in range(len(series[i])):
        l, = axes[i].plot(t_probe_time[i][j] / norm_period[i][j], 
                          t_probe_fit_res_std[i][j], 
                          "o-", alpha=0.7, color=colors[i][j], markeredgewidth=0)
        axes[i].set_ylabel(trs.choose("Normalized Electric\n Field " + r"$E_z(y=z=0)$",
                                      "Campo eléctrico \n normalizado " + r"$E_z(y=z=0)$"))
                          # label=series_label[1](series[i][j]))
        lines[i].append(l)

fig.set_size_inches([10.28,  4.8 ])

for ax in axes:
    box = ax.get_position()
    box.x1 = box.x1 - .2 * (box.x1 - box.x0)
    ax.set_position(box)
axes[-1].set_xlabel(trs.choose("Time T [MPu]", "Tiempo T [uMP]"))

plt.legend([*lines[0], *lines[1]], 
           [*[" "]*len(series[0]), *[series_label[1](series[1][j]) for j in range(len(series[1]))]],
    columnspacing=-0.5, ncol=2, bbox_to_anchor=(1.35, 1), loc="center right", frameon=False)

vs.saveplot(plot_file("NoiseVsT.png"), overwrite=True)


#%%

if len(series)>1:
    colors = [*["red", "blue"]*2]
else:
    colors = ["k"]

markers = ["o", "o"]
markers_alpha = [0.4, 0.4]

plt.figure()

plt.title(plot_title_base + trs.choose(r": Residua $\sigma$", ": $\sigma$ residuos"))

for i in range(len(series)):
    plt.plot(test_param[i], 
              [t_probe_fit_res_std[i][j][0] for j in range(len(series[i]))],
              "o", color=colorConverter.to_rgba(colors[i], alpha=0.7), 
              fillstyle="none", markersize=8, markeredgewidth=1.5,
              label=f"{series_legend[i]} t = {t_probe_time[i][j][0] / norm_period[i][j] :.2f} T")
    plt.plot(test_param[i], 
             [t_probe_fit_res_std[i][j][-1] for j in range(len(series[i]))],
             "o", color=colors[i], 
             alpha=markers_alpha[i], markeredgewidth=0, markersize=8,
             label=f"{series_legend[i]} t = {t_probe_time[i][j][-1] / norm_period[i][j] :.2f} T")

plt.legend()

plt.xlabel(test_param_label)
plt.ylabel(trs.choose("Normalized Electric Field Noise Amplitude\n", 
                      "Amplitud de ruido en campo eléctrico normalizado\n") + 
           trs.choose(r"${E_z}^{noise}(y=z=0)$", r"${E_z}^{ruido}(y=z=0)$") )
plt.tight_layout()

vs.saveplot(plot_file("NoiseVsTVsResolution.png"), overwrite=True)