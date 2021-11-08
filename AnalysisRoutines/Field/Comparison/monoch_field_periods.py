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
from matplotlib.ticker import AutoMinorLocator
import os
import vmp_materials as vmt
import vmp_utilities as vmu
import vmp_analysis as vma
import v_plot as vp
import v_theory as vt
import v_save as vs
import v_utilities as vu

english = False
trs = vu.BilingualManager(english=english)

#%% PARAMETERS

# Saving directories
folder = ["Field/Sources/MonochPlanewave/TestPeriods/Vacuum",
          "Field/Sources/MonochPlanewave/TestPeriods/Vacuum",
          "Field/Sources/MonochPlanewave/TestPeriods/Water",
          "Field/Sources/MonochPlanewave/TestPeriods/Water"]
home = vs.get_home()

# Parameter for the test
test_param_string = "source_center"
test_param_in_series = False
test_param_in_params = True
test_param_position = 0
test_param_label = trs.choose("Wavelength", "Longitud de onda")

# Sorting and labelling data series
sorting_function = [lambda l : l]*4
series_label = [lambda s : trs.choose(r"Vacuum Centered", r"Vacío centrado"),
                lambda s : trs.choose(r"Vacuum Not Centered", r"Vacío no centrado"),
                lambda s : trs.choose(r"Water Centered", r"Agua centrado"),
                lambda s : trs.choose(r"Water Centered", r"Agua no centrado")]
series_must = ["True", "False"]*2 # leave "" per default
series_mustnt = [""]*4 # leave "" per default

# Scattering plot options
plot_title_base = trs.choose('Dimnesionless monochromatic wave', 
                             "Onda monocromática adimensional")
series_legend = trs.choose(["Vacuum Centered", "Vacuum Not Centered", 
                            "Water Centered", "Water Not Centered"],
                           ["Vacío centrado", "Vacío no centrado", 
                            "Agua centrado", "Agua no centrado"])
series_colors = [plab.cm.Reds, plab.cm.Reds, 
                 plab.cm.Blues, plab.cm.Blues]
series_linestyles = ["solid", "dashed"]*2
plot_make_big = False
plot_file = lambda n : os.path.join(home, "DataAnalysis/Field/Sources/MonochPlanewave/TestPeriods/TestPeriods" + n)

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
    try:
        norm_amplitude.append( [pi["norm_amplitude"] for pi in p] )    
        norm_period.append( [pi["norm_period"] for pi in p] )
        requires_normalization = False
    except:
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
    
    period_results = [[vma.get_period_from_source(source_results[i][j], t_line[i][j])[-1] for j in range(len(series[i]))] for i in range(len(series))]
    amplitude_results = [[vma.get_amplitude_from_source(source_results[i][j])[-1] for j in range(len(series[i]))] for i in range(len(series))]
    
    results_plane = [[np.asarray(results_plane[i][j]) / amplitude_results[i][j] for j in range(len(series[i]))] for i in range(len(series))]
    results_line = [[np.asarray(results_line[i][j]) / amplitude_results[i][j] for j in range(len(series[i]))] for i in range(len(series))]

#%% SHOW SOURCE AND FOURIER USED FOR NORMALIZATION

colors = [["r"], ["maroon"], ["b"], ["navy"]]
# colors = [sc(np.linspace(0,1,len(s)+2))[2:] 
#           for sc, s in zip(series_colors, series)]

plt.figure()
plt.suptitle(plot_title_base)

lines = []
for i in range(len(series)):
    for j in range(len(series[i])):
        l, = plt.plot(t_line[i][j] / period_results[i][j], 
                      source_results[i][j],
                      label=series_label[i](series[i][j]),
                      color=colors[i][j])            
        lines.append(l)
plt.xlabel(trs.choose("Time in multiples of period", "Tiempo en múltiplos del período"))
plt.ylabel(trs.choose(r"Normalized Electric Field $E_z(y=z=0)$",
                      r"Campo eléctrico normalizado $E_z(y=z=0)$"))
plt.legend(ncol=2)

plt.savefig(plot_file("Source.png"))
        
fourier = [[np.abs(np.fft.rfft(source_results[i][j] / amplitude_results[i][j])) for j in range(len(series[i]))] for i in range(len(series))]
fourier_freq = [[np.fft.rfftfreq(len(source_results[i][j]), d=period_line[i][j])  for j in range(len(series[i]))] for i in range(len(series))]
if use_units:
    fourier_wlen = [[from_um_factor[i][j] * 1e3 / fourier_freq[i][j]  for j in range(len(series[i]))] for i in range(len(series))]
else:
    fourier_wlen = [[1 / fourier_freq[i][j]  for j in range(len(series[i]))] for i in range(len(series))]

plt.figure()
plt.suptitle(plot_title_base)

lines = []
for i in range(len(series)):
    for j in range(len(series[i])):
        plt.plot(fourier_wlen[i][j], fourier[i][j],
                 label=series_label[i](series[i][j]),
                 color=colors[i][j])
if use_units:
    plt.xlabel(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
else:
    plt.xlabel(trs.choose("Wavelength [MPu]", "Longitud de onda [uMP]"))
plt.ylabel(trs.choose(r"Electric Field Fourier $\mathcal{F}\;(E_z)$",
                      r"Transformada del campo eléctrico $\mathcal{F}\;(E_z)$"))
plt.legend(ncol=2)

plt.savefig(plot_file("SourceFFT.png"))

if use_units: plt.xlim([350, 850])
else: plt.xlim([0, 2])
        
plt.savefig(plot_file("SourceFFTZoom.png"))

#%%

peaks_index = [[vma.get_peaks_from_source(source_results[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
peaks_heights = [[source_results[i][j][peaks_index[i][j]] for j in range(len(series[i]))] for i in range(len(series))]
peaks_times = [[t_line[i][j][peaks_index[i][j]] for j in range(len(series[i]))] for i in range(len(series))]

peaks_percentual_variation = [[100 * ( max(peaks_heights[i][j]) - min(peaks_heights[i][j]) ) / min(peaks_heights[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

plt.figure()
plt.suptitle(plot_title_base)

lines = []
for i in range(len(series)):
    for j in range(len(series[i])):
        l, = plt.plot(t_line[i][j] / period_results[i][j], 
                      source_results[i][j],
                      label=series_label[i](series[i][j]),
                      color=colors[i][j])            
        l, = plt.plot(t_line[i][j][peaks_index[i][j]] / period_results[i][j], 
                      source_results[i][j][peaks_index[i][j]], "o",
                      label=series_label[i](series[i][j]),
                      color=colors[i][j])       
        lines.append(l)
plt.xlabel(trs.choose("Time in multiples of period", "Tiempo en múltiplos del período"))
plt.ylabel(trs.choose(r"Normalized Electric Field $E_z(y=z=0)$",
                      r"Campo eléctrico normalizado $E_z(y=z=0)$"))
plt.legend(ncol=2)

