#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:45:00 2020

@author: vall
"""

# Field of 120nm-diameter Au sphere given a visible monochromatic incident wave.

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
# folder = ["Field/NPMonoch/AuSphere/VacWatField/Vacuum", 
#           "Field/NPMonoch/AuSphere/VacWatField/Water"]
folder = ["Field/NPMonoch/AuSphere/VacWatTest/TestEmpty/Vacuum", 
          "Field/NPMonoch/AuSphere/VacWatTest/TestEmpty/Water",
          "Field/NPMonoch/AuSphere/VacWatTest/TestEmpty/Vacuum", 
          "Field/NPMonoch/AuSphere/VacWatTest/TestEmpty/Water"]
home = vs.get_home()

# Parameter for the test
test_param_string = "wlen"
test_param_in_params = True
test_param_in_series = True
test_param_position = 0
test_param_label = trs.choose("Wavelength", "Longitud de onda")

# Sorting and labelling data series
sorting_function = [lambda l : vu.sort_by_number(l, test_param_position)]*4
series_label = [lambda s : rf" $\lambda$ = {vu.find_numbers(s)[test_param_position]:.0f} nm"]*4
series_must = [*["Res3ERF2.0"]*2, *["Res5ERF2.0"]*2] # leave "" per default
series_mustnt = ["Old"]*4 # leave "" per default

# Scattering plot options
plot_title_ending = trs.choose("Au 60 nm sphere", "esfera de Au de 60 nm")
series_legend = trs.choose(["Vacuum Res 3", "Water Res 3",
                            "Vacuum Res 5", "Water Res 5"], 
                           ["Vacío Res 3", "Agua Res 3",
                           "Vacío Res 5", "Agua Res 5"])
series_colormaps = [plab.cm.Reds, plab.cm.Blues]*2
series_colors = ["red", "blue"]*2
series_ind_colors = [["C0", "C2", "C3"]]*4
series_markers = [*["o"]*2, *["D"]*2]
series_markersizes = [*[8]*2, *[7]*2]
series_linestyles = [*["solid"]*2, *["dashed"]*2]
plot_make_big = False
plot_file = lambda n : os.path.join(home, "DataAnalysis/Field/NPMonoch/AuSphere/VacWatField/WLen/Empty/Res" + n)

#%% LOAD BASIC DATA

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
    
# Get the corresponding data files
files_line = [[h5.File(file[i](series[i][j], "Field-Lines.h5"), "r") for j in range(len(series[i]))] for i in range(len(series))]
files_plane = [[h5.File(file[i](series[i][j], "Field-Planes.h5"), "r") for j in range(len(series[i]))] for i in range(len(series))]

# Get the corresponding field data
results_line = [[files_line[i][j]["Ez"] for j in range(len(series[i]))] for i in range(len(series))]
results_plane = [[files_plane[i][j]["Ez"] for j in range(len(series[i]))] for i in range(len(series))]

# Get the time and spatial dimensions data
t_line = [[np.asarray(files_line[i][j]["T"]) for j in range(len(series[i]))] for i in range(len(series))]
x_line = [[np.asarray(files_line[i][j]["X"]) for j in range(len(series[i]))] for i in range(len(series))]

t_plane = [[np.asarray(files_plane[i][j]["T"]) for j in range(len(series[i]))] for i in range(len(series))]
y_plane = [[np.asarray(files_plane[i][j]["Y"]) for j in range(len(series[i]))] for i in range(len(series))]
z_plane = [[np.asarray(files_plane[i][j]["Z"]) for j in range(len(series[i]))] for i in range(len(series))]

# Get the parameters of the simulations
params = [[dict(files_line[i][j]["Ez"].attrs) for j in range(len(series[i]))] for i in range(len(series))]

# Get the test parameter
if test_param_in_series:
    test_param = [[vu.find_numbers(series[i][j])[test_param_position] for j in range(len(series[i]))] for i in range(len(series))]
elif test_param_in_params:
    test_param = [[params[i][j][test_param_string] for j in range(len(series[i]))] for i in range(len(series))]
else:
    raise ValueError("Test parameter is nowhere to be found")

#%% LOAD NORMALIZATION DATA

# Extract some normalization parameters, if possible, and check if it's needed
requires_normalization = False
norm_amplitude = [[]] * len(series)
norm_period = [[]] * len(series)
norm_path = [[]] * len(series)
for i in range(len(series)):
    try:
        norm_amplitude[i] = [params[i][j]["norm_amplitude"] for j in range(len(series[i]))]
        norm_period[i] = [params[i][j]["norm_period"] for j in range(len(series[i]))]
        norm_path[i] = [params[i][j]["norm_path"] for j in range(len(series[i]))]
    except:
        requires_normalization = True
del i

# Now try to get the normalization data files
files_line_norm = []
for i in range(len(series)):
    files_line_norm.append([])
    for j in range(len(series[i])):
        try:    
            try:
                files_line_norm[i].append( h5.File(file[i](series[i][j], "Field-Lines-Norm.h5"), "r") )
            except:
                files_line_norm[i].append( h5.File(os.path.join(norm_path[i][j], "Field-Lines-Norm.h5"), "r") )
            print(f"Loading available normfield for {i},{j}")
        except:
            norm_path_ij = vm.check_normfield(params[i][j])
            try:
                files_line_norm[i].append( h5.File(os.path.join(norm_path_ij[0], "Field-Lines-Norm.h5"), "r") )
                print(f"Loading compatible normfield for {i},{j}")
                norm_path[i][j] = norm_path_ij
            except:
                files_line_norm[i].append( files_line[i][j] )
                print(f"Using NP data for {i},{j} normalization.",
                      "This is by all means not ideal!!",
                      "If possible, run again these simulations.")
del i, j

# Get the corresponding field data
results_line_norm = [[ files_line_norm[i][j]["Ez"] for j in range(len(series[i]))] for i in range(len(series))]

# Get the time and spatial dimensions data
t_line_norm = [[ np.asarray(files_line_norm[i][j]["T"]) for j in range(len(series[i]))] for i in range(len(series))]
x_line_norm = [[ np.asarray(files_line_norm[i][j]["X"]) for j in range(len(series[i]))] for i in range(len(series))]

#%% LOAD PARAMETERS AND CONTROL VARIABLES

# Get the RAM and elapsed time data
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

# Extract elapsed time data
elapsed_time = [[params[i][j]["elapsed_time"] for j in range(len(series[i]))] for i in range(len(series))]
total_elapsed_time = [[sum(params[i][j]["elapsed_time"]) for j in range(len(series[i]))] for i in range(len(series))]

# Extract RAM Data
used_ram = [[np.array(params[i][j]["used_ram"])/(1024)**2 for j in range(len(series[i]))] for i in range(len(series))]
total_used_ram = [[np.sum(used_ram[i][j], axis=1) for j in range(len(series[i]))] for i in range(len(series))]
used_swap = [[params[i][j]["used_swap"] for j in range(len(series[i]))] for i in range(len(series))]

# Extract some other parameters from the parameters dicts
from_um_factor = [[params[i][j]["from_um_factor"] for j in range(len(series[i]))] for i in range(len(series))]
resolution = [[params[i][j]["resolution"] for j in range(len(series[i]))] for i in range(len(series))]
courant = [[params[i][j]["courant"] for j in range(len(series[i]))] for i in range(len(series))]
r = [[params[i][j]["r"] for j in range(len(series[i]))] for i in range(len(series))]
material = [[params[i][j]["material"] for j in range(len(series[i]))] for i in range(len(series))]
paper = [[params[i][j]["paper"] for j in range(len(series[i]))] for i in range(len(series))]
index = [[params[i][j]["submerged_index"] for j in range(len(series[i]))] for i in range(len(series))]
wlen = [[params[i][j]["wlen"] for j in range(len(series[i]))] for i in range(len(series))]
cell_width = [[params[i][j]["cell_width"] for j in range(len(series[i]))] for i in range(len(series))]
pml_width = [[params[i][j]["pml_width"] for j in range(len(series[i]))] for i in range(len(series))]
source_center = [[params[i][j]["source_center"] for j in range(len(series[i]))] for i in range(len(series))]
period_plane = [[params[i][j]["period_plane"] for j in range(len(series[i]))] for i in range(len(series))]
period_line = [[params[i][j]["period_line"] for j in range(len(series[i]))] for i in range(len(series))]
until_time = [[params[i][j]["until_time"] for j in range(len(series[i]))] for i in range(len(series))]
time_period_factor = [[params[i][j]["time_period_factor"] for j in range(len(series[i]))] for i in range(len(series))]

# Determine some other parameters, calculating them from others
empty_width = [[cell_width[i][j]/2 - pml_width[i][j] - r[i][j] for j in range(len(series[i]))] for i in range(len(series))]
empty_r_factor = [[empty_width[i][j] / r[i][j] for j in range(len(series[i]))] for i in range(len(series))]
resolution_wlen = [[wlen[i][j] * resolution[i][j] for j in range(len(series[i]))] for i in range(len(series))]

# Guess some more parameters, calculating them from others
minor_division = [[fum * 1e3 / res for fum, res in zip(frum, reso)] for frum, reso in zip(from_um_factor, resolution)]
width_points = [[int(cell_width[i][j] * resolution[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
inner_width_points = [[int( (cell_width[i][j] - 2*pml_width[i][j]) * resolution[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
grid_points = [[wp**3 for wp in wpoints] for wpoints in width_points]
memory_B = [[2 * 12 * gp * 32 for p, gp in zip(par, gpoints)] for par, gpoints in zip(params, grid_points)] # in bytes

#%% POSITION RECONSTRUCTION

t_line_index = [[vma.def_index_function(t_line[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
x_line_index = [[vma.def_index_function(x_line[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

t_plane_index = [[vma.def_index_function(t_plane[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
y_plane_index = [[vma.def_index_function(y_plane[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
z_plane_index = [[vma.def_index_function(z_plane[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

t_line_norm_index = [[vma.def_index_function(t_line_norm[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
x_line_norm_index = [[vma.def_index_function(x_line_norm[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

x_line_cropped = [[x_line[i][j][:x_line_index[i][j](cell_width[i][j]/2 - pml_width[i][j])+1] for j in range(len(series[i]))] for i in range(len(series))]
x_line_cropped = [[x_line_cropped[i][j][x_line_index[i][j](-cell_width[i][j]/2 + pml_width[i][j]):] for j in range(len(series[i]))] for i in range(len(series))]

y_plane_cropped = [[y_plane[i][j][:y_plane_index[i][j](cell_width[i][j]/2 - pml_width[i][j])+1] for j in range(len(series[i]))] for i in range(len(series))]
y_plane_cropped = [[y_plane_cropped[i][j][y_plane_index[i][j](-cell_width[i][j]/2 + pml_width[i][j]):] for j in range(len(series[i]))] for i in range(len(series))]

z_plane_cropped = [[z_plane[i][j][:z_plane_index[i][j](cell_width[i][j]/2 - pml_width[i][j])+1] for j in range(len(series[i]))] for i in range(len(series))]
z_plane_cropped = [[z_plane_cropped[i][j][z_plane_index[i][j](-cell_width[i][j]/2 + pml_width[i][j]):] for j in range(len(series[i]))] for i in range(len(series))]

# Get dimnesion para
space_points = [[results_line[i][j].shape[0] for j in range(len(series[i]))] for i in range(len(series))]
inner_space_points = [[len(x_line_cropped[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
time_points = [[results_line[i][j].shape[-1] for j in range(len(series[i]))] for i in range(len(series))]

#%% DATA EXTRACTION

source_results = [[vma.get_source_from_line(results_line_norm[i][j], x_line_norm_index[i][j], source_center[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

if not requires_normalization:
    
    period_results, amplitude_results = norm_period, norm_amplitude
    
else:
    
    period_results = [[vma.get_period_from_source(source_results[i][j], t_line_norm[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
    amplitude_results = [[vma.get_amplitude_from_source(source_results[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
    
    results_plane = [[np.asarray(results_plane[i][j]) / amplitude_results[i][j] for j in range(len(series[i]))] for i in range(len(series))]
    results_line = [[np.asarray(results_line[i][j]) / amplitude_results[i][j] for j in range(len(series[i]))] for i in range(len(series))]

sim_source_results = [[vma.get_source_from_line(results_line[i][j], x_line_index[i][j], source_center[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

zprofile_results = [[vma.get_zprofile_from_plane(results_plane[i][j], y_plane_index[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

zprofile_integral = [[vma.integrate_field_zprofile(zprofile_results[i][j], z_plane_index[i][j],
                                                   cell_width[i][j], pml_width[i][j], period_plane[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

zprofile_max = [[vma.find_zpeaks_zprofile(zprofile_results[i][j], z_plane_index[i][j],
                                          cell_width[i][j], pml_width[i][j])[1] for j in range(len(series[i]))] for i in range(len(series))]

field_peaks_all_index = []
field_peaks_single_index = []
field_peaks_single_amplitude = []
field_peaks_zprofile = []
field_peaks_plane = []
for i in range(len(series)):
    field_peaks_all_index.append([])
    field_peaks_single_index.append([])
    field_peaks_single_amplitude.append([])
    field_peaks_zprofile.append([])
    field_peaks_plane.append([])
    for j in range(len(series[i])):
        ind, mxs, zprof, plan = vma.get_all_field_peaks_from_yzplanes(results_plane[i][j],
                                                                    y_plane_index[i][j], z_plane_index[i][j], 
                                                                    cell_width[i][j], pml_width[i][j])
        field_peaks_all_index[i].append(ind)
        
        # ind, mxs, zprof, plan = vma.get_single_field_peak_from_yzplanes(results_plane[i][j],
        ind, mxs, zprof, plan = vma.get_mean_field_peak_from_yzplanes(results_plane[i][j],
                                                                      y_plane_index[i][j], z_plane_index[i][j], 
                                                                      cell_width[i][j], pml_width[i][j])
        field_peaks_single_index[i].append(ind)
        field_peaks_single_amplitude[i].append(mxs)
        field_peaks_zprofile[i].append(zprof)
        field_peaks_plane[i].append(plan)
del i, j, ind, mxs, zprof, plan

#%% GENERAL PLOT CONFIGURATION

n = len(series)
m = max([len(s) for s in series])

if n*m <= 3:
    subfig_size = 6.5
if n*m <= 6:
    subfig_size = 4.5
else:
    subfig_size = 3.5

empty_proportion = [[2*params[i][j]["empty_width"]/(params[i][j]["cell_width"]-2*params[i][j]["pml_width"]) for j in range(len(series[i]))] for i in range(len(series))]
if (np.array(empty_proportion) > 0.6).any():
    field_area = [0.5 - 0.30/2, 0.72 - 0.30/2, 0.30, 0.30]
    legend_area = (3.5, -2.5)
elif (np.array(empty_proportion) > 0.4).any():
    field_area = [0.5 - 0.38/2, 0.72 - 0.38/2, 0.38, 0.38]
    legend_area = (3.1, -1.9)
else:
    field_area = [0.5 - 0.45/2, 0.72 - 0.45/2, 0.45, 0.45]
    legend_area = (2.6, -1.3)

colors = [sc(np.linspace(0,1,len(s)+2))[2:] 
          for sc, s in zip(series_colormaps, series)]

#%% BASIC CONTROL

plt.figure()
for i in range(len(series)):        
    plt.plot(test_param[i],
             [results_line[i][j].shape[0] for j in range(len(series[i]))], 
             series_markers[i], color=series_colors[i], alpha=0.5,
             markersize=series_markersizes[i])
plt.xlabel(test_param_label)
plt.ylabel(trs.choose("Number of points in whole cell", "Número de puntos en la celda completa"))
plt.legend(series_legend)

plt.savefig(plot_file("Points.png"))

cropped_line = [[vma.crop_field_xprofile(results_line[i][j], x_line_index[i][j], 
                                         cell_width[i][j], pml_width[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

plt.figure()
for i in range(len(series)):        
    plt.plot(test_param[i],
             [cropped_line[i][j].shape[0] for j in range(len(series[i]))], 
             series_markers[i], color=series_colors[i], alpha=0.5,
             markersize=series_markersizes[i])
plt.xlabel(test_param_label)
plt.ylabel(trs.choose("Number of points in real cell", "Número de puntos en la celda real"))
plt.legend(series_legend)

plt.savefig(plot_file("InnerPoints.png"))

plt.figure()
for i in range(len(series)):
    plt.plot(test_param[i],
             [results_line[i][j].shape[-1] for j in range(len(series[i]))], 
             series_markers[i], color=series_colors[i], alpha=0.5,
             markersize=series_markersizes[i])
plt.xlabel(test_param_label)
plt.ylabel(trs.choose("Number of points in time", "Número de puntos en el tiempo"))
plt.legend(series_legend)

plt.savefig(plot_file("TimePoints.png"))

fig = plt.figure()
ax = plt.subplot()
ax2 = plt.twinx()
lines, lines2, lines3 = [], [], []
for i in range(len(series)):
    l, = ax.plot(test_param[i],
                 [params[i][j]["courant"]/resolution[i][j] for j in range(len(series[i]))], 
                 series_markers[i], color=series_colors[i], alpha=0.5,
                 markersize=series_markersizes[i])
    l3, = ax.plot(test_param[i],
                 [period_line[i][j] for j in range(len(series[i]))], 
                 series_markers[i], color=series_colors[i], alpha=0.5,
                 markersize=series_markersizes[i], fillstyle="top")
    l2, = ax.plot(test_param[i],
                  [1/resolution[i][j] for j in range(len(series[i]))], 
                  series_markers[i], color=series_colors[i], fillstyle="none",
                  markersize=series_markersizes[i])
    lines.append(l)
    lines2.append(l2)
    lines3.append(l3)
plt.xlabel(test_param_label)
ax.set_ylabel(trs.choose("Time Minimum Division [MPu]", "Mínima división del tiempo [uMP]"))
ax2.set_ylabel(trs.choose("Space Minimum Division [MPu]", "Mínima división del espacio [uMP]"))
plt.legend([*lines, *lines3, *lines2], 
           [*[s + r" $\Delta t$" for s in series_legend], 
           *[s + r" $\Delta t_{line}$" for s in series_legend],
           *[s + r" $\Delta r$" for s in series_legend]],
           ncol=3)
plt.savefig(plot_file("MinimumDivision.png"))

fig, [ax, ax2] = plt.subplots(nrows=2, sharex=True, gridspec_kw={"hspace":0})
for i in range(len(series)):
    ax.plot(test_param[i],
            [resolution[i][j] for j in range(len(series[i]))], 
            series_markers[i], color=series_colors[i], alpha=0.5,
            markersize=series_markersizes[i])
    ax2.plot(test_param[i],
             [resolution_wlen[i][j] for j in range(len(series[i]))], 
             series_markers[i], color=series_colors[i], alpha=0.5,
             markersize=series_markersizes[i], fillstyle="none")
plt.xlabel(test_param_label)
ax.set_ylabel(trs.choose("Resolution", "Resolución") + r" [points/$\Delta r$]")
ax2.set_ylabel(trs.choose("Resolution", "Resolución") + r" [points/$\lambda$]")

first_legend = ax.legend(series_legend, loc="center left")
second_legend = ax2.legend(series_legend, loc="center right")

#%% SHOW SOURCE AND FOURIER USED FOR NORMALIZATION

fig = plt.figure()
plt.suptitle(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
             plot_title_ending)

for i in range(len(series)):
    for j in range(len(series[i])):
        l, = plt.plot(t_line_norm[i][j] / period_results[i][j], 
                      source_results[i][j] / amplitude_results[i][j],
                      label=series_legend[i] + " " + series_label[i](series[i][j]),
                      color=colors[i][j],
                      linestyle=series_linestyles[i])
plt.xlabel(trs.choose("Time in multiples of period", "Tiempo en múltiplos del período"))
plt.ylabel(trs.choose(r"Electric Field $E_z(y=z=0)$",
                      r"Campo eléctrico $E_z(y=z=0)$"))
box = fig.axes[0].get_position()
box.x1 = box.x1 - .25 * (box.x1 - box.x0)
fig.axes[0].set_position(box)
leg = plt.legend(columnspacing=-0.5, bbox_to_anchor=(1.5, .5), loc="center right")

fig.set_size_inches([9.28, 4.8])
plt.savefig(plot_file("Source.png"))
        
fourier = [[np.abs(np.fft.rfft(source_results[i][j] / amplitude_results[i][j])) for j in range(len(series[i]))] for i in range(len(series))]
fourier_freq = [[np.fft.rfftfreq(len(source_results[i][j]), d=period_line[i][j])  for j in range(len(series[i]))] for i in range(len(series))]
fourier_wlen = [[from_um_factor[i][j] * 1e3 / fourier_freq[i][j]  for j in range(len(series[i]))] for i in range(len(series))]
fourier_best = [[wlen[i][j] * from_um_factor[i][j] * 1e3 for j in range(len(series[i]))] for i in range(len(series))]
# fourier_max_wlen = [[fourier_wlen[i][j][ np.argmax(fourier[i][j]) ]  for j in range(len(series[i]))] for i in range(len(series))]

fig = plt.figure()
plt.title(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
          plot_title_ending)
lines = []
for i in range(len(series)):
    for j in range(len(series[i])):
        plt.plot(fourier_wlen[i][j], fourier[i][j],
                 label=series_legend[i] + " " + series_label[i](series[i][j]),
                 color=colors[i][j],
                 linestyle=series_linestyles[i])
plt.xlabel(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
plt.ylabel(trs.choose(r"Electric Field Fourier $\mathcal{F}\;(E_z)$",
                      r"Transformada del campo eléctrico $\mathcal{F}\;(E_z)$"))
box = fig.axes[0].get_position()
box.x1 = box.x1 - .25 * (box.x1 - box.x0)
fig.axes[0].set_position(box)
leg = plt.legend(columnspacing=-0.5, bbox_to_anchor=(1.5, .5), loc="center right")

fig.set_size_inches([9.28, 4.8])

# plt.annotate(trs.choose(f"Maximum at {fourier_max_wlen:.2f} nm",
#                         f"Máximo en {fourier_max_wlen:.2f} nm"),
#              (5, 5), xycoords='figure points')

plt.savefig(plot_file("SourceFFT.png"))

plt.xlim([350, 850])
        
plt.savefig(plot_file("SourceFFTZoom.png"))

#%% WAVELENGTH CHECK

fourier_max_wlen = [[fourier_wlen[i][j][ np.argmax(fourier[i][j]) ]  for j in range(len(series[i]))] for i in range(len(series))]
fourier_max_best = [[fourier_wlen[i][j][ np.argmin(np.abs(fourier_wlen[i][j] - fourier_best[i][j])) ]  for j in range(len(series[i]))] for i in range(len(series))]

colors = [sc(np.linspace(0,1,len(s)+2))[2:] 
          for sc, s in zip(series_colormaps, series)]

plt.figure()
plt.suptitle(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
             plot_title_ending)
for i in range(len(series)):
    plt.plot(test_param[i], 
             100 * ( np.array(fourier_max_wlen[i]) - np.array(fourier_max_best[i]) ) / np.array(fourier_max_best[i]), 
             color=series_colors[i], marker=series_markers[i], alpha=0.4,
             linestyle="", markeredgewidth=0, markersize=series_markersizes[i])
plt.axhline(0, color="k", linewidth=0.5, label="")
plt.legend(series_legend)
plt.xlabel(test_param_label)
plt.ylabel(trs.choose("Maximum Wavelength Percentual Variation", 
                      "Variación porcentual de la longitud de onda máxima\n") + 
           r"$\lambda_{max} = \argmax [\mathcal{F}\;(E_z)]$ [%]")

plt.tight_layout()
vs.saveplot(plot_file("LambdaVariation.png"), overwrite=True)

#%% SHOW SOURCE AND FOURIER DURING SIMULATION

# colors = [["C0"], ["C4"], ["C3"]]
colors = [sc(np.linspace(0,1,len(s)+2))[2:] 
          for sc, s in zip(series_colormaps, series)]

fig = plt.figure()
plt.title(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
          plot_title_ending)

lines = []
for i in range(len(series)):
    for j in range(len(series[i])):
        l, = plt.plot(t_line[i][j]/period_results[i][j], 
                      sim_source_results[i][j],
                      label=series_legend[i] + " " + series_label[i](series[i][j]),
                      linestyle=series_linestyles[i],
                      color=colors[i][j])            
        lines.append(l)
plt.xlabel(trs.choose("Time in multiples of period", "Tiempo en múltiplos del período"))
plt.ylabel(trs.choose(r"Electric Field $E_z(y=z=0)$",
                      r"Campo eléctrico $E_z(y=z=0)$"))
box = fig.axes[0].get_position()
box.x1 = box.x1 - .25 * (box.x1 - box.x0)
fig.axes[0].set_position(box)
leg = plt.legend(columnspacing=-0.5, bbox_to_anchor=(1.5, .5), loc="center right")

fig.set_size_inches([9.28, 4.8])
plt.savefig(plot_file("SimSource.png"))
        
fourier = [[np.abs(np.fft.rfft(sim_source_results[i][j])) for j in range(len(series[i]))] for i in range(len(series))]
fourier_freq = [[np.fft.rfftfreq(len(sim_source_results[i][j]), d=period_line[i][j])  for j in range(len(series[i]))] for i in range(len(series))]
fourier_wlen = [[from_um_factor[i][j] * 1e3 / fourier_freq[i][j]  for j in range(len(series[i]))] for i in range(len(series))]
# fourier_max_wlen = [[fourier_wlen[i][j][ np.argmax(fourier[i][j]) ]  for j in range(len(series[i]))] for i in range(len(series))]

fig = plt.figure()
plt.title(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
          plot_title_ending)
lines = []
for i in range(len(series)):
    for j in range(len(series[i])):
        plt.plot(fourier_wlen[i][j], fourier[i][j],
                 label=series_legend[i] + " " + series_label[i](series[i][j]),
                 linestyle=series_linestyles[i],
                 color=colors[i][j])
plt.xlabel(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
plt.ylabel(trs.choose(r"Electric Field Fourier $\mathcal{F}\;(E_z)$",
                      r"Transformada del campo eléctrico $\mathcal{F}\;(E_z)$"))
box = fig.axes[0].get_position()
box.x1 = box.x1 - .25 * (box.x1 - box.x0)
fig.axes[0].set_position(box)
leg = plt.legend(columnspacing=-0.5, bbox_to_anchor=(1.5, .5), loc="center right")

# plt.annotate(trs.choose(f"Maximum at {fourier_max_wlen:.2f} nm",
#                         f"Máximo en {fourier_max_wlen:.2f} nm"),
#              (5, 5), xycoords='figure points')

fig.set_size_inches([9.28, 4.8])
plt.savefig(plot_file("SimSourceFFT.png"))

plt.xlim([350, 850])
        
plt.savefig(plot_file("SimSourceFFTZoom.png"))

#%% SHOW FIELD PEAKS INTENSIFICATION OSCILLATIONS

fig = plt.figure()        
plt.title(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
          plot_title_ending)

for i in range(len(series)):
    for j in range(len(series[i])):
            plt.plot(t_line[i][j] / period_results[i][j], 
                     zprofile_integral[i][j], color=colors[i][j],
                     label=series_legend[i] + " " + series_label[i](series[i][j]),
                     linestyle=series_linestyles[i])
plt.xlabel(trs.choose("Time [MPu]", "Tiempo [uMP]"))
plt.ylabel(trs.choose(r"Electric Field Integral $\int E_z(z) \; dz$ [a.u.]",
                      r"Integral del campo eléctrico $\int E_z(z) \; dz$ [u.a.]"))

box = fig.axes[0].get_position()
box.x1 = box.x1 - .25 * (box.x1 - box.x0)
fig.axes[0].set_position(box)
leg = plt.legend(columnspacing=-0.5, bbox_to_anchor=(1.5, .5), loc="center right")

fig.set_size_inches([9.28, 4.8])
plt.savefig(plot_file("Integral.png"))

fig = plt.figure()        
plt.title(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
          plot_title_ending)

for i in range(len(series)):
    for j in range(len(series[i])):
            plt.plot(t_line[i][j]/period_results[i][j], 
                     zprofile_max[i][j],
                     color=colors[i][j],
                     label=series_legend[i] + " " + series_label[i](series[i][j]),
                     linestyle=series_linestyles[i])
plt.xlabel(trs.choose("Time [MPu]", "Tiempo [uMP]"))
plt.ylabel(trs.choose(r"Electric Field Maximum $max[ E_z(z) ]$",
                      r"Máximo del campo eléctrico $max[ E_z(z) ]$"))

box = fig.axes[0].get_position()
box.x1 = box.x1 - .25 * (box.x1 - box.x0)
fig.axes[0].set_position(box)
leg = plt.legend(columnspacing=-0.5, bbox_to_anchor=(1.5, .5), loc="center right")

fig.set_size_inches([9.28, 4.8])
plt.savefig(plot_file("Maximum.png"))

#%% MAXIMUM WAVELENGTH VS TEST PARAMETER

plt.figure()
plt.suptitle(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
             plot_title_ending)
for i in range(len(series)):
    plt.plot(test_param[i], 
             field_peaks_single_amplitude[i], 
             color=series_colors[i], marker=series_markers[i], alpha=0.4,
             linestyle="", markeredgewidth=0, markersize=series_markersizes[i])
plt.axhline(0, color="k", linewidth=0.5, label="")
plt.legend(series_legend)
plt.xlabel(test_param_label)
plt.ylabel(trs.choose(r"Electric Field Maximum $max[ E_z(z) ]$",
                      r"Máximo del campo eléctrico $max[ E_z(z) ]$"))

plt.tight_layout()
vs.saveplot(plot_file("MaximumTestParam.png"), overwrite=True)

#%% GET THEORY (SCATTERING)

def wlen_range(material, surrounding_index):
    if material=="Au":
        if surrounding_index == 1: return (450, 750)
        elif surrounding_index == 1.33: return (500, 800)
    else:
        raise ValueError("Please, expand this function.")

scatt_max_wlen_theory = [[vmt.max_scatt_meep(r[i][j] * from_um_factor[i][j] * 1e3, 
                                             material[i][j], 
                                             paper[i][j], 
                                             wlen_range(material[i][j],
                                                        index[i][j]),
                                             surrounding_index=index[i][j])[0] 
                          for j in range(len(series[i]))] for i in range(len(series))]

scatt_max_wlen_predict = [wlen[i][np.argmin(np.abs([wlen[i][j] * from_um_factor[i][j] * 1e3 - scatt_max_wlen_theory[i][j] for j in range(len(series[i]))]))] for i in range(len(series))]

#%% GET THEORY (FIELD)

rvec = []
for i in range(len(series)):
    rvec.append([])
    for j in range(len(series[i])):
        naux = zprofile_results[i][j].shape[-1]
        aux = np.zeros((naux, 3))
        aux[:,2] = np.linspace(-cell_width[i][j]/2 + pml_width[i][j], 
                                cell_width[i][j]/2 - pml_width[i][j], 
                                naux)
        rvec[-1].append(aux)
del aux, naux

E0 = np.array([0,0,1])

zprofile_cm_theory = []
zprofile_ku_theory = []
for i in range(len(series)):
    zprofile_cm_theory.append([])
    zprofile_ku_theory.append([])
    for j in range(len(series[i])):
        medium = vmt.import_medium(material=material[i][j], paper=paper[i][j],
                                   from_um_factor=from_um_factor[i][j])
        epsilon = medium.epsilon(1/wlen[i][j])[0,0]
        # E0 = np.array([0, 0, is_max_in_z_profile[j] * np.real(in_z_profile[j][max_in_z_profile[j], 0])])
        alpha_cm = vt.alpha_Clausius_Mosotti(epsilon, r[i][j], epsilon_ext=index[i][j]**2)
        alpha_ku = vt.alpha_Kuwata(epsilon, wlen[i][j], r[i][j], epsilon_ext=index[i][j]**2)
        theory_cm = np.array([vt.E(epsilon, alpha_cm, E0, 
                                   rv, r[i][j], epsilon_ext=index[i][j]**2) 
                              for rv in rvec[i][j]])[:,-1]
        theory_ku = np.array([vt.E(epsilon, alpha_ku, E0, 
                                   rv, r[i][j], epsilon_ext=index[i][j]**2) 
                              for rv in rvec[i][j]])[:,-1]
        zprofile_cm_theory[i].append(theory_cm)
        zprofile_ku_theory[i].append(theory_ku)
        
#%% PLOT MAXIMUM INTENSIFICATION PROFILE (THEORY)

fig = plt.figure(figsize=(n*subfig_size, subfig_size))
axes = fig.subplots(ncols=len(series), sharex=True, sharey=True,
                    gridspec_kw={"wspace":0, "hspace":0})

plt.suptitle(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
             plot_title_ending)
l_series = []
l_origin = []
for i in range(len(series)):
    for j in range(len(series[i])):
        axes[i].set_title(series_legend[i] + " " + series_label[i](series[i][j]))
        
        axes[i].axvline(r[i][j] * from_um_factor[i][j] * 1e3, 
                        color="k", linestyle="dotted", linewidth=.8)
        axes[i].axvline(-r[i][j] * from_um_factor[i][j] * 1e3, 
                        color="k", linestyle="dotted", linewidth=.8)
        
        l_cm, = axes[i].plot(rvec[i][j][:,-1] * from_um_factor[i][j] * 1e3, 
                         np.abs(zprofile_cm_theory[i][j]),
                         label=series_label[i](series[i][j]),
                         color=series_ind_colors[i][j], #linestyle="dashed",
                         alpha=0.5)
        l_ku, = axes[i].plot(rvec[i][j][:,-1]  * from_um_factor[i][j] * 1e3, 
                         np.abs(zprofile_ku_theory[i][j]),
                         label=series_label[i](series[i][j]),
                         color=series_ind_colors[i][j], linestyle="dashed",
                         alpha=1)
        l_series.append(l_cm)
        
        axes[i].axvline(r[i][j] * from_um_factor[i][j] * 1e3, 
                        color="k", linestyle="dotted", linewidth=.8)
        axes[i].axvline(-r[i][j] * from_um_factor[i][j] * 1e3, 
                        color="k", linestyle="dotted", linewidth=.8)
        
        axes[i].axhline(0, color="k", linewidth=.5)
        
        if i == 0 and j == int(len(series[0])/2):
            l_origin = [l_cm, l_ku]
        axes[i].set_xlabel(trs.choose("Position $Z$ [nm]", "Posición $Z$ [nm]"))
        
        axes[i].xaxis.set_minor_locator(AutoMinorLocator())
        axes[i].yaxis.set_minor_locator(AutoMinorLocator())
        axes[i].grid(True, axis="y", which="both", alpha=.3)
        
axes[0].set_ylabel(trs.choose(r"Electric Field $E_z(y=z=0)$",
                              r"Campo eléctrico $E_z(y=z=0)$"))

for ax in axes:
    box = ax.get_position()
    box_height = box.y1 - box.y0
    box.y1 = box.y1 - .02 * box_height
    box.y0 = box.y0 + .13 * box_height
    ax.set_position(box)

first_legend = axes[0].legend(l_origin, trs.choose(["CM Theory", "Ku Theory"],
                                               ["Teoría CM", "Teoría Ku"]),
                              loc="center", frameon=False, bbox_to_anchor=(1,-.2), ncol=2)
second_legend = axes[0].legend(
    l_series[:len(series[0])], 
    [l.get_label() for l in l_series[:len(series[0])]],
    bbox_to_anchor=(1, -.28),
    loc="center", ncol=len(series[0]), frameon=False)
axes[0].add_artist(first_legend)

plt.savefig(plot_file("FieldProfileTheory.png"))

#%% PLOT MAXIMUM INTENSIFICATION PROFILE (DATA)

# colors = [["C0"], ["C4"], ["C3"]]
colors = [sc(np.linspace(0,1,len(s)+2))[2:] 
          for sc, s in zip(series_colormaps, series)]

plt.figure()
plt.suptitle(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
             plot_title_ending)
for i in range(len(series)):
    for j in range(len(series[i])):
        plt.plot(z_plane_cropped[i][j] * from_um_factor[i][j] * 1e3, 
                 field_peaks_zprofile[i][j],
                 label=series_legend[i] + " " + series_label[i](series[i][j]),
                 color=colors[i][j])
plt.xlabel(trs.choose("Position $Z$ [nm]", "Posición $Z$ [nm]"))
plt.ylabel(trs.choose(r"Electric Field $E_z(y=z=0)$",
                      r"Campo eléctrico $E_z(y=z=0)$"))
plt.legend()

plt.savefig(plot_file("FieldProfileData.png"))

#%% PLOT MAXIMUM INTENSIFICATION PROFILE (ALL)

# colors = [["C0"], ["C4"], ["C3"]]
colors = [sc(np.linspace(0,1,len(s)+2))[2:] 
          for sc, s in zip(series_colormaps, series)]

plt.figure()
plt.suptitle(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
             plot_title_ending)
l_origin = []
l_series = []
for i in range(len(series)):
    for j in range(len(series[i])):
        l_mp, = plt.plot(z_plane_cropped[i][j] * from_um_factor[i][j] * 1e3, 
                         field_peaks_zprofile[i][j] ,
                         label=series_legend[i] + " " + series_label[i](series[i][j]),
                         color=colors[i][j])
        l_cm, = plt.plot(rvec[i][j][:,-1] * from_um_factor[i][j] * 1e3, 
                         np.abs(zprofile_cm_theory[i][j]),
                         color=colors[i][j], linestyle="dashed")
        l_ku, = plt.plot(rvec[i][j][:,-1]  * from_um_factor[i][j] * 1e3, 
                         np.abs(zprofile_ku_theory[i][j]),
                         color=colors[i][j], linestyle="dotted")
        l_series.append(l_mp)
        if i == 0 and j == len(series[i])-1:
            l_origin = [l_mp, l_cm, l_ku]
plt.xlabel(trs.choose("Position $Z$ [nm]", "Posición $Z$ [nm]"))
plt.ylabel(trs.choose(r"Electric Field $E_z(y=z=0)$",
                      r"Campo eléctrico $E_z(y=z=0)$"))

plt.legend(ncol=2)

first_legend = plt.legend(l_origin, trs.choose(["MEEP Data", "CM Theory", "Ku Theory"],
                                               ["Data MEEP", "Teoría CM", "Teoría Ku"]),
                          loc="lower right")
second_legend = plt.legend(
    l_series, 
    [l.get_label() for l in l_series],
    loc="upper center")
plt.gca().add_artist(first_legend)

plt.savefig(plot_file("FieldProfileAll.png"))

#%% PLOT MAXIMUM INTENSIFICATION PROFILE (SUBPLOTS)

n = len(series)
m = max([len(s) for s in series])

if n*m <= 3:
    subfig_size = 6.5
if n*m <= 6:
    subfig_size = 4.5
else:
    subfig_size = 3.5

fig = plt.figure(figsize=(m*subfig_size, n*subfig_size))
axes = fig.subplots(ncols=m, nrows=n, sharex=True, sharey=True, gridspec_kw={"wspace":0})
plt.suptitle(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
             plot_title_ending)
        
for i in range(len(series)):
    for j in range(len(series[i])):
        axes[i][j].axvline(r[i][j] * from_um_factor[i][j] * 1e3, 
                           color="k", linestyle="dotted")
        axes[i][j].axvline(-r[i][j] * from_um_factor[i][j] * 1e3, 
                           color="k", linestyle="dotted")
        l_mp, = axes[i][j].plot(z_plane_cropped[i][j] * from_um_factor[i][j] * 1e3, 
                                field_peaks_zprofile[i][j],
                                label=series_label[i](series[i][j]),
                                color=colors[i][j])
        l_cm, = axes[i][j].plot(rvec[i][j][:,-1] * from_um_factor[i][j] * 1e3, 
                                np.abs(zprofile_cm_theory[i][j]),
                                color=colors[i][j], linestyle="dashed")
        l_ku, = axes[i][j].plot(rvec[i][j][:,-1]  * from_um_factor[i][j] * 1e3, 
                                np.abs(zprofile_ku_theory[i][j]),
                                color=colors[i][j], linestyle="dotted")
        axes[i][j].axhline(0, color="k", linewidth=.5)
        axes[i][j].axvline(0, color="k", linewidth=.5)
        if i==len(series)-1:
            axes[i][j].set_xlabel(trs.choose("Position $Z$ [nm]", "Posición $Z$ [nm]"))
        if j==0:
            axes[i][j].set_ylabel(trs.choose(r"Electric Field $E_z(y=z=0)$",
                                              r"Campo eléctrico $E_z(y=z=0)$"))
        if i==len(series)-1 and j==len(series[i])-1:
            axes[i][j].legend([l_mp, l_cm, l_ku], 
                              trs.choose(["MEEP Data", "CM Theory", "Ku Theory"],
                                         ["Data MEEP", "Teoría CM", "Teoría Ku"]),
                              loc="upper center")
        
        axes[i][j].xaxis.set_minor_locator(AutoMinorLocator())
        axes[i][j].yaxis.set_minor_locator(AutoMinorLocator())
        # axes[i][j].grid(True, axis="y", which="major")
        axes[i][j].grid(True, axis="y", which="both", alpha=.3)
        axes[i][j].set_title(series_legend[i] + " " + series_label[i](series[i][j]))

plt.savefig(plot_file("FieldProfileAllSubplots.png"))

#%% PLOT MAXIMUM INTENSIFICATION FIELD (SUBPLOTS)

fig = plt.figure(figsize=(m*subfig_size, n*subfig_size))
axes = fig.subplots(ncols=m, nrows=n)
plt.suptitle(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
             plot_title_ending)
lims = [np.min([np.min([np.min(field_peaks_plane[i][j]) for j in range(len(series[i]))]) for i in range(len(series))]),
        np.max([np.max([np.max(field_peaks_plane[i][j]) for j in range(len(series[i]))]) for i in range(len(series))])]
lims = max([abs(l) for l in lims])
lims = [-lims, lims]

for i in range(len(series)):
    for j in range(len(series[i])):
        ims = axes[i][j].imshow(field_peaks_plane[i][j],
                                cmap='RdBu', #interpolation='spline36', 
                                vmin=lims[0], vmax=lims[1],
                                extent=[min(y_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3,
                                        max(y_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3,
                                        min(z_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3,
                                        max(z_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3])
        if i==len(series)-1:
            axes[i][j].set_xlabel(trs.choose("Position $Y$ [nm]", "Posición $Y$ [nm]"))
        if j==0:
            axes[i][j].set_ylabel(trs.choose("Position $Z$ [nm]", "Posición $Z$ [nm]"))
        if j==len(series[i])-1:
            cax = axes[i][j].inset_axes([1.04, 0, 0.07, 1], #[1.04, 0.2, 0.05, 0.6], 
                                        transform=axes[i][j].transAxes)
            cbar = fig.colorbar(ims, ax=axes[i][j], cax=cax)
            cbar.set_label(trs.choose("Electric field $E_z$",
                                      "Campo eléctrico $E_z$"))
        # if i==0:
        #     axes[i][j].set_title(series_label[i](series[i][j]), y = 1.05)
        # else:
        #     axes[i][j].set_title(series_label[i](series[i][j]), y = -.3)
        axes[i][j].set_title(series_legend[i] + " " + series_label[i](series[i][j]))
plt.savefig(plot_file("FieldPlaneAll.png"))

#%% PLOT PROFILE AND PLANE (SUBPLOTS)

vertical_plot = False

if vertical_plot:
    fig = plt.figure(figsize=(n*subfig_size, m*subfig_size))
    axes = fig.subplots(ncols=n, nrows=m, sharex=True, sharey=True, 
                        gridspec_kw={"wspace":0, "hspace":0})
    axes = axes.T
else:
    fig = plt.figure(figsize=(m*subfig_size, n*subfig_size))
    axes = fig.subplots(ncols=m, nrows=n, sharex=True, sharey=True, 
                        gridspec_kw={"wspace":0, "hspace":0})
plt.suptitle(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
             plot_title_ending)

lims = [np.min([np.min([np.min(field_peaks_plane[i][j]) for j in range(len(series[i]))]) for i in range(len(series))]),
        np.max([np.max([np.max(field_peaks_plane[i][j]) for j in range(len(series[i]))]) for i in range(len(series))])]
lims = max([abs(l) for l in lims])
lims = [-lims, lims]
        
l_meep = []
l_cmos = []
l_kuwa = []
axes_field = []
for i in range(len(series)):
    axes_field.append([])
    l_meep.append([])
    l_cmos.append([])
    l_kuwa.append([])
    for j in range(len(series[i])):
        axes[i][j].axvline(r[i][j] * from_um_factor[i][j] * 1e3, 
                           color="k", linestyle="dotted")
        axes[i][j].axvline(-r[i][j] * from_um_factor[i][j] * 1e3, 
                           color="k", linestyle="dotted")
        l_mp, = axes[i][j].plot(z_plane_cropped[i][j] * from_um_factor[i][j] * 1e3, 
                                field_peaks_zprofile[i][j],
                                label=series_label[i](series[i][j]),
                                color="k", linewidth=1.4) # color=colors[i][j]
        l_cm, = axes[i][j].plot(rvec[i][j][:,-1] * from_um_factor[i][j] * 1e3, 
                                np.abs(zprofile_cm_theory[i][j]),
                                linestyle=(0, (5, 1)), color="darkorchid") # color=colors[i][j]
        l_ku, = axes[i][j].plot(rvec[i][j][:,-1]  * from_um_factor[i][j] * 1e3, 
                                np.abs(zprofile_ku_theory[i][j]),
                                linestyle=(0, (5, 3)), color="deeppink") # color=colors[i][j]
        l_meep[-1].append(l_mp)
        l_cmos[-1].append(l_cm)
        l_kuwa[-1].append(l_ku)
        
        xlims = axes[i][j].get_xlim()
        axes[i][j].axhline(0, color="k", linewidth=.5)
        axes[i][j].axvline(0, color="k", linewidth=.5)
        axes[i][j].set_xlim(xlims)
        axes[i][j].set_title(series_label[i](series[i][j]))
        
        if vertical_plot:
            if j==len(series[i])-1:
                axes[i][j].set_xlabel(trs.choose("Position $Z$ [nm]", "Posición $Z$ [nm]"))
            if i==0:
                axes[i][j].set_ylabel(trs.choose(r"Electric Field $E_z(y=z=0)$",
                                                  r"Campo eléctrico $E_z(y=z=0)$"))
        else:
            if i==len(series)-1:
                axes[i][j].set_xlabel(trs.choose("Position $Z$ [nm]", "Posición $Z$ [nm]"))
            if j==0:
                axes[i][j].set_ylabel(trs.choose(r"Electric Field $E_z(y=z=0)$",
                                                  r"Campo eléctrico $E_z(y=z=0)$"))
        
        axes[i][j].xaxis.set_minor_locator(AutoMinorLocator())
        axes[i][j].yaxis.set_minor_locator(AutoMinorLocator())
        axes[i][j].grid(True, axis="y", which="both", alpha=.3)
        if vertical_plot:
            if i==0:
                axes[i][j].set_title(series_legend[i] + " " + series_label[i](series[i][j]), y = 1.02)
            elif i==len(series)-1:
                axes[i][j].set_title(series_legend[i] + " " + series_label[i](series[i][j]), y = -.25)
        else:
            if j==0:
                axes[i][j].set_title(series_legend[i] + " " + series_label[i](series[i][j]), y = 1.02)
            elif j==len(series[i])-1:
                axes[i][j].set_title(series_legend[i] + " " + series_label[i](series[i][j]), y = -.25)
        
        ax_field = vp.add_subplot_axes(axes[i][j], field_area)
        axes_field[-1].append(ax_field)
        
        ims = ax_field.imshow(field_peaks_plane[i][j].T,
                              cmap='RdBu', #interpolation='spline36', 
                              vmin=lims[0], vmax=lims[1],
                              extent=[min(y_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3,
                                      max(y_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3,
                                      min(z_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3,
                                      max(z_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3])
        ax_field.set_xticks([])
        ax_field.set_yticks([])
        
        if i==len(series)-1 and j==len(series[i])-1:
            if vertical_plot: ncax=m
            else: ncax=n
            cax = axes[i][j].inset_axes([1.04, 0, 0.07, ncax], #[1.04, 0, 0.07, 1], 
                                        transform=axes[i][j].transAxes)
            cbar = fig.colorbar(ims, ax=axes[i][j], cax=cax)
            cbar.set_label(trs.choose("Electric field $E_z$",
                                      "Campo eléctrico $E_z$"))
            cbar.ax.minorticks_on()
        
leg = plt.legend(
    [l_meep[-1][-1], l_cmos[-1][-1], l_kuwa[-1][-1]],
    trs.choose(["MEEP Data", "CM Theory", "Ku Theory"], ["Data MEEP", "Teoría CM", "Teoría Ku"]),
    bbox_to_anchor=legend_area, #(2.5, 1.4), 
    loc="center right", frameon=False)

fig.set_size_inches([13.5,  7.9]) # ([17.5,  7.9])
plt.savefig(plot_file("AllSubplots.png"))

#%% PLOT MAXIMUM INTENSIFICATION PROFILE (DATA) [ABSOLUTE VALUE]

# colors = [["C0"], ["C4"], ["C3"]]
colors = [sc(np.linspace(0,1,len(s)+2))[2:] 
          for sc, s in zip(series_colormaps, series)]

plt.figure()
plt.suptitle(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
             plot_title_ending)
for i in range(len(series)):
    for j in range(len(series[i])):
        plt.plot(z_plane_cropped[i][j], np.abs(field_peaks_zprofile[i][j]),
                 label=series_legend[i] + " " + series_label[i](series[i][j]),
                 color=colors[i][j], linestyle=series_linestyles[i])
plt.xlabel(trs.choose("Position $Z$ [nm]", "Posición $Z$ [nm]"))
plt.ylabel(trs.choose(r"Electric Field $|E_z|(y=z=0)$",
                      r"Campo eléctrico $|E_z|(y=z=0)$"))
plt.legend()

plt.savefig(plot_file("AbsFieldProfileData.png"))

#%% PLOT MAXIMUM INTENSIFICATION PROFILE (ALL) [ABSOLUTE VALUE]

# colors = [["C0"], ["C4"], ["C3"]]
colors = [sc(np.linspace(0,1,len(s)+2))[2:] 
          for sc, s in zip(series_colormaps, series)]

plt.figure()
plt.suptitle(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
             plot_title_ending)
l_origin = []
l_series = []
for i in range(len(series)):
    for j in range(len(series[i])):
        l_mp, = plt.plot(z_plane_cropped[i][j] * from_um_factor[i][j] * 1e3, 
                         np.abs(field_peaks_zprofile[i][j]),
                         label=series_legend[i] + " " + series_label[i](series[i][j]),
                         color=colors[i][j])
        l_cm, = plt.plot(rvec[i][j][:,-1] * from_um_factor[i][j] * 1e3, 
                         np.abs(zprofile_cm_theory[i][j]),
                         color=colors[i][j], linestyle="dashed")
        l_ku, = plt.plot(rvec[i][j][:,-1]  * from_um_factor[i][j] * 1e3, 
                         np.abs(zprofile_ku_theory[i][j]),
                         color=colors[i][j], linestyle="dotted")
        l_series.append(l_mp)
        if i == 0 and j == len(series[i])-1:
            l_origin = [l_mp, l_cm, l_ku]
plt.xlabel(trs.choose("Position $Z$ [nm]", "Posición $Z$ [nm]"))
plt.ylabel(trs.choose(r"Electric Field $|E_z|(y=z=0)$",
                      r"Campo eléctrico $|E_z|(y=z=0)$"))

plt.legend(ncol=2)

first_legend = plt.legend(l_origin, trs.choose(["MEEP Data", "CM Theory", "Ku Theory"],
                                               ["Data MEEP", "Teoría CM", "Teoría Ku"]),
                          loc="lower right")
second_legend = plt.legend(
    l_series, 
    [l.get_label() for l in l_series],
    loc="upper center")
plt.gca().add_artist(first_legend)

plt.savefig(plot_file("AbsFieldProfileAll.png"))

#%% PLOT MAXIMUM INTENSIFICATION PROFILE (SUBPLOTS) [ABSOLUTE VALUE]

fig = plt.figure(figsize=(m*subfig_size, n*subfig_size))
axes = fig.subplots(ncols=m, nrows=n, sharex=True, sharey=True, gridspec_kw={"wspace":0})
plt.suptitle(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
             plot_title_ending)
        
for i in range(len(series)):
    for j in range(len(series[i])):
        axes[i][j].axvline(r[i][j] * from_um_factor[i][j] * 1e3, 
                           color="k", linestyle="dotted")
        axes[i][j].axvline(-r[i][j] * from_um_factor[i][j] * 1e3, 
                           color="k", linestyle="dotted")
        l_mp, = axes[i][j].plot(z_plane_cropped[i][j] * from_um_factor[i][j] * 1e3, 
                                np.abs(field_peaks_zprofile[i][j]),
                                label=series_label[i](series[i][j]),
                                color=colors[i][j])
        l_cm, = axes[i][j].plot(rvec[i][j][:,-1] * from_um_factor[i][j] * 1e3, 
                                np.abs(zprofile_cm_theory[i][j]),
                                color=colors[i][j], linestyle="dashed")
        l_ku, = axes[i][j].plot(rvec[i][j][:,-1]  * from_um_factor[i][j] * 1e3, 
                                np.abs(zprofile_ku_theory[i][j]),
                                color=colors[i][j], linestyle="dotted")
        xlims = axes[i][j].get_xlim()
        axes[i][j].axhline(0, color="k", linewidth=.5)
        axes[i][j].axvline(0, color="k", linewidth=.5)
        axes[i][j].set_xlim(xlims)
        axes[i][j].set_title(series_label[i](series[i][j]))
        if i==len(series)-1:
            axes[i][j].set_xlabel(trs.choose("Position $Z$ [nm]", "Posición $Z$ [nm]"))
        if j==0:
            axes[i][j].set_ylabel(trs.choose(r"Electric Field $|E_z|(y=z=0)$",
                                              r"Campo eléctrico $|E_z|(y=z=0)$"))
        if i==len(series)-1 and j==len(series[i])-1:
            axes[i][j].legend([l_mp, l_cm, l_ku], 
                              trs.choose(["MEEP Data", "CM Theory", "Ku Theory"],
                                         ["Data MEEP", "Teoría CM", "Teoría Ku"]),
                              loc="upper center")
        
        axes[i][j].xaxis.set_minor_locator(AutoMinorLocator())
        axes[i][j].yaxis.set_minor_locator(AutoMinorLocator())
        # axes[i][j].grid(True, axis="y", which="major")
        axes[i][j].grid(True, axis="y", which="both", alpha=.3)
        axes[i][j].set_title(series_legend[i] + " " + series_label[i](series[i][j]))

plt.savefig(plot_file("AbsFieldProfileAllSubplots.png"))

#%% PLOT PROFILE AND PLANE (SUBPLOTS) [ABSOLUTE VALUE]

fig = plt.figure(figsize=(m*subfig_size, n*subfig_size))
axes = fig.subplots(ncols=m, nrows=n, sharex=True, sharey=True, 
                    gridspec_kw={"wspace":0, "hspace":0})
plt.suptitle(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
             plot_title_ending)

# lims = [np.min([np.min([np.min(np.power(np.abs(field_peaks_plane[i][j]),2)) for j in range(len(series[i]))]) for i in range(len(series))]), 
#         np.max([np.max([np.max(np.power(np.abs(field_peaks_plane[i][j]),2)) for j in range(len(series[i]))]) for i in range(len(series))])] 
lims = [np.min([np.min([np.min(np.abs(field_peaks_plane[i][j])) for j in range(len(series[i]))]) for i in range(len(series))]), 
        np.max([np.max([np.max(np.abs(field_peaks_plane[i][j])) for j in range(len(series[i]))]) for i in range(len(series))])] 
lims = max([abs(l) for l in lims])
lims = [0, lims]
        
l_meep = []
l_cmos = []
l_kuwa = []
axes_field = []
for i in range(len(series)):
    axes_field.append([])
    l_meep.append([])
    l_cmos.append([])
    l_kuwa.append([])
    for j in range(len(series[i])):
        axes[i][j].axvline(r[i][j] * from_um_factor[i][j] * 1e3, 
                           color="k", linestyle="dotted")
        axes[i][j].axvline(-r[i][j] * from_um_factor[i][j] * 1e3, 
                           color="k", linestyle="dotted")
        l_mp, = axes[i][j].plot(z_plane_cropped[i][j] * from_um_factor[i][j] * 1e3, 
                                np.abs(field_peaks_zprofile[i][j]),
                                label=series_label[i](series[i][j]),
                                color="k", linewidth=1.4) # color=colors[i][j]
        l_cm, = axes[i][j].plot(rvec[i][j][:,-1] * from_um_factor[i][j] * 1e3, 
                                np.abs(zprofile_cm_theory[i][j]),
                                linestyle=(0, (5, 1)), color="darkorchid") # color=colors[i][j]
        l_ku, = axes[i][j].plot(rvec[i][j][:,-1]  * from_um_factor[i][j] * 1e3, 
                                np.abs(zprofile_ku_theory[i][j]),
                                linestyle=(0, (5, 3)), color="deeppink") # color=colors[i][j]
        l_meep[-1].append(l_mp)
        l_cmos[-1].append(l_cm)
        l_kuwa[-1].append(l_ku)
        
        xlims = axes[i][j].get_xlim()
        axes[i][j].axhline(0, color="k", linewidth=.5)
        axes[i][j].axvline(0, color="k", linewidth=.5)
        axes[i][j].set_xlim(xlims)
        axes[i][j].set_title(series_label[i](series[i][j]))
        if i==len(series)-1:
            axes[i][j].set_xlabel(trs.choose("Position $Z$ [nm]", "Posición $Z$ [nm]"))
        if j==0:
            axes[i][j].set_ylabel(trs.choose(r"Electric Field $|E_z|(y=z=0)$",
                                             r"Campo eléctrico $|E_z|(y=z=0)$"))
        # else:
        #     axes[i][j].set_ytick_labels([])
        
        axes[i][j].xaxis.set_minor_locator(AutoMinorLocator())
        axes[i][j].yaxis.set_minor_locator(AutoMinorLocator())
        axes[i][j].grid(True, axis="y", which="both", alpha=.3)
        if i==0:
            axes[i][j].set_title(series_label[i](series[i][j]), y = 1.02)
        else:
            axes[i][j].set_title(series_label[i](series[i][j]), y = -.25)
        
        ax_field = vp.add_subplot_axes(axes[i][j], field_area)
        axes_field[-1].append(ax_field)
        
        ims = ax_field.imshow(np.abs(field_peaks_plane[i][j]).T, #np.power(np.abs(field_peaks_plane[i][j]),2).T
                              cmap='Reds', #interpolation='spline36', 
                              vmin=lims[0], vmax=lims[1],
                              extent=[min(y_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3,
                                      max(y_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3,
                                      min(z_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3,
                                      max(z_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3])
        ax_field.set_xticks([])
        ax_field.set_yticks([])
        
        if i==len(series)-1 and j==len(series[i])-1:
            cax = axes[i][j].inset_axes([1.04, 0, 0.07, n], #[1.04, 0, 0.07, 1], 
                                        transform=axes[i][j].transAxes)
            cbar = fig.colorbar(ims, ax=axes[i][j], cax=cax)
            cbar.set_label(trs.choose(r"Squared Electric Field $|E_z|^2(y=z=0)$",
                                      r"Cuadrado del campo eléctrico $|E_z|^2(y=z=0)$"))
            cbar.ax.minorticks_on()
        
leg = plt.legend(
    [l_meep[-1][-1], l_cmos[-1][-1], l_kuwa[-1][-1]],
    trs.choose(["MEEP Data", "CM Theory", "Ku Theory"], ["Data MEEP", "Teoría CM", "Teoría Ku"]),
    bbox_to_anchor=legend_area, 
    loc="center right", frameon=False)

fig.set_size_inches([13.5,  7.9]) # [17.5,  7.9]
plt.savefig(plot_file("AllSubplotsAbs.png"))

#%% MAKE PROFILE GIF

maxnframes = 300

# What should be parameters
nframes = min(maxnframes, np.max([[zprofile_results[i][j].shape[-1] for j in range(len(series[i]))] for i in range(len(series))]))
nframes = int( vu.round_to_multiple(nframes, np.max(time_period_factor) ) )
nframes_period = int(nframes/np.max(time_period_factor))

iref = -1
jref = 0
label_function = lambda k : trs.choose('Time: {:.1f} of period',
                                       'Tiempo: {:.1f} del período').format(
                                           t_line[iref][jref][k]/period_results[iref][jref])

# Animation base
fig = plt.figure(figsize=(m*subfig_size, n*subfig_size))
axes = fig.subplots(ncols=m, nrows=n)
lims = [np.min([np.min([np.min(zprofile_results[i][j]) for j in range(len(series[i]))]) for i in range(len(series))]), 
        np.max([np.max([np.max(zprofile_results[i][j]) for j in range(len(series[i]))]) for i in range(len(series))])] 

def draw_pml_box(i, j):
    axes[i][j].axvline((-cell_width[i][j]/2 + pml_width[i][j]) * from_um_factor[i][j] * 1e3, 
                        linestyle=":", color='k')
    axes[i][j].axvline((cell_width[i][j]/2 - pml_width[i][j]) * from_um_factor[i][j] * 1e3, 
                       linestyle=":", color='k')
    axes[i][j].axhline(0, color='k', linewidth=1)

def get_cut_points(i, j):
    cut_points = []
    for k in range(int(time_period_factor[i][j])):
        cut_points.append( np.argmin(np.abs(t_line[i][j] / period_results[i][j] - (k + 1))) )
    return [0, *cut_points]
cut_points_index = [[get_cut_points(i, j) for j in range(len(series[i]))] for i in range(len(series))]

nframes = int( vu.round_to_multiple(nframes, np.max(time_period_factor) ) )
nframes_period = int(nframes/np.max(time_period_factor))

def set_frame_numbers(i, j):
    frames_index = []
    for k in range(len(cut_points_index[i][j])):
        if k < len(cut_points_index[i][j])-1:
            intermediate_frames = np.linspace(
                cut_points_index[i][j][k],
                cut_points_index[i][j][k+1],
                nframes_period+1)[:-1]
            intermediate_frames = [int(fr) for fr in intermediate_frames]
            frames_index = [*frames_index, *intermediate_frames]
    return frames_index
frames_index = [[set_frame_numbers(i, j) for j in range(len(series[i]))] for i in range(len(series))]

def make_pic_line(k):
    for i in range(len(series)):
        for j in range(len(series[i])):
            kij = frames_index[i][j][k]
            axes[i][j].clear()
            axes[i][j].plot(z_plane[i][j] * from_um_factor[i][j] * 1e3, 
                            zprofile_results[i][j][:, kij],
                            color=colors[i][j])
            axes[i][j].axhline(zprofile_max[i][j][kij] ,
                               linestyle="dashed", color=colors[i][j])
            axes[i][j].axhline(0, linewidth=1, color="k")
            axes[i][j].set_ylim(*lims)
            if i==len(series)-1 and j==0:
                axes[i][j].text(-.2, -.2, label_function(kij), 
                                transform=axes[i][j].transAxes)
            draw_pml_box(i, j)
            if i==len(series)-1:
                axes[i][j].set_xlabel(trs.choose("Position $Z$ [nm]", "Posición $Z$ [nm]"))
            if j==0:
                axes[i][j].set_ylabel(trs.choose(r"Electric Field Profile $E_z|_{z=0}$",
                                                 r"Campo eléctrico $E_z|_{z=0}$"))
            axes[i][j].set_xlim(min(z_plane[i][j]) * from_um_factor[i][j] * 1e3, 
                                max(z_plane[i][j]) * from_um_factor[i][j] * 1e3)
            plt.show()
    return axes

def make_gif_line(gif_filename):
    pics = []
    for k in range(nframes):
        axes = make_pic_line(k)
        plt.savefig('temp_pic.png') 
        pics.append(mim.imread('temp_pic.png')) 
        print(str(k+1)+'/'+str(nframes))
    mim.mimsave(gif_filename+'.gif', pics, fps=5)
    os.remove('temp_pic.png')
    print('Saved gif')

make_gif_line(plot_file("AxisZ"))
plt.close(fig)
# del fig, ax, lims, nframes_step, nframes, call_series, label_function

#%% MAKE YZ PLANE FIELD GIF

maxnframes = 300

# What should be parameters
nframes = min(maxnframes, np.max([[zprofile_results[i][j].shape[-1] for j in range(len(series[i]))] for i in range(len(series))]))
nframes = int( vu.round_to_multiple(nframes, np.max(time_period_factor) ) )
nframes_period = int(nframes/np.max(time_period_factor))

iref = -1
jref = 0
label_function = lambda k : trs.choose('Time: {:.1f} of period',
                                       'Tiempo: {:.1f} del período').format(
                                           t_line[iref][jref][k]/period_results[iref][jref])

# Animation base
fig = plt.figure(figsize=(m*subfig_size, n*subfig_size))
axes = fig.subplots(ncols=m, nrows=n)
plt.suptitle(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
             plot_title_ending)
lims = [np.min([np.min([np.min(results_plane[i][j]) for j in range(len(series[i]))]) for i in range(len(series))]),
        np.max([np.max([np.max(results_plane[i][j]) for j in range(len(series[i]))]) for i in range(len(series))])]
lims = max([abs(l) for l in lims])
lims = [-lims, lims]

def draw_pml_box(i, j):
    axes[i][j].axvline((-cell_width[i][j]/2 + pml_width[i][j]) * from_um_factor[i][j] * 1e3, 
                        linestyle=":", color='k')
    axes[i][j].axvline((cell_width[i][j]/2 - pml_width[i][j]) * from_um_factor[i][j] * 1e3, 
                       linestyle=":", color='k')
    axes[i][j].axhline((-cell_width[i][j]/2 + pml_width[i][j]) * from_um_factor[i][j] * 1e3, 
                        linestyle=":", color='k')
    axes[i][j].axhline((cell_width[i][j]/2 - pml_width[i][j]) * from_um_factor[i][j] * 1e3, 
                       linestyle=":", color='k')

def get_cut_points(i, j):
    cut_points = []
    for k in range(int(time_period_factor[i][j])):
        cut_points.append( np.argmin(np.abs(t_line[i][j] / period_results[i][j] - (k + 1))) )
    return [0, *cut_points]
cut_points_index = [[get_cut_points(i, j) for j in range(len(series[i]))] for i in range(len(series))]

nframes = int( vu.round_to_multiple(nframes, np.max(time_period_factor) ) )
nframes_period = int(nframes/np.max(time_period_factor))

def set_frame_numbers(i, j):
    frames_index = []
    for k in range(len(cut_points_index[i][j])):
        if k < len(cut_points_index[i][j])-1:
            intermediate_frames = np.linspace(
                cut_points_index[i][j][k],
                cut_points_index[i][j][k+1],
                nframes_period+1)[:-1]
            intermediate_frames = [int(fr) for fr in intermediate_frames]
            frames_index = [*frames_index, *intermediate_frames]
    return frames_index
frames_index = [[set_frame_numbers(i, j) for j in range(len(series[i]))] for i in range(len(series))]

def make_pic_line(k):
    for i in range(len(series)):
        for j in range(len(series[i])):
            kij = frames_index[i][j][k]
            axes[i][j].clear()
            axes[i][j].imshow(results_plane[i][j][...,kij].T,
                              cmap='RdBu', #interpolation='spline36', 
                              vmin=lims[0], vmax=lims[1],
                              extent=[min(y_plane[i][j]) * from_um_factor[i][j] * 1e3,
                                      max(y_plane[i][j]) * from_um_factor[i][j] * 1e3,
                                      min(z_plane[i][j]) * from_um_factor[i][j] * 1e3,
                                      max(z_plane[i][j]) * from_um_factor[i][j] * 1e3])
            if i==len(series)-1:
                axes[i][j].set_xlabel(trs.choose("Position $Y$ [nm]", "Posición $Y$ [nm]"))
            if j==0:
                axes[i][j].set_ylabel(trs.choose("Position $Z$ [nm]", "Posición $Z$ [nm]"))
            if j==len(series[i])-1:
                cax = axes[i][j].inset_axes([1.04, 0, 0.07, 1], #[1.04, 0.2, 0.05, 0.6], 
                                            transform=axes[i][j].transAxes)
                cbar = fig.colorbar(ims, ax=axes[i][j], cax=cax)
                cbar.set_label(trs.choose("Electric field $E_z$",
                                          "Campo eléctrico $E_z$"))
            if i==len(series)-1 and j==0:
                axes[i][j].text(-.2, -.2, label_function(kij), 
                                transform=axes[i][j].transAxes)
            draw_pml_box(i, j)
            plt.show()
    return axes

def make_gif_line(gif_filename):
    pics = []
    for k in range(nframes):
        axes = make_pic_line(k)
        plt.savefig('temp_pic.png') 
        pics.append(mim.imread('temp_pic.png')) 
        print(str(k+1)+'/'+str(nframes))
    mim.mimsave(gif_filename+'.gif', pics, fps=5)
    os.remove('temp_pic.png')
    print('Saved gif')

make_gif_line(plot_file("PlanesYZ"))
plt.close(fig)
# del fig, ax, lims, nframes_step, nframes, call_series, label_function

#%% MAKE CROPPED PROFILE GIF

zprofile_cropped = [[vma.crop_field_zprofile(zprofile_results[i][j], 
                                             z_plane_index[i][j], 
                                             cell_width[i][j], pml_width[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

maxnframes = 300

# What should be parameters
nframes = min(maxnframes, np.max([[zprofile_cropped[i][j].shape[-1] for j in range(len(series[i]))] for i in range(len(series))]))
nframes = int( vu.round_to_multiple(nframes, np.max(time_period_factor) ) )
nframes_period = int(nframes/np.max(time_period_factor))

iref = -1
jref = 0
label_function = lambda k : trs.choose('Time: {:.1f} of period',
                                       'Tiempo: {:.1f} del período').format(
                                           t_line[iref][jref][k]/period_results[iref][jref])

# Animation base
n = len(series)
m = max([len(s) for s in series])

if n*m <= 3:
    subfig_size = 6.5
if n*m <= 6:
    subfig_size = 4.5
else:
    subfig_size = 3.5

fig = plt.figure(figsize=(m*subfig_size, n*subfig_size))
axes = fig.subplots(ncols=m, nrows=n)
lims = [np.min([np.min([np.min(zprofile_cropped[i][j]) for j in range(len(series[i]))]) for i in range(len(series))]), 
        np.max([np.max([np.max(zprofile_cropped[i][j]) for j in range(len(series[i]))]) for i in range(len(series))])] 

def get_cut_points(i, j):
    cut_points = []
    for k in range(int(time_period_factor[i][j])):
        cut_points.append( np.argmin(np.abs(t_line[i][j] / period_results[i][j] - (k + 1))) )
    return [0, *cut_points]
cut_points_index = [[get_cut_points(i, j) for j in range(len(series[i]))] for i in range(len(series))]

nframes = int( vu.round_to_multiple(nframes, np.max(time_period_factor) ) )
nframes_period = int(nframes/np.max(time_period_factor))

def set_frame_numbers(i, j):
    frames_index = []
    for k in range(len(cut_points_index[i][j])):
        if k < len(cut_points_index[i][j])-1:
            intermediate_frames = np.linspace(
                cut_points_index[i][j][k],
                cut_points_index[i][j][k+1],
                nframes_period+1)[:-1]
            intermediate_frames = [int(fr) for fr in intermediate_frames]
            frames_index = [*frames_index, *intermediate_frames]
    return frames_index
frames_index = [[set_frame_numbers(i, j) for j in range(len(series[i]))] for i in range(len(series))]

def make_pic_line(k):
    for i in range(len(series)):
        for j in range(len(series[i])):
            kij = frames_index[i][j][k]
            axes[i][j].clear()
            axes[i][j].plot(z_plane_cropped[i][j] * from_um_factor[i][j] * 1e3, 
                            zprofile_cropped[i][j][:, kij],
                            color=colors[i][j])
            axes[i][j].axvline(r[i][j] * from_um_factor[i][j] * 1e3, 
                               color="k", linestyle="dotted")
            axes[i][j].axvline(-r[i][j] * from_um_factor[i][j] * 1e3, 
                               color="k", linestyle="dotted")
            axes[i][j].axhline(zprofile_max[i][j][kij] ,
                               linestyle="dashed", color=colors[i][j])
            axes[i][j].axhline(0, color="k", linewidth=1)
            axes[i][j].set_ylim(*lims)
            if i==len(series)-1 and j==0:
                axes[i][j].text(-.2, -.2, label_function(kij), 
                                transform=axes[i][j].transAxes)
            if i==len(series)-1:
                axes[i][j].set_xlabel(trs.choose("Position $Z$ [nm]", "Posición $Z$ [nm]"))
            if j==0:
                axes[i][j].set_ylabel(trs.choose(r"Electric Field Profile $E_z|_{z=0}$",
                                                 r"Campo eléctrico $E_z|_{z=0}$"))
            axes[i][j].set_xlim(min(z_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3, 
                                max(z_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3)
            plt.show()
    return axes

def make_gif_line(gif_filename):
    pics = []
    for k in range(nframes):
        axes = make_pic_line(k)
        plt.savefig('temp_pic.png') 
        pics.append(mim.imread('temp_pic.png')) 
        print(str(k+1)+'/'+str(nframes))
    mim.mimsave(gif_filename+'.gif', pics, fps=5)
    os.remove('temp_pic.png')
    print('Saved gif')
#%%

make_gif_line(plot_file("CroppedAxisZ"))
plt.close(fig)
# del fig, ax, lims, nframes_step, nframes, call_series, label_function

#%% MAKE YZ PLANE FIELD GIF

plane_cropped = [[vma.crop_field_yzplane(results_plane[i][j], 
                                         y_plane_index[i][j], z_plane_index[i][j], 
                                         cell_width[i][j], pml_width[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

maxnframes = 300

# What should be parameters
nframes = min(maxnframes, np.max([[plane_cropped[i][j].shape[-1] for j in range(len(series[i]))] for i in range(len(series))]))
nframes = int( vu.round_to_multiple(nframes, np.max(time_period_factor) ) )
nframes_period = int(nframes/np.max(time_period_factor))

iref = -1
jref = 0
label_function = lambda k : trs.choose('Time: {:.1f} of period',
                                       'Tiempo: {:.1f} del período').format(
                                           t_line[iref][jref][k]/period_results[iref][jref])

# Animation base
fig = plt.figure(figsize=(m*subfig_size, n*subfig_size))
axes = fig.subplots(ncols=m, nrows=n)
plt.suptitle(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
             plot_title_ending)
lims = [np.min([np.min([np.min(plane_cropped[i][j]) for j in range(len(series[i]))]) for i in range(len(series))]),
        np.max([np.max([np.max(plane_cropped[i][j]) for j in range(len(series[i]))]) for i in range(len(series))])]
lims = max([abs(l) for l in lims])
lims = [-lims, lims]

def get_cut_points(i, j):
    cut_points = []
    for k in range(int(time_period_factor[i][j])):
        cut_points.append( np.argmin(np.abs(t_line[i][j] / period_results[i][j] - (k + 1))) )
    return [0, *cut_points]
cut_points_index = [[get_cut_points(i, j) for j in range(len(series[i]))] for i in range(len(series))]

nframes = int( vu.round_to_multiple(nframes, np.max(time_period_factor) ) )
nframes_period = int(nframes/np.max(time_period_factor))

def set_frame_numbers(i, j):
    frames_index = []
    for k in range(len(cut_points_index[i][j])):
        if k < len(cut_points_index[i][j])-1:
            intermediate_frames = np.linspace(
                cut_points_index[i][j][k],
                cut_points_index[i][j][k+1],
                nframes_period+1)[:-1]
            intermediate_frames = [int(fr) for fr in intermediate_frames]
            frames_index = [*frames_index, *intermediate_frames]
    return frames_index
frames_index = [[set_frame_numbers(i, j) for j in range(len(series[i]))] for i in range(len(series))]

def make_pic_line(k):
    for i in range(len(series)):
        for j in range(len(series[i])):
            kij = frames_index[i][j][k]
            axes[i][j].clear()
            axes[i][j].imshow(plane_cropped[i][j][...,kij].T,
                              cmap='RdBu', #interpolation='spline36', 
                              vmin=lims[0], vmax=lims[1],
                              extent=[min(y_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3,
                                      max(y_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3,
                                      min(z_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3,
                                      max(z_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3])
            if i==len(series)-1:
                axes[i][j].set_xlabel(trs.choose("Position $Y$ [nm]", "Posición $Y$ [nm]"))
            if j==0:
                axes[i][j].set_ylabel(trs.choose("Position $Z$ [nm]", "Posición $Z$ [nm]"))
            if j==len(series[i])-1:
                cax = axes[i][j].inset_axes([1.04, 0, 0.07, 1], #[1.04, 0.2, 0.05, 0.6], 
                                            transform=axes[i][j].transAxes)
                cbar = fig.colorbar(ims, ax=axes[i][j], cax=cax)
                cbar.set_label(trs.choose("Electric field $E_z$",
                                          "Campo eléctrico $E_z$"))
            if i==len(series)-1 and j==0:
                axes[i][j].text(-.2, -.2, label_function(kij), 
                                transform=axes[i][j].transAxes)
            plt.show()
    return axes

def make_gif_line(gif_filename):
    pics = []
    for k in range(nframes):
        axes = make_pic_line(k)
        plt.savefig('temp_pic.png') 
        pics.append(mim.imread('temp_pic.png')) 
        print(str(k+1)+'/'+str(nframes))
    mim.mimsave(gif_filename+'.gif', pics, fps=5)
    os.remove('temp_pic.png')
    print('Saved gif')
   
make_gif_line(plot_file("CroppedPlanesYZ"))
plt.close(fig)
# del fig, ax, lims, nframes_step, nframes, call_series, label_function

#%% MAKE PROFILE AND PLANE GIF (SUBPLOTS)

zprofile_cropped = [[vma.crop_single_field_zprofile(zprofile_results[i][j], z_plane_index[i][j], 
                                             cell_width[i][j], pml_width[i][j])
                     for j in range(len(series[i]))] for i in range(len(series))]
plane_cropped = [[vma.crop_single_field_yzplane(results_plane[i][j], 
                                         y_plane_index[i][j], z_plane_index[i][j], 
                                         cell_width[i][j], pml_width[i][j])
                     for j in range(len(series[i]))] for i in range(len(series))]


maxnframes = 300

# What should be parameters
nframes = min(maxnframes, np.max([[zprofile_cropped[i][j].shape[-1] for j in range(len(series[i]))] for i in range(len(series))]))
nframes = int( vu.round_to_multiple(nframes, np.max(time_period_factor) ) )
nframes_period = int(nframes/np.max(time_period_factor))

iref = -1
jref = 0
label_function = lambda k : trs.choose('     Time:\n{:.1f} of period',
                                       '     Tiempo:\n{:.1f} del período').format(
                                           t_line[iref][jref][k]/period_results[iref][jref])

# Animation base
fig = plt.figure(figsize=(m*subfig_size, n*subfig_size))
fig.set_size_inches([13.5,  7.9])
axes = fig.subplots(ncols=m, nrows=n, sharex=True, sharey=True, 
                    gridspec_kw={"wspace":0, "hspace":0})
axes_field = [[vp.add_subplot_axes(axes[i][j], field_area)
               for j in range(len(series[i]))] for i in range(len(series))]
colors = [sc(np.linspace(0,1,len(s)+2))[2:] 
          for sc, s in zip(series_colormaps, series)]

plt.suptitle(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
             plot_title_ending)
lims = [np.min([np.min([np.min(plane_cropped[i][j]) for j in range(len(series[i]))]) for i in range(len(series))]), 
        np.max([np.max([np.max(plane_cropped[i][j]) for j in range(len(series[i]))]) for i in range(len(series))])] 
lims = max([abs(l) for l in lims])
lims = [-lims, lims]

def get_cut_points(i, j):
    cut_points = []
    for k in range(int(time_period_factor[i][j])):
        cut_points.append( np.argmin(np.abs(t_line[i][j] / period_results[i][j] - (k + 1))) )
    return [0, *cut_points]
cut_points_index = [[get_cut_points(i, j) for j in range(len(series[i]))] for i in range(len(series))]

nframes = int( vu.round_to_multiple(nframes, np.max(time_period_factor) ) )
nframes_period = int(nframes/np.max(time_period_factor))

def set_frame_numbers(i, j):
    frames_index = []
    for k in range(len(cut_points_index[i][j])):
        if k < len(cut_points_index[i][j])-1:
            intermediate_frames = np.linspace(
                cut_points_index[i][j][k],
                cut_points_index[i][j][k+1],
                nframes_period+1)[:-1]
            intermediate_frames = [int(fr) for fr in intermediate_frames]
            frames_index = [*frames_index, *intermediate_frames]
    return frames_index
frames_index = [[set_frame_numbers(i, j) for j in range(len(series[i]))] for i in range(len(series))]

def make_pic_line(k):       
    for i in range(len(series)):
        for j in range(len(series[i])):
            kij = frames_index[i][j][k]
            
            axes[i][j].clear()
            axes_field[i][j].clear()
            
            axes[i][j].axvline(r[i][j] * from_um_factor[i][j] * 1e3, 
                               color="k", linestyle="dotted")
            axes[i][j].axvline(-r[i][j] * from_um_factor[i][j] * 1e3, 
                               color="k", linestyle="dotted")
            l_mp, = axes[i][j].plot(z_plane_cropped[i][j] * from_um_factor[i][j] * 1e3, 
                                    zprofile_cropped[i][j][...,kij],
                                    label=series_label[i](series[i][j]),
                                    color="k", linewidth=1.4) # color=colors[i][j]
            l_cm, = axes[i][j].plot(rvec[i][j][:,-1] * from_um_factor[i][j] * 1e3, 
                                    np.abs(zprofile_cm_theory[i][j]),
                                    linestyle=(0, (5, 1)), color="darkorchid") # color=colors[i][j]
            l_ku, = axes[i][j].plot(rvec[i][j][:,-1]  * from_um_factor[i][j] * 1e3, 
                                    np.abs(zprofile_ku_theory[i][j]),
                                    linestyle=(0, (5, 3)), color="deeppink") # color=colors[i][j]
            axes[i][j].plot(rvec[i][j][:,-1] * from_um_factor[i][j] * 1e3, 
                            -np.abs(zprofile_cm_theory[i][j]),
                            linestyle=(0, (5, 1)), color="darkorchid") # color=colors[i][j]
            axes[i][j].plot(rvec[i][j][:,-1]  * from_um_factor[i][j] * 1e3, 
                            -np.abs(zprofile_ku_theory[i][j]),
                            linestyle=(0, (5, 3)), color="deeppink") # color=colors[i][j]
            
            axes[i][j].axhline(0, color="k", linewidth=.5)
            axes[i][j].axvline(0, color="k", linewidth=.5)
            axes[i][j].set_title(series_label[i](series[i][j]))
            axes[i][j].set_ylim(lims)
            if i==len(series)-1 and j==0:
                axes[i][j].text(-.15, -.2, label_function(kij), 
                                transform=axes[i][j].transAxes)
            if i==len(series)-1:
                axes[i][j].set_xlabel(trs.choose("Position $Z$ [nm]", "Posición $Z$ [nm]"))
            if j==0:
                axes[i][j].set_ylabel(trs.choose(r"Electric Field $E_z(y=z=0)$",
                                                  r"Campo eléctrico $E_z(y=z=0)$"))
            # else:
            #     axes[i][j].set_ytick_labels([])
            
            axes[i][j].xaxis.set_minor_locator(AutoMinorLocator())
            axes[i][j].yaxis.set_minor_locator(AutoMinorLocator())
            axes[i][j].grid(True, axis="y", which="both", alpha=.3)
            if i==0:
                axes[i][j].set_title(series_label[i](series[i][j]), y = 1.02)
            else:
                axes[i][j].set_title(series_label[i](series[i][j]), y = -.25)
                        
            ims = axes_field[i][j].imshow(plane_cropped[i][j][...,kij].T,
                                          cmap='RdBu', #interpolation='spline36', 
                                          vmin=lims[0], vmax=lims[1],
                                          extent=[min(y_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3,
                                                  max(y_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3,
                                                  min(z_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3,
                                                  max(z_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3])
            axes_field[i][j].set_xticks([])
            axes_field[i][j].set_yticks([])
            
            if i==len(series)-1 and j==len(series[i])-1:
                cax = axes[i][j].inset_axes([1.04, 0, 0.07, 2], #[1.04, 0, 0.07, 1], 
                                            transform=axes[i][j].transAxes)
                cbar = fig.colorbar(ims, ax=axes[i][j], cax=cax)
                cbar.set_label(trs.choose("Electric field $E_z$",
                                          "Campo eléctrico $E_z$"))
                cbar.ax.minorticks_on()
            
    leg = plt.legend(
        [l_mp, l_cm, l_ku],
        trs.choose(["MEEP Data", "CM Theory", "Ku Theory"], ["Data MEEP", "Teoría CM", "Teoría Ku"]),
        bbox_to_anchor=legend_area, #(2.5, 1.4), 
        loc="center right", frameon=False)
    
    return axes

def make_gif_line(gif_filename):
    pics = []
    for k in range(nframes):
        axes = make_pic_line(k)
        plt.savefig('temp_pic.png') 
        pics.append(mim.imread('temp_pic.png')) 
        print(str(k+1)+'/'+str(nframes))
    mim.mimsave(gif_filename+'.gif', pics, fps=5)
    os.remove('temp_pic.png')
    print('Saved gif')  

#%%

make_gif_line(plot_file("All2"))
plt.close(fig)