#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 17:36:16 2021

@author: vall
"""

# Planewave pulse field and flux

from socket import gethostname
if "Nano" in gethostname():
    syshome = "/home/nanofisica/Documents/Vale/ThesisPython"
elif "vall" in gethostname():
    syshome = "/home/vall/Documents/Thesis/ThesisPython"
else:
    raise ValueError("Your PC must be registered at the top of this code")

import sys
sys.path.append(syshome)

import h5py as h5
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.pylab as plab
import matplotlib.gridspec as gridspec
from matplotlib import use as use_backend
import os
import v_analysis as va
import v_meep_analysis as vma
import v_plot as vp
import v_save as vs
import v_utilities as vu

english = False
trs = vu.BilingualManager(english=english)
vp.set_style()

#%% PARAMETERS <<

# Saving directories
folder = ["Field/Sources/PulsePlanewave/6)TestPulse/085115/Res10",
          "Field/Sources/PulsePlanewave/6)TestPulse/085115/Res10"]
home = vs.get_home()

# Parameter for the test
test_param_string = "pml_wlen_factor"
test_param_calculation = False
test_param_in_params = False
test_param_in_series = True
test_param_position = 0
test_param_ij_expression = "pml_width[i][j]" # Leave "" by default
test_param_name = trs.choose("PML Width", "Espesor PML")
test_param_units = r"$\lambda_{0}$" # Leave "" by default

# Sorting and labelling data series
sorting_function = [lambda l : vu.sort_by_number(l, test_param_position)]*2
series_label = [lambda s : f"PML {vu.find_numbers(s)[test_param_position]:.2f} "+r"$\lambda_{0}$"]*2
series_must = ["Vac", "Wat"] # leave "" per default
series_mustnt = ["Res"]*2 # leave "" per default

# Plot options
plot_title_ending = trs.choose("Planewave Pulse", "pulso de frente de ondas plano")
series_legend = trs.choose(["Vacuum", "Water"], ["Vacío", "Agua"])
series_colormaps = [plab.cm.Reds, plab.cm.Blues] # plab.cm.summer.reversed()
series_colors = ["r", "b"] # "k"
series_markers = ["o"]*2
series_markersizes = [8]*2
series_linestyles = ["solid"]*2
theory_linestyles = ["dashed"]*2
plot_for_display = False
plot_folder = "DataAnalysis/Field/Sources/PulsePlanewave/PMLWidth/085115"

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

# Get the corresponding field data
files_line = [[h5.File(file[i](series[i][j], "Field-Lines.h5"), "r") 
               for j in range(len(series[i]))] for i in range(len(series))]
results_line = [[files_line[i][j]["Ez"] 
                 for j in range(len(series[i]))] for i in range(len(series))]
t_line = [[np.asarray(files_line[i][j]["T"])
           for j in range(len(series[i]))] for i in range(len(series))]
x_line = [[np.asarray(files_line[i][j]["X"])
           for j in range(len(series[i]))] for i in range(len(series))]

# Get the corresponding flux data
data = [[np.loadtxt(file[i](series[i][j], "Results.txt")) for j in range(len(series[i]))] for i in range(len(series))]
flux_wlens = [[data[i][j][:,0] for j in range(len(series[i]))] for i in range(len(series))]
flux_intensity = [[data[i][j][:,1:] for j in range(len(series[i]))] for i in range(len(series))]

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
        f = h5.File(file[i](series[i][j], "Resources.h5"))
        params[i][j]["used_ram"] = np.array(f["RAM"])
        try:
            params[i][j]["used_swap"] = np.array(f["SWAP"])
        except:
            print(f"No SWAP data found for {series[i][j]}")
        params[i][j]["elapsed_time"] = np.array(f["ElapsedTime"])
del i, j

# Extract some other parameters from the parameters dict
from_um_factor = [[params[i][j]["from_um_factor"] for j in range(len(series[i]))] for i in range(len(series))]
resolution = [[params[i][j]["resolution"] for j in range(len(series[i]))] for i in range(len(series))]
courant = [[params[i][j]["courant"] for j in range(len(series[i]))] for i in range(len(series))]
index = [[params[i][j]["submerged_index"] for j in range(len(series[i]))] for i in range(len(series))]
empty_width = [[params[i][j]["empty_width"] for j in range(len(series[i]))] for i in range(len(series))]
cell_width = [[params[i][j]["cell_width"] for j in range(len(series[i]))] for i in range(len(series))]
wlen_range = [[params[i][j]["wlen_range"] for j in range(len(series[i]))] for i in range(len(series))]
source_center = [[params[i][j]["source_center"] for j in range(len(series[i]))] for i in range(len(series))]
pml_width = [[params[i][j]["pml_width"] for j in range(len(series[i]))] for i in range(len(series))]
flux_wall_positions = [[params[i][j]["flux_wall_positions"] for j in range(len(series[i]))] for i in range(len(series))]
n_flux_walls = [[params[i][j]["n_flux_walls"] for j in range(len(series[i]))] for i in range(len(series))]
period_line = [[params[i][j]["period_line"] for j in range(len(series[i]))] for i in range(len(series))]
until_after_sources = [[params[i][j]["until_after_sources"] for j in range(len(series[i]))] for i in range(len(series))]
time_factor_cell = [[params[i][j]["time_factor_cell"] for j in range(len(series[i]))] for i in range(len(series))]
wlen_in_vacuum = [[params[i][j]["wlen_in_vacuum"] for j in range(len(series[i]))] for i in range(len(series))]
units = [[params[i][j]["units"] for j in range(len(series[i]))] for i in range(len(series))]
sysname = [[params[i][j]["sysname"] for j in range(len(series[i]))] for i in range(len(series))]

#%% EXTRACT SOME MORE PARAMETERS <<

# Guess some more parameters, calculating them from others
minor_division_space_nm = [[from_um_factor[i][j] * 1e3 / resolution[i][j] for j in range(len(series[i]))] for i in range(len(series))]
minor_division_time_as = [[courant[i][j] * from_um_factor[i][j] * 1e12 / ( resolution[i][j] * 299792458 ) for j in range(len(series[i]))] for i in range(len(series))]

width_points = [[int(cell_width[i][j] * resolution[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
inner_width_points = [[int( (cell_width[i][j] - 2*pml_width[i][j]) * resolution[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
grid_points = [[width_points[i][j]**3 for j in range(len(series[i]))] for i in range(len(series))]
inner_grid_points = [[inner_width_points[i][j]**3 for j in range(len(series[i]))] for i in range(len(series))]
memory_B = [[2 * 12 * grid_points[i][j] * 32 for j in range(len(series[i]))] for i in range(len(series))]

freq_range = [[1/np.array(wlen_range[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
freq_center = [[np.mean(freq_range[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
freq_width = [[abs(np.diff(freq_range[i][j])[0]) for j in range(len(series[i]))] for i in range(len(series))]
use_units = True in [True in units[i] for i in range(len(series))]

# Extract runtime and RAM data
elapsed_time = [[p["elapsed_time"] for p in par] for par in params]
total_elapsed_time = [[sum(p["elapsed_time"]) for p in par] for par in params]

used_ram = [[np.array(p["used_ram"])/(1024)**2 for p in par] for par in params]
used_swap = [[p["used_swap"] for p in par] for par in params]
total_used_ram = []
for i in range(len(series)):
    total_used_ram.append([])
    for j in range(len(series[i])):
        try: total_used_ram[i].append( np.sum(used_ram[i][j], axis=1) )
        except: total_used_ram[i].append( used_ram[i][j] )

# Calculate test parameter if needed
if test_param_calculation:
    test_param = []
    for i in range(len(series)):
        test_param.append([])
        for j in range(len(series[i])):
            test_param[i].append( eval(test_param_ij_expression) )

#%% POSITION RECONSTRUCTION <<

t_line_index = [[vma.def_index_function(t_line[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
x_line_index = [[vma.def_index_function(x_line[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

x_line_cropped = [[x_line[i][j][:x_line_index[i][j](cell_width[i][j]/2 - pml_width[i][j])+1] for j in range(len(series[i]))] for i in range(len(series))]
x_line_cropped = [[x_line_cropped[i][j][x_line_index[i][j](-cell_width[i][j]/2 + pml_width[i][j]):] for j in range(len(series[i]))] for i in range(len(series))]

#%% DATA EXTRACTION <<

source_results = [[vma.get_source_from_line(results_line[i][j], x_line_index[i][j], source_center[i][j])
                   for j in range(len(series[i]))] for i in range(len(series))]

walls_results =  [[ [results_line[i][j][x_line_index[i][j](fx),:] for fx in flux_wall_positions[i][j]]
                   for j in range(len(series[i]))] for i in range(len(series))]
    
results_cropped_line = [[vma.crop_field_xprofile(results_line[i][j], x_line_index[i][j], cell_width[i][j], pml_width[i][j])
                         for j in range(len(series[i]))] for i in range(len(series))]

#%% GENERAL PLOT CONFIGURATION <<

n = len(series)
m = max([len(s) for s in series])

if test_param_units != "":
    test_param_label = test_param_name + f" [{test_param_units}]"
else: test_param_label = test_param_name

colors = [sc(np.linspace(0,1,len(s)+2))[2:] 
          for sc, s in zip(series_colormaps, series)]

if not os.path.isdir(os.path.join(home, plot_folder)):
    os.makedirs(os.path.join(home, plot_folder))
plot_file = lambda n : os.path.join(home, plot_folder, n)

#%% MAKE FOURIER ANALYSIS FOR SOURCE <<

fourier = [[np.abs(np.fft.rfft(source_results[i][j])) for j in range(len(series[i]))] for i in range(len(series))]
fourier_freq = [[np.fft.rfftfreq(len(source_results[i][j]), d=period_line[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
fourier_wlen = [[1 / fourier_freq[i][j] / index[i][j]  for j in range(len(series[i]))] for i in range(len(series))]
fourier_best = [[1 / (np.mean(1/np.array(wlen_range[i][j])) * index[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

fourier_max_wlen = [[fourier_wlen[i][j][ np.argmax(fourier[i][j]) ]  for j in range(len(series[i]))] for i in range(len(series))]
fourier_max_best = [[fourier_wlen[i][j][ np.argmin(np.abs(fourier_wlen[i][j] - fourier_best[i][j])) ]  for j in range(len(series[i]))] for i in range(len(series))]

fourier_theory = [[norm.pdf(fourier_freq[i][j], freq_center[i][j], freq_width[i][j]/(2*np.pi)) for j in range(len(series[i]))] for i in range(len(series))]
fourier_msqdiff = [[np.mean(np.square(fourier[i][j]/np.sum(fourier[i][j]) - fourier_theory[i][j]/np.sum(fourier_theory[i][j]))) for j in range(len(series[i]))] for i in range(len(series))]

plot_fourier_freq = [[np.linspace(np.min(fourier_freq[i][j]), np.max(fourier_freq[i][j]), 500) for j in range(len(series[i]))] for i in range(len(series))]
plot_fourier_wlen = [[from_um_factor[i][j] * 1e3 / plot_fourier_freq[i][j] / index[i][j] for j in range(len(series[i]))] for i in range(len(series))]
plot_fourier_theory = [[norm.pdf(plot_fourier_freq[i][j], freq_center[i][j], freq_width[i][j]/(2*np.pi)) for j in range(len(series[i]))] for i in range(len(series))]

#%% SHOW SOURCE AND FOURIER USED FOR NORMALIZATION

fig = plt.figure()
plt.title(trs.choose("Source for ", "Fuente para ") + plot_title_ending)

series_lines = []
for i in range(len(series)):
    for j in range(len(series[i])):
        l, = plt.plot(t_line[i][j], 
                      source_results[i][j],
                      color=colors[i][j])
        l.set_label(series_label[i](series[i][j]))
        if j == int( 2 * len(series[i]) / 3 ):
            series_lines.append(l)
plt.xlabel(trs.choose("Time $T$ [MPu]", "Tiempo $T$ [uMP]"))
plt.ylabel(trs.choose(r"Electric Field $E_z(y=z=0)$",
                      r"Campo eléctrico $E_z(y=z=0)$"))

box = fig.axes[0].get_position()
box.x1 = box.x1 - .3 * (box.x1 - box.x0)
fig.axes[0].set_position(box)
leg = plt.legend(bbox_to_anchor=(1.6, .5), 
                 # ncol=len(series), columnspacing=-0.5, 
                 loc="center right", frameon=False)

plt.savefig(plot_file("Source.png"))
        
fig = plt.figure()
plt.title(trs.choose("Source Fourier for ", "Fourier de fuente para ") + plot_title_ending)

series_lines = []
for i in range(len(series)):
    for j in range(len(series[i])):
        l, = plt.plot(fourier_wlen[i][j], fourier[i][j], color=colors[i][j])
        # l2, = plt.plot(plot_fourier_wlen[i][j], 
        #                plot_fourier_theory[i][j] * np.max(fourier[i][j]) / np.max(plot_fourier_theory[i][j]),
        #                color=colors[i][j],
        #                linestyle=theory_linestyles[i])
        l.set_label(series_label[i](series[i][j]))
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
second_legend = plt.legend(bbox_to_anchor=(1.6, .5), 
                           # ncol=len(series), columnspacing=-0.5, 
                           loc="center right", frameon=False)
plt.gca().add_artist(first_legend)

plt.savefig(plot_file("SourceFFT.png"))

if use_units: plt.xlim([350, 850])
else: plt.xlim([0.25, 1.5])
# else: plt.xlim([0, 4])
        
plt.savefig(plot_file("SourceFFTZoom.png"))

#%% WAVELENGTH OPTIMIMUM VARIATION PLOT

plt.figure()
plt.title(trs.choose("Wavelength difference for ", 
                     "Diferencia en longitud de onda para ") + plot_title_ending)
for i in range(len(series)):
    plt.plot(test_param[i], 
             100 * ( np.array(fourier_max_wlen[i]) - np.array(fourier_max_best[i]) ) / np.array(fourier_max_best[i]), 
             color=series_colors[i], marker=series_markers[i], alpha=.6,
             markersize=series_markersizes[i], markeredgewidth=0, linestyle="solid")
plt.legend(series_legend)
plt.xlabel(test_param_label)
plt.ylabel(trs.choose("Maximum Wavelength Percentual Variation", 
                      "Variación porcentual de la longitud de onda máxima\n") + 
           r"$\lambda_{max} = \argmax [\mathcal{F}\;(E_z)]$ [%]")
plt.tight_layout()
plt.savefig(plot_file("LambdaVariation.png"))

#%% MEAN SQUARED DIFFERENCE PLOT ==

if plot_for_display: use_backend("Agg")

fig = plt.figure()
if plot_for_display: fig.dpi = 200
if not plot_for_display:
    plt.title(trs.choose("Spectrum source contrast for ", 
                         "Contraste de espectro de la fuente para ") + plot_title_ending)
for i in range(len(series)):
    plt.plot(test_param[i], 
             fourier_msqdiff[i], 
             color=series_colors[i], marker=series_markers[i], alpha=.6,
             markersize=series_markersizes[i], linestyle="solid", markeredgewidth=0)
plt.legend(series_legend)
plt.xlabel(test_param_label)
plt.ylabel(trs.choose("Electric Field Fourier\nMean Squared Difference\n",
                      "Diferencia cuadrática media \nen transformada del campo eléctrico\n") + 
           r"$MSD[\;\mathcal{F}^{MEEP}(\lambda) - \mathcal{F}^{GAUSS}(\lambda)\;]$")
fig.axes[0].set_yscale("log")

for ax in fig.axes:
    box = ax.get_position()
    width = box.x1 - box.x0
    box.x1 = box.x1 + .08 * width
    box.x0 = box.x0 + .08 * width
    box.y1 = box.y1 + .05*(box.y1 - box.y0)
    ax.set_position(box)
    
fig.set_size_inches([6 , 4.32])
plt.savefig(plot_file("FourierMSD.png"))

if plot_for_display: use_backend("Qt5Agg")

#%% ANALYSE FLUX WALLS <<

flux_max_intensity = [[ [np.max(flux_intensity[i][j][:,k]) for k in range(n_flux_walls[i][j])] for j in range(len(series[i]))] for i in range(len(series))]

flux_freqs = [[ [index[i][j] / flux_wlens[i][j][k] for k in range(n_flux_walls[i][j])] for j in range(len(series[i]))] for i in range(len(series))]

flux_theory = [[ [norm.pdf(flux_freqs[i][j][k], freq_center[i][j], freq_width[i][j]/(2*np.pi)) for k in range(n_flux_walls[i][j])] for j in range(len(series[i]))] for i in range(len(series))]
flux_msqdiff = [[ [np.mean(np.square(flux_intensity[i][j][k]/np.sum(flux_intensity[i][j]) - flux_theory[i][j][k]/np.sum(flux_theory[i][j][k]))) for k in range(n_flux_walls[i][j])] for j in range(len(series[i]))] for i in range(len(series))]

# plot_fourier_freq = [[np.linspace(np.min(fourier_freq[i][j]), np.max(fourier_freq[i][j]), 500) for j in range(len(series[i]))] for i in range(len(series))]
# plot_fourier_wlen = [[from_um_factor[i][j] * 1e3 / plot_fourier_freq[i][j] / index[i][j] for j in range(len(series[i]))] for i in range(len(series))]
# plot_fourier_theory = [[norm.pdf(plot_fourier_freq[i][j], freq_center[i][j], freq_width[i][j]/(2*np.pi)) for j in range(len(series[i]))] for i in range(len(series))]

#%% SHOW FLUX VARIATIONS
    
fig = plt.figure()
plt.title(trs.choose("Flux maximum intensity for ", 
                     "Intensidad máxima del flujo para ") + plot_title_ending)
for i in range(len(series)):
    for j in range(len(series[i])):
        plt.plot(np.array(flux_wall_positions[i][j]) * index[i][j] / np.mean(wlen_range[i][j]), 
                 flux_max_intensity[i][j] / np.mean(flux_max_intensity[i][j]), 
                 "o-", color=colors[i][j], alpha=0.6, 
                 label=series_label[i](series[i][j]))
        plt.axvline(-empty_width[i][j] * index[i][j], color=colors[i][j],
                    linestyle=(0, (5,10)), linewidth=.8)
        plt.axvline(empty_width[i][j] * index[i][j], color=colors[i][j],
                    linestyle=(0, (5,10)), linewidth=.8)
plt.axvline(linewidth=1, color="k")

plt.xlabel(trs.choose(r"Position $X$ [$\lambda_0/n$]", r"Posición  $X$ [$\lambda_0/n$]"))
plt.ylabel(trs.choose(r"Electromagnetic Flux Maximum $P_{max}(\lambda)$ [au]",
                      r"Máximo flujo electromagnético $P_{max}(\lambda)$ [ua]"))

box = fig.axes[0].get_position()
box.x1 = box.x1 - .15 * (box.x1 - box.x0)
fig.axes[0].set_position(box)
plt.legend(bbox_to_anchor=(1.3, .5), loc="center right", frameon=False)

fig.set_size_inches([9.91, 4.8 ])

plt.savefig(plot_file("FluxMaximum.png"))

#%% FLUX MEAN SQUARED DIFFERENCE PLOT

fig = plt.figure()
plt.title(trs.choose("Flux spectrum contrast for ", 
                     "Contraste en espectro de flujo para ") + plot_title_ending)
for i in range(len(series)):
    for j in range(len(series[i])):
        plt.plot(np.array(flux_wall_positions[i][j]) * index[i][j] / np.mean(wlen_range[i][j]), 
                 flux_msqdiff[i][j], "o-", color=colors[i][j], 
                 alpha=0.6, label=series_label[i](series[i][j]))
        plt.axvline(-empty_width[i][j] * index[i][j], color=colors[i][j],
                    linestyle=(0, (5,10)), linewidth=.8)
        plt.axvline(empty_width[i][j] * index[i][j], color=colors[i][j],
                    linestyle=(0, (5,10)), linewidth=.8)
plt.axvline(linewidth=1, color="k")

plt.xlabel(trs.choose(r"Position $X$ [$\lambda/n$]", r"Posición  $X$ [$\lambda/n$]"))
plt.ylabel(trs.choose(r"Electromagnetic Flux Maximum $P_{max}(\lambda)$ [au]",
                      r"Máximo flujo electromagnético $P_{max}(\lambda)$ [ua]"))
plt.ylabel(trs.choose("Electromagnetic Flux\nMean Squared Difference\n",
                      "Diferencia cuadrática media \nen flujo electromagnético\n") + 
           r"$MSD[\;P^{MEEP}(\lambda) - P^{MEEP}(\lambda)\;]$")

box = fig.axes[0].get_position()
box.x1 = box.x1 - .15 * (box.x1 - box.x0)
fig.axes[0].set_position(box)
plt.legend(bbox_to_anchor=(1.3, .5), loc="center right", frameon=False)

fig.set_size_inches([9.91, 4.8 ])

plt.savefig(plot_file("FluxMSD.png"))

#%% EXTRACT INTERFERENCE AMPLITUDE <<

flux_max_amplitude = [[vma.get_amplitude_from_source(flux_max_intensity[i][j], 
                                                     peaks_sep_sensitivity=None, # .5
                                                     amplitude_sensitivity=None,
                                                     last_stable_periods=None)[-1]
                       for j in range(len(series[i]))] for i in range(len(series))]

flux_max_relative_amplitude = [[flux_max_amplitude[i][j] / np.mean(flux_max_intensity[i][j])
                                for j in range(len(series[i]))] for i in range(len(series))]

#%% PLOT INTERFERENCE AMPLITUDE ==

alternate_axis = True

if plot_for_display: use_backend("Agg")

fig = plt.figure()
if plot_for_display: fig.dpi = 200
if not plot_for_display:
    plt.title(trs.choose("Flux maximum intensity for ", 
                         "Intensidad máxima del flujo para ") + plot_title_ending)

for i in range(len(series)):
    plt.plot(test_param[i], 
             100 * np.array(flux_max_relative_amplitude[i]), 
             color=series_colors[i], linestyle="solid",
             marker=series_markers[i], markersize=series_markersizes[i], 
             alpha=0.6, markeredgewidth=0, zorder=10)
if max(fig.axes[0].get_ylim()) >= 0 >= min(fig.axes[0].get_ylim()):
    plt.axhline(color="k", linewidth=.5, zorder=0)
plt.legend(fig.axes[0].lines, series_legend)
plt.xlabel(test_param_label)
if max([len(tp) for tp in test_param])<=4: 
    plt.xticks( test_param[ np.argmax([len(data) for data in test_param]) ] )
plt.ylabel(trs.choose("Oscillations Amplitude in Flux ", 
                      "Amplitud de las oscilaciones del flujo ") + 
           "$\Delta P_{max}$ [%]")
fig.axes[0].set_yscale("log")
if alternate_axis:
    fig.axes[0].yaxis.tick_right()
    fig.axes[0].yaxis.set_label_position("right")
fig.set_size_inches([6 , 4.32])

for ax in fig.axes:
    box = ax.get_position()
    box.x1 = box.x1 - .05 * (box.x1 - box.x0)
    ax.set_position(box)

vs.saveplot(plot_file("FluxMaximumVariation.png"), overwrite=True)

if plot_for_display: use_backend("Qt5Agg")

#%% COMPARE TEST PARAMETER VALUES: SOURCE AND FIELD ==

test_param_to_compare = [0.10, 0.25, 0.50]
which_series = 1
N = len(test_param_to_compare)
which_inside_series = [test_param[which_series].index(test_param_to_compare[k]) for k in range(N)]

if plot_for_display: use_backend("Agg")

fig, axes = plt.subplots(ncols=N, nrows=2, sharex=True,
                         gridspec_kw={"hspace":0, "wspace":0, "height_ratios":(3,2)})
axes = np.array(axes).T
if plot_for_display: fig.dpi = 200

for k in range(N):
    i = which_series
    j = which_inside_series[k]

    axes[k,0].set_title(series_legend[i]+":\n"+series_label[i](series[i][j]))

    T, X = np.meshgrid(t_line[i][j], 
                       x_line_cropped[i][j] * index[i][j] / np.mean(wlen_range[i][j]))
    pic = axes[k,0].contourf(T, X, results_cropped_line[i][j], 100, cmap='RdBu')
    axes[k,0].axhline(color="k", linewidth=.5)

    axes[k,1].plot(t_line[i][j], source_results[i][j] / np.max(np.abs(source_results[i][j])),
                   label=series_label[i](series[i][j]))
    axes[k,1].axhline(color="k", linewidth=.5)

axes[0,0].set_ylabel(trs.choose(r"Position $X$ [$\lambda_0/n$]", 
                                r"Posición  $X$ [$\lambda_0/n$]"))
axes[0,1].set_ylabel(trs.choose("Electric Field\n"+r"$E_z(x=x_0,\,y=z=0)$",
                                "Campo eléctrico\n"+r"$E_z(x=x_0,\,y=z=0)$"))

for k in range(N):
    axes[k,-1].set_xlabel(trs.choose("Time $T$ [MPu]", "Tiempo $T$ [uMP]"))
    axes[k,0].grid(False)
    axes[k,0].set_ylim(np.max([np.min(x_line_cropped[i][j] * index[i][j] / np.mean(wlen_range[i][j])) 
                               for j in which_inside_series]),
                       np.min([np.max(x_line_cropped[i][j] * index[i][j] / np.mean(wlen_range[i][j]))
                               for j in which_inside_series]))
    axes[k,1].set_ylim(axes[0,1].get_ylim())
    if k!=0:
        axes[k,0].yaxis.set_ticklabels( [] )
        axes[k,1].yaxis.set_ticklabels( [] )
        
cax = axes[-1,1].inset_axes([1.04, 0, 0.07, 2.5],
                            transform=axes[-1,1].transAxes)
cbar = fig.colorbar(pic, ax=axes[-1,0], cax=cax)
cbar.set_label(trs.choose("Electric Field\n"+r"$E_z(x,\,y=z=0)$",
                          "Campo eléctrico\n"+r"$E_z(x,\,y=z=0)$"))

for ax in axes.flatten():
    box = ax.get_position()
    width = box.x1 - box.x0
    box.x1 = box.x1 - .1 * width
    box.x0 = box.x0 - .1 * width
    ax.set_position(box)
           
fig.set_size_inches([10.02, 5.94])

plt.savefig(plot_file("ComparisonSourceField.png"))

if plot_for_display: use_backend("Qt5Agg")

#%% COMPARE TEST PARAMETER VALUES: FLUX WALLS ==

test_param_to_compare = [0.10, 0.25, 0.50]
which_series = 1
N = len(test_param_to_compare)
which_inside_series = [test_param[which_series].index(test_param_to_compare[k]) for k in range(N)]

if np.std(n_flux_walls)<10e-6:
    
    with_legend = False
    these_colors = plab.cm.Greens(np.linspace(0,1,n_flux_walls[0][0]))
    
    if plot_for_display: use_backend("Agg")
    
    fig, axes = plt.subplots(ncols=N, nrows=2, 
                             gridspec_kw={"hspace":0, "wspace":0, "height_ratios":(1,1)})
    axes = np.array(axes).T
    if plot_for_display: fig.dpi = 200
    
    inside_lines = []
    for k in range(N):
        i = which_series
        j = which_inside_series[k]
    
        axes[k,0].set_title(series_legend[i]+": "+series_label[i](series[i][j]))
    
        for fw in range(n_flux_walls[i][j]):
            l, = axes[k,0].plot(flux_wlens[i][j], 
                                flux_intensity[i][j][:,fw] / np.max(flux_intensity[i][j]), 
                                color=these_colors[fw], 
                                alpha=0.7, linewidth=2,
                                label=f"x = {flux_wall_positions[i][j][fw] * index[i][j] / np.mean(wlen_range[i][j]):.1f} $\lambda_0/n$")
            if j==which_inside_series[0]:
                inside_lines.append(l)
        axes[k,0].axhline(color="k", linewidth=.5)
    
        axes[k,1].plot(np.array(flux_wall_positions[i][j]) * index[i][j] / np.mean(wlen_range[i][j]), 
                       flux_max_intensity[i][j] / np.mean(flux_max_intensity[i][j]), "o-")
        axes[k,1].axvline(linewidth=1, color="k")
        
    
    axes[0,0].set_ylabel(trs.choose("Normalized\nElectromagnetic Flux "+r"$P(\lambda)$",
                                    "Flujo electromagnético\nnormalizado "+r"$P(\lambda)$"))
    axes[0,1].set_ylabel(trs.choose("Normalized\nElectromagnetic Flux Maximum "+r"$P_{max}(\lambda)$",
                                    "Máximo flujo electromagnético\nnormalizado "+r"$P_{max}(\lambda)$"))
    
    ylims = (np.min([axes[k,0].get_ylim() for k in range(N)]),
             np.max([axes[k,0].get_ylim() for k in range(N)]))
    for k in range(N):
        if use_units:
            axes[k,0].set_xlabel(trs.choose(r"Wavelength $\lambda$ [nm]", 
                                            r"Longitud de onda $\lambda$ [nm]"))
        else:
            axes[k,0].set_xlabel(trs.choose(r"Wavelength $\lambda/n$ [$\lambda_0$]", 
                                            r"Longitud de onda $\lambda/n$ [$\lambda_0$]"))
        axes[k,0].xaxis.tick_top()
        axes[k,0].xaxis.set_label_position("top")
        axes[k,0].set_ylim(ylims)
        axes[k,1].set_xlabel(trs.choose(r"Position $X$ [$\lambda_0/n$]", 
                                        r"Posición  $X$ [$\lambda_0/n$]"))
        axes[k,1].set_ylim(axes[0,1].get_ylim())
        if k!=0:
            axes[k,0].yaxis.set_ticklabels( [] )
            axes[k,1].yaxis.set_ticklabels( [] )
        
    if with_legend:
        plt.legend(inside_lines, [l.get_label() for l in inside_lines], ncol=4,
                         frameon=True, framealpha=1, bbox_to_anchor=(2, 0),
                         bbox_transform=axes[0,0].transAxes)
    
    fig.set_size_inches([10.02, 7])
    
    plt.savefig(plot_file("ComparisonFluxWalls.png"))
    
    # Me falta ver si a este gráfico le puedo poner una colorbar en vez de leyenda.
    
    fig = plt.figure()
    plt.imshow(np.array([np.array(flux_wall_positions[i][0]) * np.array(index[i][0]) / np.mean(wlen_range[i][0]) for i in range(len(series))]),
               cmap=plab.cm.Greens)
    plt.colorbar(label=trs.choose(r"Position $X$ [$\lambda_0/n$]", r"Posición  $X$ [$\lambda_0/n$]"))
    fig.set_size_inches([10.02, 7])
    plt.savefig(plot_file("ComparisonFluxWallsColorbar.png"))
    
    if plot_for_display: use_backend("Qt5Agg")