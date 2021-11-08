#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 01:01:53 2021

@author: vall
"""

import imageio as mim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import vmp_utilities as vmu
import vmp_analysis as vma
import v_plot as vp
import v_utilities as vu

from scipy.signal import find_peaks
from matplotlib import use as use_backend

vp.set_style()

#%% PARAMETERS

series = "ResWlen10" # Para MonochNormalizationPlot
folder = "Field/Sources/MonochPlanewave/TestRes/Not Centered/Vacuum"

hfield = False

make_plots = True
make_gifs = True

english = False
maxnframes = 300
        
#%% SETUP

# Computation
pm = vmu.ParallelManager()
n_processes, n_cores, n_nodes = pm.specs
parallel = pm.parallel

# Saving directories
sa = vmu.SavingAssistant(series, folder)

trs = vu.BilingualManager(english=english)

#%% GET READY TO LOAD DATA
    
f = pm.hdf_file(sa.file("Field-Lines.h5"), "r+")
results_line = f["Ez"]
t_line = np.array(f["T"])
x_line = np.array(f["X"])

g = pm.hdf_file(sa.file("Field-Planes.h5"), "r+")
results_plane = g["Ez"]
t_plane = np.array(g["T"])
y_plane = np.array(g["Y"])
z_plane = np.array(g["Z"])

params = dict(f["Ez"].attrs)

from_um_factor = params["from_um_factor"]
wlen = params["wlen"]
submerged_index = params["submerged_index"]

cell_width = params["cell_width"]
pml_width = params["pml_width"]
source_center = params["source_center"]

until_time = params["until_time"]
period_line = params["period_line"]
period_plane = params["period_plane"]
# period = submerged_index * wlen

units = params["units"]
try:
    norm_amplitude, norm_period = params["norm_amplitude"], params["norm_period"]
    requires_normalization = False
except:
    requires_normalization = True

if units:
    plot_title_base = trs.choose('Monochromatic wave ', 
                                 "Onda monocromática de ") + f' {wlen * from_um_factor * 1e3:.0f} nm'
else:
    plot_title_base = trs.choose('Dimnesionless monochromatic wave', 
                                 "Onda monocromática adimensional")

#%% POSITION RECONSTRUCTION

t_line_index = vma.def_index_function(t_line)
x_line_index = vma.def_index_function(x_line)

t_plane_index = vma.def_index_function(t_plane)
y_plane_index = vma.def_index_function(y_plane)
z_plane_index = vma.def_index_function(z_plane)

x_line_cropped = x_line[:x_line_index(cell_width/2 - pml_width)+1]
x_line_cropped = x_line_cropped[x_line_index(-cell_width/2 + pml_width):]

y_plane_cropped = y_plane[:y_plane_index(cell_width/2 - pml_width)+1]
y_plane_cropped = y_plane_cropped[y_plane_index(-cell_width/2 + pml_width):]

z_plane_cropped = z_plane[:z_plane_index(cell_width/2 - pml_width)+1]
z_plane_cropped = z_plane_cropped[z_plane_index(-cell_width/2 + pml_width):]

#%% DATA EXTRACTION

source_results = vma.get_source_from_line(results_line, x_line_index, source_center)

if not requires_normalization:
    
    period_results, amplitude_results = norm_period, norm_amplitude
    
else:
    
    period = vma.get_period_from_source(source_results, t_line)[-1]
    amplitude = vma.get_amplitude_from_source(source_results)[-1]
    
    results_plane = np.asarray(results_plane) / amplitude
    results_line = np.asarray(results_line) / amplitude
    
results_cropped_line = vma.crop_field_xprofile(results_line, x_line_index,
                                               cell_width, pml_width)

#%% PEAKS DETECTION

source_field = source_results
peaks_sep_sensitivity = 0.10
periods_sensitivity = 0.03
amplitude_sensitivity = 0.02
last_stable_periods = 5

peaks = vma.get_peaks_from_source(source_results,
                                  peaks_sep_sensitivity=peaks_sep_sensitivity,
                                  last_stable_periods=last_stable_periods)

period_peaks, period = vma.get_period_from_source(source_field, t_line,
                                                  peaks_sep_sensitivity=peaks_sep_sensitivity,
                                                  periods_sensitivity=periods_sensitivity,
                                                  last_stable_periods=last_stable_periods)

amplitude_peaks, amplitude = vma.get_amplitude_from_source(source_field,
                                                           peaks_sep_sensitivity=peaks_sep_sensitivity,
                                                           amplitude_sensitivity=amplitude_sensitivity,
                                                           last_stable_periods=last_stable_periods)

#%% PLOT

plot_for_display = True

if plot_for_display: use_backend("Agg")

fig = plt.figure()
if plot_for_display: fig.dpi = 200

plt.title(plot_title_base)
plt.plot(t_line, source_results, label="Señal")
plt.axhline(color="k", linewidth=0.5)

plt.xlabel(trs.choose("Time [MPu]", "Tiempo [uMP]"))
plt.ylabel(trs.choose(r"Electric Field $E_z(y=z=0)$ [a.u.]", 
                      r"Campo eléctrico $E_z(y=z=0)$ [u.a.]"))

plt.plot(t_line[peaks], source_results[peaks], "ok", label=f"Picos detectados con {peaks_sep_sensitivity*100:.0f}%")

plt.plot(t_line[period_peaks], source_results[period_peaks], "ok", 
         label=f"Tomado para período con {periods_sensitivity*100:.0f}%", markersize=7, markeredgecolor="r")

plt.plot(t_line[amplitude_peaks], source_results[amplitude_peaks], "o", color="gold",
         label=f"Tomado para amplitud con {amplitude_sensitivity*100:.0f}%", markersize=4, markeredgewidth=1, markeredgecolor="k")

box = fig.axes[0].get_position()
box_height = box.y1 - box.y0
# box.y1 = box.y1 - .02 * box_height
box.y0 = box.y0 + .25 * box_height
fig.axes[0].set_position(box)

fig.set_size_inches([6.4 , 6.38])

fig.axes[0].legend(loc="lower center", frameon=False, 
                   bbox_to_anchor=(.5,-.5), bbox_transform=fig.axes[0].transAxes)

plt.savefig("MonochNormalization.png")

if plot_for_display: use_backend("Qt5Agg")