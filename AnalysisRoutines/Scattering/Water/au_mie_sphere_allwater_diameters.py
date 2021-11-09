#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 12:24:04 2020

@author: vall
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plab
import matplotlib.gridspec as gridspec
import os
import PyMieScatt as ps
from vmp_materials import import_medium
import v_save as vs
import v_utilities as vu
import vmp_materials as vml

english = False
trs = vu.BilingualManager(english=english)

#%% PARAMETERS

# Saving directories
folder = ["Scattering/AuSphere/AllWatDiam"]
home = vs.get_home()

# Sorting and labelling data series
sorting_function = [vu.sort_by_number]
series_must = [""] # leave "" per default
series_mustnt = ["More"] # leave "" per default
series_column = [1]

# Scattering plot options
plot_title = trs.choose("Scattering for Au spheres of different diameters in water",
                        "Dispersión de esferas de Au de diferentes diámetros en agua")
series_colors = [plab.cm.Blues]
series_label = [lambda s : trs.choose("MEEP Data", "Datos MEEP") + 
                f" {vu.find_numbers(s)[0]} nm"]
series_linestyles = ["solid"]
theory_label = lambda s : trs.choose("Mie Theory", "Teoría Mie") + f" {vu.find_numbers(s)[0]} nm"
theory_linestyle = "dashed"
plot_make_big = True
plot_file = lambda n : os.path.join(home, "DataAnalysis", "AllWaterDiameters"+n)

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
    series[-1] = vu.filter_to_only_directories(series[-1])
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
material = []
paper = []
index = []
for p in params:
    r.append( [pi["r"] for pi in p] )
    from_um_factor.append( [pi["from_um_factor"] for pi in p] )
    resolution.append( [pi["resolution"] for pi in p] )
    index.append( [] )
    for p in params[-1]:
        try:
            index[-1].append( p["submerged_index"] )
        except KeyError:
            index[-1].append( 1.333 )
    try:
        paper.append( [pi["paper"] for pi in p])
    except:
        paper.append( ["R" for pi in p] )
    try:
        material.append( [pi["material"] for pi in p] )
    except:
        material.append( ["Au" for pi in p] )

#%% GET THEORY

theory = [[vml.sigma_scatt_meep(r[i][j] * from_um_factor[i][j] * 1e3,
                                material[i][j], paper[i][j], 
                                data[i][j][:,0], # wavelength in nm
                                surrounding_index=index[i][j],
                                asEfficiency=True) 
           for j in range(len(series[i]))] for i in range(len(series))]

max_wlen_theory = [[ data[i][j][ np.argmax(theory[i][j]) , 0 ] for j in range(len(series[i]))] for i in range(len(series))]

#%% GET MAX WAVELENGTH

# def wlen_range(material, surrounding_index):
#     if material=="Au":
#         if surrounding_index == 1: return (450, 750)
#         elif surrounding_index in (1.33, 1.333) : return (550, 850)
#     else:
#         raise ValueError("Please, expand this function.")

# max_wlen_theory = [[vml.max_scatt_meep(r[i][j], 
#                                        material[i][j], 
#                                        paper[i][j], 
#                                        wlen_range(material[i][j],
#                                                   index[i][j]),
#                                        surrounding_index=index[i][j])[0] for j in range(len(series[i]))] for i in range(len(series))]

max_wlen = []
for d, sc in zip(data, series_column):
    max_wlen.append( [d[i][np.argmax(d[i][:,sc]), 0] for i in range(len(d))] )
   
e_max_wlen = []
for d, t, sc in zip(data, theory, series_column):
    e_max_wlen.append( [ np.mean([
        abs(d[i][np.argmax(d[i][:,sc])-1, 0] - d[i][np.argmax(t[i]), 0]),
        abs(d[i][np.argmax(d[i][:,sc])+1, 0] - d[i][np.argmax(t[i]), 0])
        ]) for i in range(len(d)) ] )
dif_max_wlen = [max_wlen[0][i] - max_wlen_theory[0][i] for i in range(len(data[0]))]

#%% PLOT NORMALIZED

colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

plt.figure()
plt.title(plot_title)
for i in range(len(series)):
    for j in range(len(series[i])):
        plt.plot(data[i][j][:,0], 
                 data[i][j][:,series_column[i]] / max(data[i][j][:,series_column[i]]), 
                 linestyle=series_linestyles[i], color=colors[i][j], 
                 label=series_label[i](series[i][j]))
        plt.plot(data[i][j][:,0], 
                 theory[i][j] / max(theory[i][j]), 
                 linestyle=theory_linestyle, color=colors[i][j], 
                 label=theory_label(series[i][j]))

plt.xlabel(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
plt.ylabel(trs.choose("Normalized Scattering Cross Section",
                      "Sección eficaz de dispersión normalizada"))
plt.legend()
if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
del mng
vs.saveplot(plot_file("AllScattNorm.png"), overwrite=True)

#%% PLOT EFFIENCIENCY

colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]
        
plt.figure()
plt.title(plot_title)
for i in range(len(series)):
    for j in range(len(series[i])):
        plt.plot(data[i][j][:,0], 
                 data[i][j][:,series_column[i]], 
                 linestyle=series_linestyles[i], color=colors[i][j], 
                 label=series_label[i](series[i][j]))
        plt.plot(data[i][j][:,0], 
                 theory[i][j], 
                 linestyle=theory_linestyle, color=colors[i][j], 
                 label=theory_label(series[i][j]))

plt.xlabel(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
plt.ylabel(trs.choose("Scattering Efficiency", "Eficacia de Dispersión"))
plt.legend()
if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
del mng
vs.saveplot(plot_file("AllScattEff.png"), overwrite=True)

#%% PLOT IN UNITS

colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]
        
plt.figure()
plt.title(plot_title)
for i in range(len(series)):
    for j in range(len(series[i])):
        plt.plot(data[i][j][:,0], 
                 data[i][j][:,series_column[i]] * np.pi * (r[i][j] * from_um_factor[i][j] * 1e3)**2,
                 linestyle=series_linestyles[i], color=colors[i][j], 
                 label=series_label[i](series[i][j]))
        plt.plot(data[i][j][:,0], 
                 theory[i][j] * np.pi * (r[i][j] * from_um_factor[i][j] * 1e3)**2,
                 linestyle=theory_linestyle, color=colors[i][j], 
                 label=theory_label(series[i][j]))

plt.xlabel(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
plt.ylabel(trs.choose("Scattering Cross Section [nm$^2$]",
                      "Sección eficaz de dispersión [nm$^2$]"))
plt.legend()
if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
del mng
vs.saveplot(plot_file("AllScatt.png"), overwrite=True)

#%% ONE HUGE PLOT

colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

fig = plt.figure()
plot_grid = gridspec.GridSpec(ncols=4, nrows=2, hspace=0.5, wspace=0.5, figure=fig)

main_ax = fig.add_subplot(plot_grid[:,0:2])
main_ax.set_title(trs.choose("All Diameters", "Todos los diámetros"))
main_ax.xaxis.set_label_text(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
main_ax.yaxis.set_label_text(trs.choose("Normalized Scattering Cross Section",
                                        "Sección eficaz de dispersión normalizada"))
        
for i in range(len(series)):
    for j in range(len(series[i])):
        main_ax.plot(data[i][j][:,0], 
                     data[i][j][:,series_column[i]] / max(data[i][j][:,series_column[i]]), 
                     linestyle=series_linestyles[i], color=colors[i][j], 
                     label=series_label[i](series[i][j]))
        main_ax.plot(data[i][j][:,0], 
                     theory[i][j] / max(theory[i][j]), 
                     linestyle=theory_linestyle, color=colors[i][j], 
                     label=theory_label(series[i][j]))
main_ax.legend()

plot_list = [[plot_grid[0,2], plot_grid[0,3], plot_grid[1,2], plot_grid[1,3]]]
axes_list = [[ fig.add_subplot(pl) for pl in plot_list[0] ]]

for i in range(len(series)):
    for j in range(len(series[i])):
        axes_list[i][j].set_title(trs.choose("Diameter", "Diámetro") + 
                                  f" {2 * r[i][j] * from_um_factor[i][j] * 1e3: 1.0f} nm")
        axes_list[i][j].plot(data[i][j][:,0], 
                             data[i][j][:,series_column[i]] * np.pi * (r[i][j] * from_um_factor[i][j] * 1e3)**2,
                             linestyle=series_linestyles[i], color=colors[i][j], 
                             label=series_label[i](series[i][j]))
        axes_list[i][j].plot(data[i][j][:,0], 
                             theory[i][j] * np.pi * (r[i][j] * from_um_factor[i][j] * 1e3)**2,
                             linestyle=theory_linestyle, color=colors[i][j], 
                             label=theory_label(series[i][j]))
        axes_list[i][j].xaxis.set_label_text(r"Longitud de onda $\lambda$ [nm]")
        axes_list[i][j].yaxis.set_label_text("Sección eficaz [nm$^2$]")
        axes_list[i][j].xaxis.set_label_text(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
        axes_list[i][j].yaxis.set_label_text(trs.choose("Scattering Cross Section [nm$^2$]",
                                                        "Sección eficaz de dispersión [nm$^2$]"))

fig.set_size_inches([13.09,  5.52])

vs.saveplot(plot_file("AllScattBig.png"), overwrite=True)