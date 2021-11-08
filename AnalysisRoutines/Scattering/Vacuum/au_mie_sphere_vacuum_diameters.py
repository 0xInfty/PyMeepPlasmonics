#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 12:24:04 2020

@author: vall
"""

import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.pylab as plab
import os
import v_save as vs
import v_utilities as vu

english = False
trs = vu.BilingualManager(english=english)

#%% PARAMETERS

# Saving directories
folder = ["Scattering/AuSphere/AllVacTest/7)Diameters/WLen4560",
          "Scattering/AuSphere/AllVacTest/7)Diameters/WLen4560"]
home = vs.get_home()

# Sorting and labelling data series
sorting_function = [vu.sort_by_number, vu.sort_by_number]
series_label = [lambda s : f"Meep {vu.find_numbers(s)[0]} nm",
                lambda s : f"Mie {vu.find_numbers(s)[0]} nm"]
series_label = [lambda s : trs.choose("MEEP Data", "Datos MEEP") + f" {vu.find_numbers(s)[0]} nm",
                lambda s : trs.choose("Mie Theory", "Teoría Mie") + f" {vu.find_numbers(s)[0]} nm",]
series_must = ["SC", "SC"] # leave "" per default
series_column = [1, 2]

# Scattering plot options
plot_title = trs.choose("Scattering for Au spheres of different diameters in vacuum",
                        "Dispersión de esferas de Au de diferentes diámetros en vacío")
series_colors = [plab.cm.Reds, plab.cm.Reds]
series_linestyles = ["solid", "dashed"]
plot_make_big = True
plot_file = lambda n : os.path.join(home, "DataAnalysis", n)

#%% LOAD DATA

path = []
file = []
series = []
data = []
params = []
header = []

for f, sf, sm in zip(folder, sorting_function, series_must):

    path.append( os.path.join(home, f) )
    file.append( lambda f, s : os.path.join(path[-1], f, s) )
    
    series.append( os.listdir(path[-1]) )
    series[-1] = vu.filter_to_only_directories(series[-1])
    series[-1] = vu.filter_by_string_must(series[-1], sm)
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
    
    # r = [p["r"] for p in params]
    # from_um_factor = [p["from_um_factor"] for p in params]

#%% GET MAX WAVELENGTH

max_wlen = []
for d, sc in zip(data, series_column):
    max_wlen.append( [d[i][np.argmax(d[i][:,sc]), 0] for i in range(len(d))] )
dif_max_wlen = [max_wlen[0][i] - max_wlen[1][i] for i in range(len(data[0]))]

e_wlen = []
for d in data:
    e_wlen.append( np.mean([ d[i][j, 0] - d[i][j+1, 0] 
                    for j in range(len(d[i])-1) 
                    for i in range(len(d))]) )

max_wlen = []
e_max_wlen = []
part_max_wlen = []
for d, sc in zip(data, series_column):
    max_wlen.append( [d[i][np.argmax(d[i][:,sc]), 0] for i in range(len(d))] )
    part_max_wlen.append( [
        [d[i][np.argmax(d[i][:,sc])-1, 0], d[i][np.argmax(d[i][:,sc])+1, 0] ]
        for i in range(len(d)) ] )
    e_max_wlen.append( [ np.mean([
        abs(d[i][np.argmax(d[i][:,sc])-1, 0] - d[i][np.argmax(d[i][:,sc]), 0]),
        abs(d[i][np.argmax(d[i][:,sc])+1, 0] - d[i][np.argmax(d[i][:,sc]), 0])
        ]) for i in range(len(d)) ] )
dif_max_wlen = [max_wlen[0][i] - max_wlen[1][i] for i in range(len(data[0]))]
e_dif_max_wlen = [np.mean([e_max_wlen[0][i], e_max_wlen[1][i]]) for i in range(len(data[0]))]

#%% NORMALIZED PLOT

colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

plt.figure()
plt.title(plot_title)
for s, d, p, sc, psl, pc, pls in zip(series, data, params, series_column, 
                                     series_label, colors, series_linestyles):

    for ss, sd, sp, spc in zip(s, d, p, pc):
        plt.plot(sd[:,0], sd[:,sc] / max(sd[:,sc]), 
                 linestyle=pls, color=spc, label=psl(ss))

plt.xlabel(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
plt.ylabel(trs.choose("Normalized Scattering Cross Section",
                      "Sección eficaz de dispersión normalizada"))
plt.legend()
if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
del mng
vs.saveplot(plot_file("AllScatt.png"), overwrite=True)

#%% NON NORMALIZED PLOT

colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

plt.figure()
plt.title(plot_title)
for s, d, p, sc, psl, pc, pls in zip(series, data, params, series_column, 
                                     series_label, colors, series_linestyles):

    for ss, sd, sp, spc in zip(s, d, p, pc):
        plt.plot(sd[:,0], sd[:,sc], 
                 linestyle=pls, color=spc, label=psl(ss))

plt.xlabel(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
plt.ylabel(trs.choose("Scattering Efficiency", "Eficacia de Dispersión"))
plt.legend()
if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
del mng
vs.saveplot(plot_file("AllScattEf.png"), overwrite=True)

#%% IN UNITS PLOT

colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

plt.figure()
plt.title(plot_title)
for s, d, p, sc, psl, pc, pls in zip(series, data, params, series_column, 
                                     series_label, colors, series_linestyles):

    for ss, sd, sp, spc in zip(s, d, p, pc):
        plt.plot(sd[:,0], sd[:,sc] * np.pi * (sp['r'] * sp['from_um_factor'] * 1e3)**2, 
                 linestyle=pls, color=spc, label=psl(ss))

plt.xlabel(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
plt.ylabel(trs.choose("Scattering Cross Section [nm$^2$]",
                      "Sección eficaz de dispersión [nm$^2$]"))
plt.legend()
if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
del mng
vs.saveplot(plot_file("AllScattSec.png"), overwrite=True)


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

for s, d, p, sc, psl, pc, pls in zip(series, data, params, series_column, 
                                     series_label, colors, series_linestyles):

    for ss, sd, sp, spc in zip(s, d, p, pc):
        main_ax.plot(sd[:,0], sd[:,sc] / max(sd[:,sc]), 
                     linestyle=pls, color=spc, label=psl(ss))
main_ax.legend()

plot_list = [plot_grid[0,2], plot_grid[0,3], plot_grid[1,2], plot_grid[1,3]]
axes_list = []
for pl in plot_list:
    axes_list.append(fig.add_subplot(pl))

for s, d, p, sc, psl, pc, pls in zip(series, data, params, series_column, 
                                         series_label, colors, series_linestyles):    
    for ax, ss, sd, sp, spc in zip(axes_list, s, d, p, pc):
        ax.set_title(trs.choose("Diameter", "Diámetro") + 
                     f" {2*sp['r']*sp['from_um_factor']*1e3: 1.0f} nm")
        ax.plot(sd[:,0], sd[:,sc] * np.pi * (sp['r'] * sp['from_um_factor'] * 1e3)**2, 
                linestyle=pls, color=spc, label=psl(ss))
        ax.xaxis.set_label_text(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
        ax.yaxis.set_label_text(trs.choose("Scattering Cross Section [nm$^2$]",
                                           "Sección eficaz de dispersión [nm$^2$]"))

fig.set_size_inches([13.09,  5.52])

vs.saveplot(plot_file("AllScattBig.png"), overwrite=True)