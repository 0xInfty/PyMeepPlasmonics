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
from v_materials import import_medium
import v_save as vs
import v_utilities as vu
import v_materials as vmt

english = False
trs = vu.BilingualManager(english=english)

#%% PARAMETERS

# Saving directories
folder = ["Scattering/AuSphere/AllVacTest/7)Diameters/WLen4560",
          "Scattering/AuSphere/AllWatDiam"]
home = vs.get_home()

# Sorting and labelling data series
sorting_function = [vu.sort_by_number]*2
series_must = ["SC", ""] # leave "" per default
series_mustnt = ["", "More"] # leave "" per default
series_column = [1]*2

# Scattering plot options
plot_title = trs.choose("Scattering for Au spheres of different diameters",
                        "Dispersión de esferas de Au de diferentes diámetros")
series_colors = [plab.cm.Reds, plab.cm.Blues]
series_label = [lambda s : trs.choose("Vacuum", "Vacío") + f" {vu.find_numbers(s)[0]} nm",
                lambda s : trs.choose("Water", "Agua") + f" {vu.find_numbers(s)[0]} nm"]
series_linestyles = ["solid"]*2
theory_linestyles = ["dashed"]*2

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
for i, p in enumerate(params):
    r.append( [pi["r"] for pi in p] )
    from_um_factor.append( [pi["from_um_factor"] for pi in p] )
    resolution.append( [pi["resolution"] for pi in p] )
    index.append( [] )
    for p in params[-1]:
        try:
            index[-1].append( p["submerged_index"] )
        except KeyError:
            if i==0:
                index[-1].append( 1 )
            else:
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

# theory = [[vmt.sigma_scatt_meep(r[i][j], material[i][j], paper[i][j], 
#                                 data[i][j][:,0], # wavelength in nm
#                                 index[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

theory = [] # Scattering effiency
for i in range(len(series)):
    theory.append([])    
    for j in range(len(series[i])):
        wlen_ij = data[i][j][:,0] # nm
        freq_ij = 1 / wlen_ij # 1/nm
        freqmeep_ij = (1e3 * from_um_factor[i][j]) / wlen_ij # Meep units
        medium_ij = import_medium(material[i][j], 
                                  paper=paper[i][j],
                                  from_um_factor=from_um_factor[i][j])
        theory[-1].append(np.array(
            [ps.MieQ(np.sqrt(medium_ij.epsilon(freqmeep_ij[k])[0,0] * medium_ij.mu(freqmeep_ij[k])[0,0]), 
                     wlen_ij[k], # Wavelength (nm)
                     2*r[i][j]*1e3*from_um_factor[i][j], # Diameter (nm)
                     nMedium=index[i][j], # Refraction Index of Medium
                     asDict=True)['Qsca'] 
             for k in range(len(wlen_ij))]))

#%% GET MAX WAVELENGTH

# max_wlen_theory = [[ data[i][j][ np.argmax(theory[i][j]) , 0 ] for j in range(len(series[i]))] for i in range(len(series))]

# def wlen_range(material, surrounding_index):
#     if material=="Au":
#         if surrounding_index == 1: return (450, 750)
#         elif surrounding_index in (1.33, 1.333) : return (550, 850)
#     else:
#         raise ValueError("Please, expand this function.")

# max_wlen_theory = [[vmt.max_scatt_meep(r[i][j], 
#                                        material[i][j], 
#                                        paper[i][j], 
#                                        wlen_range(material[i][j],
#                                                   index[i][j]),
#                                        surrounding_index=index[i][j])[0] for j in range(len(series[i]))] for i in range(len(series))]

max_wlen = []
for d, sc in zip(data, series_column):
    max_wlen.append( [d[i][np.argmax(d[i][:,sc]), 0] for i in range(len(d))] )

max_wlen_theory = []
for t, d in zip(theory, data):
    max_wlen_theory.append( [d[i][np.argmax(t[i]), 0] for i in range(len(t))] )
    
e_max_wlen = []
for d, sc in zip(data, series_column):
    e_max_wlen.append( [ np.mean([
        abs(d[i][np.argmax(d[i][:,sc])-1, 0] - d[i][np.argmax(t[i]), 0]),
        abs(d[i][np.argmax(d[i][:,sc])+1, 0] - d[i][np.argmax(t[i]), 0])
        ]) for i in range(len(d)) ] )
dif_max_wlen = [max_wlen[0][i] - max_wlen_theory[0][i] for i in range(len(data[0]))]

#%% PLOT NORMALIZED

colors = [sc(np.linspace(0,1,len(s)+2))[2:] 
          for sc, s in zip(series_colors, series)]

plt.figure()
plt.title(plot_title)

lines_origin = []
lines_series = []
for i in range(len(series)):
    for j in range(len(series[i])):
        l_meep, = plt.plot(data[i][j][:,0], 
                           data[i][j][:,series_column[i]] / max(data[i][j][:,series_column[i]]), 
                           linestyle=series_linestyles[i], color=colors[i][j], 
                           label=series_label[i](series[i][j]))
        l_theory, = plt.plot(data[i][j][:,0], 
                             theory[i][j] / max(theory[i][j]), 
                             linestyle=theory_linestyles[i], color=colors[i][j], 
                             label=theory_label[i](series[i][j]))
        if i==0 and j==len(series[0])-1:
            lines_origin.append( [l_meep, l_theory] )
        lines_series.append(l_meep)

plt.xlabel(trs.choose("Wavelength [nm]", "Longitud de onda [nm]"))
plt.ylabel(trs.choose("Normalized Scattering Cross Section",
                      "Sección eficaz de dispersión normalizada"))
# plt.legend()

first_legend = plt.legend(lines_origin, trs.choose(["CM Theory", "Ku Theory"],
                                               ["Teoría CM", "Teoría Ku"]),
                          loc="center")
second_legend = plt.legend(
    l_series, 
    [l.get_label() for l in l_series],
    loc="upper center")
plt.gca().add_artist(first_legend)

if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
del mng
vs.saveplot(plot_file("AllScattNorm.png"), overwrite=True)

#%% PLOT EFFIENCIENCY

colors = [sc(np.linspace(0,1,len(s)+2))[2:] 
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
                 linestyle=theory_linestyles[i], color=colors[i][j], 
                 label=theory_label[i](series[i][j]))

plt.xlabel(trs.choose("Wavelength [nm]", "Longitud de onda [nm]"))
plt.ylabel(trs.choose("Scattering Efficiency", "Eficacia de Dispersión"))
plt.legend()
if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
del mng
vs.saveplot(plot_file("AllScattEff.png"), overwrite=True)

#%% PLOT IN UNITS

colors = [sc(np.linspace(0,1,len(s)+2))[2:] 
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
                 linestyle=theory_linestyles[i], color=colors[i][j], 
                 label=theory_label[i](series[i][j]))

plt.xlabel(trs.choose("Wavelength [nm]", "Longitud de onda [nm]"))
plt.ylabel(trs.choose("Scattering Cross Section [nm$^2$]",
                      "Sección eficaz de dispersión [nm$^2$]"))
plt.legend()
if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
del mng
vs.saveplot(plot_file("AllScatt.png"), overwrite=True)

#%% ONE HUGE PLOT

colors = [sc(np.linspace(0,1,len(s)+2))[2:] 
          for sc, s in zip(series_colors, series)]

fig = plt.figure()
plot_grid = gridspec.GridSpec(ncols=4, nrows=2, hspace=0.4, wspace=0.45, figure=fig)

main_ax = fig.add_subplot(plot_grid[:,0:2])
main_ax.set_title(trs.choose("All Diameters", "Todos los diámetros"))
main_ax.xaxis.set_label_text(trs.choose("Wavelength [nm]", "Longitud de onda [nm]"))
main_ax.yaxis.set_label_text(trs.choose("Normalized Scattering Cross Section",
                                        "Sección eficaz de dispersión normalizada"))
        
lines_origin = []
lines_series = []
for i in range(len(series)):
    for j in range(len(series[i])):
        l_meep, = main_ax.plot(data[i][j][:,0], 
                               data[i][j][:,series_column[i]] / max(data[i][j][:,series_column[i]]), 
                               linestyle=series_linestyles[i], color=colors[i][j], 
                               label=series_label[i](series[i][j]))
        l_theory, = main_ax.plot(data[i][j][:,0], 
                                 theory[i][j] / max(theory[i][j]), 
                                 linestyle=theory_linestyles[i], color=colors[i][j], 
                                 label=series_label[i](series[i][j]))
        if i==0 and j==len(series[0])-1:
            lines_origin = [l_meep, l_theory]
        lines_series.append(l_meep)
# main_ax.legend()

first_legend = main_ax.legend(lines_origin, trs.choose(["MEEP Data", "Mie Theory"],
                                                       ["Datos MEEP", "Teoría Mie"]),
                          loc="lower left")
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
        axes_list[i][j].set_title(f"Diámetro {2 * r[i][j] * from_um_factor[i][j] * 1e3: 1.0f} nm")
        if i == 0:
            ax = axes_list[i][j]
        else:
            ax = right_axes_list[i][j]
        ax.plot(data[i][j][:,0], 
                data[i][j][:,series_column[i]] * np.pi * (r[i][j] * from_um_factor[i][j] * 1e3)**2,
                linestyle=series_linestyles[i], color=colors[i][j], 
                label=series_label[i](series[i][j]))
        ax.plot(data[i][j][:,0], 
                theory[i][j] * np.pi * (r[i][j] * from_um_factor[i][j] * 1e3)**2,
                linestyle=theory_linestyles[i], color=colors[i][j], 
                label=series_label[i](series[i][j]))
        ax.xaxis.set_label_text(trs.choose("Wavelength [nm]", "Longitud de onda [nm]"))
        if i == 0 and (j == 0 or j == 2):
            ax.yaxis.set_label_text(trs.choose("Vacuum \n Scattering Cross Section [nm$^2$]",
                                               "Vacío \n Sección eficaz de dispersión [nm$^2$]"))
        if i == 1 and (j == 1 or j == 3):
            ax.yaxis.set_label_text(trs.choose("Scattering Cross Section [nm$^2$] \n Water",
                                               "Sección eficaz de dispersión [nm$^2$] \n Agua"))

fig.set_size_inches([18.45,  6.74])

vs.saveplot(plot_file("AllScattBig.png"), overwrite=True)