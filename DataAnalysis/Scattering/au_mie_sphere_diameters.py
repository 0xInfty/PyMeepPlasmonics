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
marian_folder = "Scattering/AuSphere/AllVacTest/7)Diameters/Marians"
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
        
minor_division = [[fum * 1e3 / res for fum, res in zip(frum, reso)] for frum, reso in zip(from_um_factor, resolution)]
try:
    width_points = [[int(p["cell_width"] * p["resolution"]) for p in par] for par in params] 
    effective_width_points = [[(p["cell_width"] - 2 * params["pml_width"]) * p["resolution"] for p in par] for par in params]
except:
    width_points = [[2*int((p["pml_width"] + p["air_width"] + p["r"]) * p["resolution"]) for p in par] for par in params] 
    effective_width_points = [[2*int((p["air_width"] + p["r"]) * p["resolution"]) for p in par] for par in params]
grid_points = [[wp**3 for wp in wpoints] for wpoints in width_points]
memory_B = [[2 * 12 * gp * 32 for p, gp in zip(par, gpoints)] for par, gpoints in zip(params, grid_points)] # in bytes

#%% GET EXPERIMENTAL DATA

marian_path = os.path.join(home, marian_folder)
marian_file = lambda s : os.path.join(marian_path, s)

marian_series = os.listdir(marian_path)

marian_series = vu.filter_by_string_must(marian_series, "exp")
marian_series = vu.filter_by_string_must(marian_series, "glass")
marian_series = vu.sort_by_number(marian_series)

marian_data = []
for s in marian_series:
    marian_data.append(np.loadtxt(marian_file(s)))
marian_data = [marian_data]

#%% GET THEORY

theory = [[vmt.sigma_scatt_meep(r[i][j] * from_um_factor[i][j] * 1e3,
                                material[i][j], paper[i][j], 
                                data[i][j][:,0], # wavelength in nm
                                surrounding_index=index[i][j],
                                asEffiency=True) 
           for j in range(len(series[i]))] for i in range(len(series))]

max_wlen_theory = [[ data[i][j][ np.argmax(theory[i][j]) , 0 ] for j in range(len(series[i]))] for i in range(len(series))]

#%% GET MAX WAVELENGTH

# def wlen_range(material, surrounding_index):
#     if material=="Au":
#         if surrounding_index == 1: return (450, 750)
#         elif surrounding_index in (1.33, 1.333) : return (550, 850)
#     else:
#         raise ValueError("Please, expand this function.")

# max_wlen_theory = [[vmt.max_scatt_meep(r[i][j] * from_um_factor[i][j] * 1e3, 
#                                        material[i][j], 
#                                        paper[i][j], 
#                                        wlen_range(material[i][j],
#                                                   index[i][j]),
#                                        surrounding_index=index[i][j])[0] for j in range(len(series[i]))] for i in range(len(series))]

max_wlen = []
for d, sc in zip(data, series_column):
    max_wlen.append( [d[i][np.argmax(d[i][:,sc]), 0] for i in range(len(d))] )

max_wlen_marian = []
for md in marian_data[0]:
    max_wlen_marian.append( md[np.argmax(md[:,1]), 0] )
max_wlen_marian = [max_wlen_marian]
    
e_max_wlen = []
for d, t, sc in zip(data, theory, series_column):
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
for i in range(len(series)):
    for j in range(len(series[i])):
        plt.plot(data[i][j][:,0], 
                 data[i][j][:,series_column[i]] / max(data[i][j][:,series_column[i]]), 
                 linestyle=series_linestyles[i], color=colors[i][j], 
                 label=trs.choose("MEEP Data ", "Data MEEP ") + series_label[i](series[i][j]))
        plt.plot(data[i][j][:,0], 
                 theory[i][j] / max(theory[i][j]), 
                 linestyle=theory_linestyles[i], color=colors[i][j], 
                 label=trs.choose("Mie Theory ", "Teoría Mie ") + series_label[i](series[i][j]))

plt.xlabel(trs.choose("Wavelength [nm]", "Longitud de onda [nm]"))
plt.ylabel(trs.choose("Normalized Scattering Cross Section",
                      "Sección eficaz de dispersión normalizada"))
plt.legend(ncol=2)

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
                 label=trs.choose("MEEP Data ", "Data MEEP ") + series_label[i](series[i][j]))
        plt.plot(data[i][j][:,0], 
                 theory[i][j], 
                 linestyle=theory_linestyles[i], color=colors[i][j], 
                 label=trs.choose("Mie Theory ", "Teoría Mie ") + series_label[i](series[i][j]))

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
                 label=trs.choose("MEEP Data ", "Data MEEP ") + series_label[i](series[i][j]))
        plt.plot(data[i][j][:,0], 
                 theory[i][j] * np.pi * (r[i][j] * from_um_factor[i][j] * 1e3)**2,
                 linestyle=theory_linestyles[i], color=colors[i][j], 
                 label=trs.choose("Mie Theory ", "Teoría Mie ") + series_label[i](series[i][j]))

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