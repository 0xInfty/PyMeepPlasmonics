#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 12:24:04 2020

@author: vall
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plab
import os
import PyMieScatt as ps
from vmp_materials import import_medium
import v_save as vs
import v_utilities as vu

#%% PARAMETERS ALL VACUUM

# Saving directories
folder = ["AuMieSphere/AuMie/7)Diameters/WLen4560",
          "AuMieSphere/AuMie/13)TestPaper/4)PaperJCFit/TestPaperJCFitDiams/AllVac450600"]
home = vs.get_home()

# Sorting and labelling data series
sorting_function = [vu.sort_by_number, vu.sort_by_number]
series_must = ["Res4", ""] # leave "" per default
series_mustnt = ["", ""] # leave "" per default
series_column = [1, 1]

# Scattering plot options
plot_title = "Scattering for Au spheres of different diameters and material source in vacuum"
series_colors = [plab.cm.Blues, plab.cm.Reds]
series_label = [lambda s : f"Vacuum R Meep {vu.find_numbers(s)[0]} nm",
                lambda s : f"Vacuum JC Meep {vu.find_numbers(s)[0]} nm"]
series_linestyles = ["solid", "solid"]

theory_label = [lambda s : f"Vacuum R Theory {vu.find_numbers(s)[0]} nm",
                lambda s : f"Vacuum JC Theory {vu.find_numbers(s)[0]} nm"]
theory_linestyle = ["dashed", "dashed"]

plot_make_big = True
plot_file = lambda n : os.path.join(home, "DataAnalysis/PaperDiameters", "AllVacPaperDiameters"+n)

#%% PARAMETERS ALL WATER

# Saving directories
folder = ["AuMieMediums/AllWater",
          "AuMieSphere/AuMie/13)TestPaper/4)PaperJCFit/TestPaperJCFitDiams/AllWater500650"]
home = vs.get_home()

# Sorting and labelling data series
sorting_function = [vu.sort_by_number, vu.sort_by_number]
series_must = ["Res4", ""] # leave "" per default
series_mustnt = ["", ""] # leave "" per default
series_column = [1, 1]

# Scattering plot options
plot_title = "Scattering for Au spheres of different diameters and material source in water"
series_colors = [plab.cm.Purples, plab.cm.Reds]
series_label = [lambda s : f"Water R Meep {vu.find_numbers(s)[0]} nm",
                lambda s : f"Water JC Meep {vu.find_numbers(s)[0]} nm"]
series_linestyles = ["solid", "solid"]

theory_label = [lambda s : f"Water R Theory {vu.find_numbers(s)[0]} nm",
                lambda s : f"Water JC Theory {vu.find_numbers(s)[0]} nm"]
theory_linestyle = ["dashed", "dashed"]

plot_make_big = True
plot_file = lambda n : os.path.join(home, "DataAnalysis/PaperDiameters", "AllWatPaperDiameters"+n)

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
paper = []
index = []
for p in params:
    r.append( [pi["r"] for pi in p] )
    from_um_factor.append( [pi["from_um_factor"] for pi in p] )
    resolution.append( [pi["resolution"] for pi in p] )
    try:
        paper.append( [pi["paper"] for pi in p])
    except:
        paper.append( ["R" for pi in p] )
    index.append( [] )
    for p in params[-1]:
        try:
            index[-1].append( p["submerged_index"] )
        except KeyError:
            index[-1].append( 1.333 )

#%% GET THEORY

theory = [] # Scattering effiency
for di, ri, fi, resi, ppi, ii in zip(data, r, from_um_factor, resolution, paper, index):
    theory.append([])    
    for dj, rj, fj, resj, ppij, ij in zip(di, ri, fi, resi, ppi, ii):
        wlenj = dj[:,0] # nm
        freqj = 1 / wlenj # 1/nm
        freqmeepj = (1e3 * fj) / wlenj # Meep units
        mediumj = import_medium("Au", from_um_factor=fj, paper=ppij)
        theory[-1].append(np.array(
            [ps.MieQ(np.sqrt(mediumj.epsilon(fqm)[0,0]*mediumj.mu(fqm)[0,0]), 
                     wl, # Wavelength (nm)
                     2*rj*1e3*fj, # Diameter (nm)
                     nMedium=ij, # Refraction Index of Medium
                     asDict=True)['Qsca'] 
             for wl, fq, fqm in zip(wlenj, freqj, freqmeepj)]))

#%% GET MAX WAVELENGTH

max_wlen = []
for d, sc in zip(data, series_column):
    max_wlen.append( [d[i][np.argmax(d[i][:,sc]), 0] for i in range(len(d))] )

max_wlen_theory = []
for t, d in zip(theory, data):
    max_wlen_theory.append( [d[i][np.argmax(t[i]), 0] for i in range(len(t))] )
    
max_wlen_diff = []
for md, mt in zip(max_wlen, max_wlen_theory):
    max_wlen_diff.append( [d - t for d,t in zip(md, mt)] )
    
e_max_wlen = []
for d, sc in zip(data, series_column):
    e_max_wlen.append( [ np.mean([
        abs(d[i][np.argmax(d[i][:,sc])-1, 0] - d[i][np.argmax(t[i]), 0]),
        abs(d[i][np.argmax(d[i][:,sc])+1, 0] - d[i][np.argmax(t[i]), 0])
        ]) for i in range(len(d)) ] )

#%% PLOT NORMALIZED

colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

fig, axes = plt.subplots(2, 1, sharex=True)
fig.suptitle(plot_title)
axes[1].yaxis.tick_right()
axes[1].yaxis.set_label_position("right")
    
for ax, s, d, t, p, n, sc, psl, tsl, pc, pls, tls in zip(axes, series, 
                                                         data, theory, 
                                                         params, index, 
                                                         series_column, 
                                                         series_label, 
                                                         theory_label, 
                                                         colors, 
                                                         series_linestyles, 
                                                         theory_linestyle):

    for ss, sd, td, sp, nd, spc in zip(s, d, t, p, n, pc):
        ax.plot(sd[:,0], sd[:,sc] / max(sd[:,sc]), 
                linestyle=pls, color=spc, label=psl(ss))
        ax.plot(sd[:,0], td / max(td),
                linestyle=tls, color=spc, label=tsl(ss))
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_label_text(r"Wavelength $\lambda$ [nm]")
        ax.yaxis.set_label_text("Normalized Scattering Cross Section")
        if 1.33 <= nd < 1.34:
            ax.set_facecolor( np.array([230, 241, 255])/255 )
            # frame = leg.get_frame()
            # frame.set_facecolor( np.array([230, 241, 255])/255 )

# plt.xlabel(r"Wavelength $\lambda$ [nm]")
# plt.ylabel("Normalized Scattering Cross Section")
if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
del mng
fig.tight_layout()
plt.subplots_adjust(hspace=0)

vs.saveplot(plot_file("AllScattNorm.png"), overwrite=True)

#%% PLOT EFFIENCIENCY

colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

fig, axes = plt.subplots(2, 1, sharex=True)
fig.suptitle(plot_title)
axes[1].yaxis.tick_right()
axes[1].yaxis.set_label_position("right")
    
for ax, s, d, t, p, n, sc, psl, tsl, pc, pls, tls in zip(axes, series, 
                                                         data, theory, 
                                                         params, index, 
                                                         series_column, 
                                                         series_label, 
                                                         theory_label, 
                                                         colors, 
                                                         series_linestyles, 
                                                         theory_linestyle):

    for ss, sd, td, sp, nd, spc in zip(s, d, t, p, n, pc):
        ax.plot(sd[:,0], sd[:,sc], 
                linestyle=pls, color=spc, label=psl(ss))
        ax.plot(sd[:,0], td,
                linestyle=tls, color=spc, label=tsl(ss))
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_label_text(r"Wavelength $\lambda$ [nm]")
        ax.yaxis.set_label_text("Scattering Efficiency")
        if 1.33 <= nd < 1.34:
            ax.set_facecolor( np.array([230, 241, 255])/255 )
            # frame = leg.get_frame()
            # frame.set_facecolor( np.array([230, 241, 255])/255 )

# plt.xlabel(r"Wavelength $\lambda$ [nm]")
# plt.ylabel("Normalized Scattering Cross Section")
if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
del mng
fig.tight_layout()
plt.subplots_adjust(hspace=0)

vs.saveplot(plot_file("AllScattEff.png"), overwrite=True)

#%% PLOT IN UNITS

colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

fig, axes = plt.subplots(2, 1, sharex=True)
fig.suptitle(plot_title)
axes[1].yaxis.tick_right()
axes[1].yaxis.set_label_position("right")
    
for ax, s, d, t, p, n, sc, psl, tsl, pc, pls, tls in zip(axes, series, 
                                                         data, theory, 
                                                         params, index, 
                                                         series_column, 
                                                         series_label, 
                                                         theory_label, 
                                                         colors, 
                                                         series_linestyles, 
                                                         theory_linestyle):

    for ss, sd, td, sp, nd, spc in zip(s, d, t, p, n, pc):
        ax.plot(sd[:,0], sd[:,sc]  * np.pi * (sp['r'] * sp['from_um_factor'] * 1e3)**2,
                linestyle=pls, color=spc, label=psl(ss))
        ax.plot(sd[:,0], td * np.pi * (sp['r'] * sp['from_um_factor'] * 1e3)**2, 
                linestyle=tls, color=spc, label=tsl(ss))
        leg = ax.legend()
        ax.grid(True)
        ax.xaxis.set_label_text(r"Wavelength $\lambda$ [nm]")
        ax.yaxis.set_label_text(r"Scattering Cross Section [nm$^2$]")
        if 1.33 <= nd < 1.34:
            ax.set_facecolor( np.array([230, 241, 255])/255 )
            # frame = leg.get_frame()
            # frame.set_facecolor( np.array([230, 241, 255])/255 )

# plt.xlabel(r"Wavelength $\lambda$ [nm]")
# plt.ylabel("Normalized Scattering Cross Section")
if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
del mng
fig.tight_layout()
plt.subplots_adjust(hspace=0)

vs.saveplot(plot_file("AllScatt.png"), overwrite=True)
