#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 12:24:04 2020

@author: vall
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plab
from matplotlib.ticker import AutoMinorLocator
import os
import PyMieScatt as ps
from v_materials import import_medium
import v_save as vs
import v_utilities as vu

#%% PARAMETERS ALL VACUUM

marian_folder = "AuMieSphere/AuMie/7)Diameters/Marians"

# Saving directories
folder = ["AuMieSphere/AuMie/7)Diameters/WLen4560",
          "AuMieSphere/AuMie/13)TestPaper/4)PaperJCFit/TestPaperJCFitDiams/AllVac450600",
          "AuMieMediums/AllWater",
          "AuMieSphere/AuMie/13)TestPaper/4)PaperJCFit/TestPaperJCFitDiams/AllWater500650"]
home = vs.get_home()

# Sorting and labelling data series
sorting_function = [vu.sort_by_number, vu.sort_by_number, vu.sort_by_number, vu.sort_by_number]
series_must = ["Res4", "", "Res4", ""] # leave "" per default
series_mustnt = ["", "", "", ""] # leave "" per default
series_column = [1, 1, 1, 1]

# Scattering plot options
plot_title = "Scattering for Au spheres of different diameters and material source"
series_colors = [plab.cm.Blues, plab.cm.Reds, plab.cm.Purples, plab.cm.Greens]
subtitles_label = [lambda s : f"Diameter {vu.find_numbers(s)[0]} nm",
                   lambda s : f"Diameter {vu.find_numbers(s)[0]} nm",
                   lambda s : f"Diameter {vu.find_numbers(s)[0]} nm",
                   lambda s : f"Diameter {vu.find_numbers(s)[0]} nm"]
series_label = ["Vacuum R Meep", "Vacuum JC Meep", "Water R Meep", "Water JC Meep"]
series_linestyles = ["solid", "solid", "solid", "solid"]

theory_label = ["Vacuum R Theory", "Vacuum JC Theory", "Water R Theory", "Water JC Theory"]
theory_linestyles = ["dashed", "dashed", "dashed", "dashed"]

plot_make_big = True
plot_file = lambda n : os.path.join(home, "DataAnalysis", "PaperDiameters"+n)

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
# index = []
for p in params:
    r.append( [pi["r"] for pi in p] )
    from_um_factor.append( [pi["from_um_factor"] for pi in p] )
    resolution.append( [pi["resolution"] for pi in p] )
    try:
        paper.append( [pi["paper"] for pi in p])
    except:
        paper.append( ["R" for pi in p] )
    # index.append( [] )
    # for p in params[-1]:
    #     try:
    #         index[-1].append( p["submerged_index"] )
    #     except KeyError:
    #         index[-1].append( 1.333 )
index = [ [*[1]*4] , [*[1]*4], [*[1.33]*4], [*[1.33]*4]]

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

#%% LOAD MARIAN'S DATA

marian_path = os.path.join(home, marian_folder)
marian_file = lambda s : os.path.join(marian_path, s)

marian_series = os.listdir(marian_path)

marian_exp_series = vu.filter_by_string_must(marian_series, "exp")
marian_exp_series = vu.filter_by_string_must(marian_exp_series, "glass")
marian_exp_series = vu.sort_by_number(marian_exp_series)

marian_mie_series = vu.filter_by_string_must(marian_series, "mie")
marian_mie_series = vu.sort_by_number(marian_mie_series)

marian_exp_data = []
for s in marian_exp_series:
    marian_exp_data.append(np.loadtxt(marian_file(s)))
    
marian_mie_data = []
for s in marian_mie_series:
    marian_mie_data.append(np.loadtxt(marian_file(s)))

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
    
marian_exp_max = [marian_exp_data[i][np.argmax(marian_exp_data[i][:,1]), 0] for i in range(len(marian_exp_data))]
marian_mie_max = [marian_mie_data[i][np.argmax(marian_mie_data[i][:,1]), 0] for i in range(len(marian_mie_data))]

marian_wlen_diff = []
for md in max_wlen:
    marian_wlen_diff.append( [d - m for d,m in zip(md, marian_exp_max)] )

#%%

colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

fig = plt.figure()
for i in range(len(data)):
    for j in range(len(data[i])):
        if j==0: label=series_label[i]
        else: label=None
        plt.plot(max_wlen_diff[i][j], f"{2 * r[i][j] * from_um_factor[i][j] * 1e3:.0f} nm",
                 '*', markersize=12, color=colors[i][1], label=label)

plt.grid(True, axis="x", which="minor")
plt.grid(True, axis="x", which="major")
plt.legend(loc="lower right")
plt.xlabel(r"Wavelength difference $\lambda_{MEEP} - \lambda_{MIE}$ [nm]")
plt.title(plot_title)
ax = fig.axes[0]
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.show()
fig.set_size_inches([6.4 , 4.07])
plt.tight_layout()

vs.saveplot(plot_file("WlenDif.png"), overwrite=True)

#%%

colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

fig = plt.figure()
for i in range(len(data)):
    for j in range(len(data[i])):
        if j==0: label=series_label[i]
        else: label=None
        plt.plot(marian_wlen_diff[i][j], f"{2 * r[i][j] * from_um_factor[i][j] * 1e3:.0f} nm",
                 '*', markersize=12, color=colors[i][1], label=label)

plt.grid(True, axis="x", which="minor")
plt.grid(True, axis="x", which="major")
plt.legend(loc="upper center")
plt.xlabel(r"Wavelength difference $\lambda_{MEEP} - \lambda_{EXP}$ [nm]")
plt.title(plot_title)
ax = fig.axes[0]
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.show()
fig.set_size_inches([6.4 , 4.07])
plt.tight_layout()

vs.saveplot(plot_file("WlenDifMarian.png"), overwrite=True)

#%% PLOT NORMALIZED VACUUM

colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
fig.suptitle(plot_title)

axes = axes.reshape(len(data))
# for ax, tit in zip(axes, plot_title):
    
for i in range(len(data)):
    for j in range(len(data[i])):
        axes[j].plot(data[i][j][:,0], data[i][j][:,1] / max(data[i][j][:,1]),
                     linestyle=series_linestyles[i], 
                      color=colors[i][1], 
                     label=series_label[i])
        axes[j].plot(data[i][j][:,0], theory[i][j] / max(theory[i][j]),
                     linestyle=theory_linestyles[i], 
                      color=colors[i][1], 
                     label=theory_label[i])
        if i==3:
            axes[j].plot(marian_exp_data[j][:,0], marian_exp_data[j][:,1],
                         "-k", label="Experimental Data")
        axes[j].set_title(subtitles_label[i](series[i][j]))
        axes[j].grid(True)
        if j==2 or j==3:
            axes[j].xaxis.set_label_text(r"Wavelength $\lambda$ [nm]")
        if j==0 or j==2:
            axes[j].yaxis.set_label_text("Normalized Scattering")
        axes[j].set_xlim(450, 650)
axes[0].legend(ncol=2)

if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
del mng
fig.tight_layout()
plt.subplots_adjust(wspace=0)

vs.saveplot(plot_file("AllScattNorm.png"), overwrite=True)
