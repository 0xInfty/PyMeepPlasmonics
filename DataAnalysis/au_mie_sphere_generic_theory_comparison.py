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
from v_meep import import_medium
import v_save as vs
import v_utilities as vu

#%% PARAMETERS

# Saving directories
folder = ["AuMieSphere/AuMie/7)Diameters/WLen4560"]
home = vs.get_home()

# Sorting and labelling data series
sorting_function = [vu.sort_by_number]
series_must = ["SC"] # leave "" per default
series_mustnt = [""] # leave "" per default
series_column = [1]

# Scattering plot options
plot_title = "Scattering for generic Au spheres series"
series_colors = [plab.cm.Reds]
series_label = [lambda s : f"Meep {vu.find_numbers(s)[0]} nm"]
series_linestyles = ["solid"]
theory_label = lambda s : f"Theory {vu.find_numbers(s)[0]} nm"
theory_linestyle = "dashed"
plot_make_big = True
plot_file = lambda n : os.path.join(home, "DataAnalysis", n)

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
index = []
for p in params:
    r.append( [pi["r"] for pi in p] )
    from_um_factor.append( [pi["from_um_factor"] for pi in p] )
    resolution.append( [pi["resolution"] for pi in p] )
    index.append( [] )
    for p in params[-1]:
        try:
            index[-1].append( p["index"] )
        except KeyError:
            index[-1].append( 1 )

#%% GET THEORY

theory = [] # Scattering effiency
for di, ri, fi, resi, ii in zip(data, r, from_um_factor, resolution, index):
    theory.append([])    
    for dj, rj, fj, resj, ij in zip(di, ri, fi, resi, ii):
        wlenj = dj[:,0] # nm
        freqj = 1 / wlenj # 1/nm
        freqmeepj = (1e3 * fj) / wlenj # Meep units
        # wlenj = 1e3*from_um_factor/freqmeepj
        mediumj = import_medium("Au", fj)
        # wlenj = 1e3*from_um_factor/freqmeepj 
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

#%% PLOT

colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

plt.figure()
plt.title(plot_title)
for s, d, t, p, sc, psl, pc, pls in zip(series, data, theory, params, series_column, 
                                        series_label, colors, series_linestyles):

    for ss, sd, td, sp, spc in zip(s, d, t, p, pc):
        plt.plot(sd[:,0], sd[:,sc] / max(sd[:,sc]), 
                 linestyle=pls, color=spc, label=psl(ss))
        plt.plot(sd[:,0], td / max(td),
                 linestyle=theory_linestyle, color=spc, label=theory_label(ss))

plt.xlabel("Wavelength [nm]")
plt.ylabel("Normalized Scattering Cross Section")
plt.legend()
if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
del mng
vs.saveplot(plot_file("AllScatt.png"), overwrite=True)