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
import v_save as vs

#%% PARAMETERS

# Saving directories
folder = ["AuMieSphere/AuMie/10)MaxRes/Max103Res",
          "AuMieSphere/AuMie/10)MaxRes/Max103Res"]
home = vs.get_home()

# Sorting and labelling data series
sorting_function = [lambda l : vs.sort_by_number(l, -1), 
                    lambda l : vs.sort_by_number(l, -1)]
def special_label(s):
    if "5" in s:
        return "Mie"
    else:
        return ""
series_label = [lambda s : f"Meep Resolution {vs.find_numbers(s)[-1]}",
                special_label]
series_must = ["", ""] # leave "" per default
series_column = [1, 2]

# Scattering plot options
plot_title = "Scattering for Au spheres in vacuum with 103 nm diameter"
series_colors = [plab.cm.Reds, plab.cm.Reds]
series_linestyles = ["solid", "dashed"]
plot_make_big = True
plot_file = lambda n : os.path.join(home, "DataAnalysis/Max103Res" + n)

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
    series[-1] = vs.filter_by_string_must(series[-1], sm)
    series[-1] = sf(series[-1])
    
    data.append( [] )
    params.append( [] )
    for s in series[-1]:
        data[-1].append(np.loadtxt(file[-1](s, "Results.txt")))
        params[-1].append(vs.retrieve_footer(file[-1](s, "Results.txt")))
    header.append( vs.retrieve_header(file[-1](s, "Results.txt")) )
    
    for i in range(len(params[-1])):
        if not isinstance(params[-1][i], dict): 
            params[-1][i] = vs.fix_params_dict(params[-1][i])
    
    # r = [p["r"] for p in params]
    # from_um_factor = [p["from_um_factor"] for p in params]

#%% GET MAX WAVELENGTH

max_wlen = []
for d, sc in zip(data, series_column):
    max_wlen.append( [d[i][np.argmax(d[i][:,sc]), 0] for i in range(len(d))] )

dif_max_wlen = [ml - max_wlen[1][0] for ml in max_wlen[0]]

resolution = [p["resolution"] for p in params[0]]

plt.figure()
plt.title("Difference in scattering maximum's wavelength for Au 103 nm sphere")
plt.plot(resolution, dif_max_wlen, 'o-')
plt.xlabel("Resolution")
plt.ylabel("Difference in wavelength $\lambda_{max}^{MEEP}-\lambda_{max}^{MIE}$")
vs.saveplot(plot_file("WLenDiff.png"), overwrite=True)

#%% PLOT

colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

plt.figure()
plt.title(plot_title)
for s, d, p, sc, psl, pc, pls in zip(series, data, params, series_column, 
                                     series_label, colors, series_linestyles):

    for ss, sd, sp, spc in zip(s, d, p, pc):
        plt.plot(sd[:,0], sd[:,sc] / max(sd[:,sc]), 
                 linestyle=pls, color=spc, label=psl(ss))

plt.xlabel("Wavelength [nm]")
plt.ylabel("Normalized Scattering Cross Section")
plt.legend()
if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
del mng
vs.saveplot(plot_file("AllScatt.png"), overwrite=True)