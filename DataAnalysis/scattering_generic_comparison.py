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
folder = ["AuMieMediums/AllWaterTest"]
home = vs.get_home()

# Sorting and labelling data series
sorting_method = ["classic"] # "number2"
series_label = [lambda s : s] #lambda s : f"Second Time Factor x{vs.find_numbers(s)[2]}"
series_must = ["BackToVacuum"] # leave "" per default

# Scattering plot options
series_colors = [plab.cm.Blues]
series_linestyles = ["solid"]

#%% SORTING METHOD

def sorting_function_definer(sorting_method):
    
    if sorting_method=="classic":
        def sorting_function(l):
            l.sort()
            return l
    
    elif "number" in sorting_method:
        reference_index = vs.find_numbers(sorting_method)[0]
        def sorting_function(l):
            numbers = [vs.find_numbers(s)[reference_index] for s in l]
            index = np.argsort(numbers)
            return [l[i] for i in index]
    
    else:
        raise ValueError("Define your own function instead of using this :P")
    
    return sorting_function

sorting_function = [sorting_function_definer(sm) for sm in sorting_method]

# sorting_function = lambda l : [*l[-4:-1], l[-5]]

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
    to_remove_items = []
    for s in series[-1]: 
        if sm not in s: to_remove_items.append(s)
    for s in to_remove_items: series[-1].remove(s)
    del to_remove_items
    series[-1] = sf(series[-1])
    
    data.append( [] )
    params.append( [] )
    for s in series[-1]:
        data[-1].append(np.loadtxt(file[-1](s, "Results.txt")))
        params[-1].append(vs.retrieve_footer(file[-1](s, "Results.txt")))
    header.append( vs.retrieve_header(file[-1](s, "Results.txt")) )
    
    fixed_params = []
    for p in params[-1]:
        problem = p.split("wlen_range=")[1].split(", nfreq")[0]
        solved = ", ".join(problem.split(" "))
        fixed = solved.join(p.split(problem))
        fixed_params.append(eval(f"dict({fixed})"))
    params[-1] = fixed_params
    del p, problem, solved, fixed, fixed_params
    
    # r = [p["r"] for p in params]
    # from_um_factor = [p["from_um_factor"] for p in params]

#%% GET MAX WAVELENGTH

max_wlen = []
for d in data:
    max_wlen.append( [d[i][np.argmax(d[i][:,1]), 0] for i in range(len(d))] )

#%% PLOT

colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

plt.figure()
for s, d, p, sl, c, ls in zip(series, data, params, series_label, 
                              colors, series_linestyles):

    for ss, sd, sp, sc in zip(s, d, p, c):
        plt.plot(sd[:,0], sd[:,1] / max(sd[:,1]), 
                 linestyle=ls, color=sc, label=sl(ss))

plt.xlabel("Wavelength [nm]")
plt.ylabel("Normalized Scattering Cross Section")
plt.legend()
vs.saveplot(file("", "AllScatt.png"), overwrite=True)