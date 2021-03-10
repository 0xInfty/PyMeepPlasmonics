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
folder = "AuMieMediums/AllWaterTest"
home = vs.get_home()

# Sorting and labelling data series
sorting_method = "classic"#"number2"
series_label = lambda s : "" #lambda s : f"Second Time Factor x{vs.find_numbers(s)[2]}"
recognize_string = "BackToVacuum" # leave "" per default

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

sorting_function = sorting_function_definer(sorting_method)

# sorting_function = lambda l : [*l[-4:-1], l[-5]]

#%% LOAD DATA

path = os.path.join(home, folder)
file = lambda f, s : os.path.join(path, f, s)

series = os.listdir(path)
to_remove_items = []
for s in series: 
    if recognize_string not in s: to_remove_items.append(s)
for s in to_remove_items: series.remove(s)
del to_remove_items
series = sorting_function(series)

data = []
params = []
for s in series:
    data.append(np.loadtxt(file(s, "Results.txt")))
    params.append(vs.retrieve_footer(file(s, "Results.txt")))
header  = vs.retrieve_header(file(s, "Results.txt"))

fixed_params = []
for p in params:
    problem = p.split("wlen_range=")[1].split(", nfreq")[0]
    solved = ", ".join(problem.split(" "))
    fixed = solved.join(p.split(problem))
    fixed_params.append(eval(f"dict({fixed})"))
params = fixed_params
del p, problem, solved, fixed, fixed_params

r = [p["r"] for p in params]
from_um_factor = [p["from_um_factor"] for p in params]

#%% GET MAX WAVELENGTH

max_wlen = [data[i][np.argmax(data[i][:,1]), 0] for i in range(len(data))]

#%% PLOT

colors = plab.cm.Blues(np.linspace(0,1,len(series)+3))[3:]

plt.figure()
for s, d, c, p in zip(series, data, colors, params):
    plt.plot(d[:,0], d[:,1] / max(d[:,1]), linestyle='solid', color=c, label=series_label(s))
plt.xlabel("Wavelength [nm]")
plt.ylabel("Normalized Scattering Cross Section")
plt.legend()
vs.saveplot(file("", "AllScatt.png"), overwrite=True)