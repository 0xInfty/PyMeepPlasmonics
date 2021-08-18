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
import v_utilities as vu

#%% PARAMETERS

# Saving directories
folder = "AuMieMediums/AllWaterTest/TestSecTime/Test103Res2SecTime"
home = vs.get_home()

# Sorting and labelling data series
sorting_function = lambda l : vu.sort_by_number(l, 2)
series_label = lambda s : f"Second Time Factor x{vu.find_numbers(s)[2]}"
recognize_string = "Test" # leave "" per default

#%% LOAD DATA

path = os.path.join(home, folder)
file = lambda f, s : os.path.join(path, f, s)

series = os.listdir(path)
series = vu.filter_by_string_must(series, recognize_string)
series = sorting_function(series)

data = []
params = []
for s in series:
    data.append(np.loadtxt(file(s, "Results.txt")))
    params.append(vs.retrieve_footer(file(s, "Results.txt")))
header  = vs.retrieve_header(file(s, "Results.txt"))

params = [vu.fix_params_dict(p) for p in params]

r = [p["r"] for p in params]
from_um_factor = [p["from_um_factor"] for p in params]

#%% GET MAX WAVELENGTH

max_wlen = [data[i][np.argmax(data[i][:,1]), 0] for i in range(len(data))]

#%% PLOT

colors = plab.cm.Blues(np.linspace(0,1,len(series)+3))[3:]

plt.figure()
for s, d, c, p in zip(series, data, colors, params):
    scatt = d[:,1] * np.pi * (p["r"] * p["from_um_factor"]) **2
    scatt = scatt
    plt.plot(d[:,0], scatt, linestyle='solid', color=c, label=series_label(s))
plt.xlabel("Wavelength [nm]")
plt.ylabel("Scattering Cross Section [$\mu$m$^2$]")
plt.legend()
vs.saveplot(file("", "AllScatt.png"), overwrite=True)