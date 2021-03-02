#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 12:24:04 2020

@author: vall
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import v_save as vs

#%% PARAMETERS

# Saving directories
folder = "AuMieSphere/AuMie/Diameters/WLen4560"
home = vs.get_home()

# nprocesses = np.arange(8) + 1

#%% LOAD DATA

path = os.path.join(home, folder)
file = lambda f, s : os.path.join(path, f, s)

series = os.listdir(path)
series.sort()
series = [*series[1:4], series[0]]#, *series[4:]] # For AllWater

data = []
params = []
for s in series:
    data.append(np.loadtxt(file(s, "Results.txt")))
    params.append(vs.retrieve_footer(file(s, "Results.txt")))
header = vs.retrieve_header(file(s, "Results.txt"))

fixed_params = []
for p in params:
    problem = p.split("wlen_range=")[1].split(", nfreq")[0]
    solved = ", ".join(problem.split(" "))
    fixed = solved.join(p.split(problem))
    fixed_params.append(eval(f"dict({fixed})"))
params = fixed_params
del p, problem, solved, fixed, fixed_params

enlapsed = [p["enlapsed"] for p in params]
total = np.array([sum(e) for e in enlapsed])
enlapsed = np.array(enlapsed)

# series = series[:-2]
# total = total[:-2]
# enlapsed = enlapsed[:-2, :]

#%% PLOT

plt.figure()
for d in data:
    scatt = d[:,1]
    scatt = ( scatt - min(scatt) ) / (max(scatt) - min(scatt))
    plt.plot(d[:,0], scatt) #d[:,1])
plt.xlabel("Wavelength [nm]")
plt.ylabel("Rescaled Scattering Effiency [a.u.]")
vs.saveplot(file(s, "Scatt.png"))