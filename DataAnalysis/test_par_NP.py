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
folder = "AuMieSphere/AuMie/8)TestParNP/2nd"
home = vs.get_home()

nprocesses = np.arange(8) + 1

#%% LOAD DATA

path = os.path.join(home, folder)
file = lambda f, s : os.path.join(path, f, s)

series = os.listdir(path)
series.sort()

params = []
for s in series:
    params.append(vs.retrieve_footer(file(s, "Results.txt")))

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

#%% PLOT

plt.figure()
plt.plot(nprocesses, total_0, 'ro-', label="Serie 1")
plt.plot(nprocesses, total, 'bo-', label="Serie 2")
plt.xlabel("Número de subprocesos")
plt.ylabel("Tiempo total (s)")
plt.legend(["Serie 1", "Serie 2"])
plt.savefig(os.path.join(path, "Total.png"), bbox_inches='tight')

#%% 

plt.figure()
# plt.plot(nprocesses, enlapsed_0[:,1], 's-r', label="Serie 1: Sim I")
# plt.plot(nprocesses, enlapsed_0[:,-1], 'o-r', label="Serie 1: Sim II")
plt.plot(nprocesses, enlapsed[:,1], 's-b', label="Serie 2: Sim I")
plt.plot(nprocesses, enlapsed[:,-1], 'o-b', label="Serie 2: Sim II")
plt.xlabel("Número de subprocesos")
plt.ylabel("Tiempo (s)")
plt.legend()
plt.savefig(os.path.join(path, "Simulations.png"), bbox_inches='tight')

#%%

plt.figure()
# plt.plot(nprocesses, enlapsed_0[:,0], 's-r', label="Serie 1: Init I")
plt.plot(nprocesses, enlapsed[:,0], 's-b', label="Serie 2: Init I")
# plt.plot(nprocesses, enlapsed_0[:,2], 'o-r', label="Serie 1: Init II")
plt.plot(nprocesses, enlapsed[:,2], 'o-b', label="Serie 2: Init II")
plt.xlabel("Número de subprocesos")
plt.ylabel("Tiempo (s)")
plt.legend()
plt.savefig(os.path.join(path, "Building.png"), bbox_inches='tight')

#%%

plt.figure()
# plt.plot(nprocesses, enlapsed_0[:,3], 'o-r', label="Serie 1: Load Flux")
plt.plot(nprocesses, enlapsed[:,3], 'o-b', label="Serie 2: Load Flux")
plt.xlabel("Número de subprocesos")
plt.ylabel("Tiempo (s)")
plt.legend()
plt.savefig(os.path.join(path, "LoadFlux.png"), bbox_inches='tight')

#%%

