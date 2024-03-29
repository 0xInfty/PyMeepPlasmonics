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
import v_utilities as vu

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

params = [vu.fix_params_dict(p) for p in params]

elapsed = [p["elapsed"] for p in params]
total = np.array([sum(e) for e in elapsed])
elapsed = np.array(elapsed)

#%% PLOT

plt.figure()
# plt.plot(nprocesses, total_0, 'ro-', label="Serie 1")
plt.plot(nprocesses, total, 'bo-', label="Serie 2")
plt.xlabel("Número de subprocesos")
plt.ylabel("Tiempo total (s)")
plt.legend(["Serie 1", "Serie 2"])
plt.savefig(os.path.join(path, "Total.png"), bbox_inches='tight')

#%% 

plt.figure()
# plt.plot(nprocesses, elapsed_0[:,1], 's-r', label="Serie 1: Sim I")
# plt.plot(nprocesses, elapsed_0[:,-1], 'o-r', label="Serie 1: Sim II")
plt.plot(nprocesses, elapsed[:,1], 's-b', label="Serie 2: Sim I")
plt.plot(nprocesses, elapsed[:,-1], 'o-b', label="Serie 2: Sim II")
plt.xlabel("Número de subprocesos")
plt.ylabel("Tiempo (s)")
plt.legend()
plt.savefig(os.path.join(path, "Simulations.png"), bbox_inches='tight')

#%%

plt.figure()
# plt.plot(nprocesses, elapsed_0[:,0], 's-r', label="Serie 1: Init I")
plt.plot(nprocesses, elapsed[:,0], 's-b', label="Serie 2: Init I")
# plt.plot(nprocesses, elapsed_0[:,2], 'o-r', label="Serie 1: Init II")
plt.plot(nprocesses, elapsed[:,2], 'o-b', label="Serie 2: Init II")
plt.xlabel("Número de subprocesos")
plt.ylabel("Tiempo (s)")
plt.legend()
plt.savefig(os.path.join(path, "Building.png"), bbox_inches='tight')

#%%

plt.figure()
# plt.plot(nprocesses, elapsed_0[:,3], 'o-r', label="Serie 1: Load Flux")
plt.plot(nprocesses, elapsed[:,3], 'o-b', label="Serie 2: Load Flux")
plt.xlabel("Número de subprocesos")
plt.ylabel("Tiempo (s)")
plt.legend()
plt.savefig(os.path.join(path, "LoadFlux.png"), bbox_inches='tight')

#%%

