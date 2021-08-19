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
folder = "AuMieSphere/AuMie/8)TestParNP"
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

elapsed = [p["elapsed"] for p in params]
total = np.array([sum(e) for e in elapsed])
elapsed = np.array(elapsed)

#%% PLOT

plt.plot(nprocesses, total)