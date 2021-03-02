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
vacuum_folder = "AuMieSphere/AuMie/7)Diameters/WLen5065"
water_folder = "AuMieMediums/AllWater"
marian_folder = "AuMieSphere/AuMie/7)Diameters/Marians"
home = vs.get_home()

# nprocesses = np.arange(8) + 1

#%% LOAD VACUUM DATA

vacuum_path = os.path.join(home, vacuum_folder)
vacuum_file = lambda f, s : os.path.join(vacuum_path, f, s)

vacuum_series = os.listdir(vacuum_path)
vacuum_series.sort()
vacuum_series = [*vacuum_series[1:4], vacuum_series[0]]#, *series[4:]] # For AllWater

vacuum_data = []
vacuum_params = []
for s in vacuum_series:
    vacuum_data.append(np.loadtxt(vacuum_file(s, "Results.txt")))
    vacuum_params.append(vs.retrieve_footer(vacuum_file(s, "Results.txt")))
vacuum_header = vs.retrieve_header(vacuum_file(s, "Results.txt"))

fixed_params = []
for p in vacuum_params:
    problem = p.split("wlen_range=")[1].split(", nfreq")[0]
    solved = ", ".join(problem.split(" "))
    fixed = solved.join(p.split(problem))
    fixed_params.append(eval(f"dict({fixed})"))
params = fixed_params
del p, problem, solved, fixed, fixed_params

#%% LOAD WATER DATA

water_path = os.path.join(home, water_folder)
water_file = lambda f, s : os.path.join(water_path, f, s)

water_series = os.listdir(water_path)
water_series.sort()
water_series = [*water_series[1:4], water_series[0]]#, *series[4:]] # For AllWater

water_data = []
water_params = []
for s in water_series:
    water_data.append(np.loadtxt(water_file(s, "Results.txt")))
    water_params.append(vs.retrieve_footer(water_file(s, "Results.txt")))
water_header = vs.retrieve_header(water_file(s, "Results.txt"))

fixed_params = []
for p in water_params:
    problem = p.split("wlen_range=")[1].split(", nfreq")[0]
    solved = ", ".join(problem.split(" "))
    fixed = solved.join(p.split(problem))
    fixed_params.append(eval(f"dict({fixed})"))
params = fixed_params
del p, problem, solved, fixed, fixed_params

#%% LOAD MARIAN'S DATA

marian_path = os.path.join(home, marian_folder)
marian_file = lambda s : os.path.join(marian_path, s)

marian_series = os.listdir(marian_path)
marian_series.sort()
marian_exp_series = marian_series[1:5]#marian_series[-4:]
marian_exp_series = [*marian_exp_series[1:], marian_exp_series[0]]
marian_mie_series = marian_series[-4:]
marian_mie_series = [*marian_mie_series[1:], marian_mie_series[0]]

marian_exp_data = []
for s in marian_exp_series:
    marian_exp_data.append(np.loadtxt(marian_file(s)))
    
marian_mie_data = []
for s in marian_mie_series:
    marian_mie_data.append(np.loadtxt(marian_file(s)))

#%% PLOT

diameters = [48, 64, 80, 103]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

plt.figure()
for vdt, d, c in zip(vacuum_data, diameters, colors):
    scatt = vdt[:,1]
    scatt = ( scatt - min(scatt) ) / (max(scatt) - min(scatt))
    plt.plot(vdt[:,0], scatt, linestyle='solid', color=c, label=f"Meep Vacío {d} nm") #d[:,1])
    # plt.plot(dt[:,0], dt[:,1], label=f"Meep {d} nm")
for wdt, d, c in zip(water_data, diameters, colors):
    scatt = wdt[:,1]
    scatt = ( scatt - min(scatt) ) / (max(scatt) - min(scatt))
    plt.plot( wdt[:,0], scatt, linestyle='dashdot', color=c, label=f"Meep Agua {d} nm") #d[:,1])
    # plt.plot(dt[:,0], dt[:,1], label=f"Meep {d} nm")
plt.style.use('default')
for mdt, d, c in zip(marian_exp_data, diameters, colors):
    plt.plot(mdt[:,0], mdt[:,1], linestyle='dashed', color=c, label=f"Exp {d} nm")
plt.style.use('default')
for mdt, d, c in zip(marian_mie_data, diameters, colors):
    plt.plot(mdt[:,0], mdt[:,1], linestyle="dotted", color=c, label=f"Mie {d} nm")
plt.xlabel("Wavelength [nm]")
plt.ylabel("Rescaled Scattering [a.u.]")
plt.xlim([500, 650])
plt.legend(loc="lower center", ncol=4, framealpha=1)
vs.saveplot(water_file("", "AllScatt.png"))