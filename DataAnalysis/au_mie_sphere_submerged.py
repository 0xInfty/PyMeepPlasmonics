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
vacuum_folder = "AuMieSphere/AuMie/7)Diameters/WLen4560"
water_folder = "AuMieMediums/AllWater"
glassnwater_folder = "AuMieMediums/GlassNWater"
marian_folder = "AuMieSphere/AuMie/7)Diameters/Marians"

home = vs.get_home()

#%% LOAD VACUUM DATA

vacuum_path = os.path.join(home, vacuum_folder)
vacuum_file = lambda f, s : os.path.join(vacuum_path, f, s)

vacuum_series = os.listdir(vacuum_path)
vacuum_series = vu.filter_by_string_must(vacuum_series, "SC")
vacuum_series = vu.sort_by_number(vacuum_series, 0)

vacuum_data = []
vacuum_params = []
for s in vacuum_series:
    vacuum_data.append(np.loadtxt(vacuum_file(s, "Results.txt")))
    vacuum_params.append(vs.retrieve_footer(vacuum_file(s, "Results.txt")))
vacuum_header = vs.retrieve_header(vacuum_file(s, "Results.txt"))

vacuum_params = [vu.fix_params_dict(p) for p in vacuum_params]

#%% LOAD WATER DATA

water_path = os.path.join(home, water_folder)
water_file = lambda f, s : os.path.join(water_path, f, s)

water_series = os.listdir(water_path)
water_series = vu.filter_by_string_must(water_series, "AllWater")
water_series = vu.sort_by_number(water_series, 0)

water_data = []
water_params = []
for s in water_series:
    water_data.append(np.loadtxt(water_file(s, "Results.txt")))
    water_params.append(vs.retrieve_footer(water_file(s, "Results.txt")))
water_header = vs.retrieve_header(water_file(s, "Results.txt"))

water_params = [vu.fix_params_dict(p) for p in water_params]

#%% LOAD GLASS N WATER DATA

glassnwater_path = os.path.join(home, glassnwater_folder)
glassnwater_file = lambda f, s : os.path.join(glassnwater_path, f, s)

glassnwater_series = os.listdir(glassnwater_path)
glassnwater_series = vu.sort_by_number(glassnwater_series, 0)

glassnwater_data = []
glassnwater_params = []
for s in glassnwater_series:
    glassnwater_data.append(np.loadtxt(glassnwater_file(s, "Results.txt")))
    glassnwater_params.append(vs.retrieve_footer(glassnwater_file(s, "Results.txt")))
glassnwater_header = vs.retrieve_header(glassnwater_file(s, "Results.txt"))

glassnwater_params = [vu.fix_params_dict(p) for p in glassnwater_params]

#%% LOAD MARIAN'S DATA

marian_path = os.path.join(home, marian_folder)
marian_file = lambda s : os.path.join(marian_path, s)

marian_series = os.listdir(marian_path)

marian_exp_series = vu.filter_by_string_must(marian_series, "exp")
marian_exp_series = vu.filter_by_string_must(marian_exp_series, "glass")
marian_exp_series = vu.sort_by_number(marian_exp_series)

marian_mie_series = vu.filter_by_string_must(marian_series, "mie")
marian_mie_series = vu.sort_by_number(marian_mie_series)

marian_exp_data = []
for s in marian_exp_series:
    marian_exp_data.append(np.loadtxt(marian_file(s)))
    
marian_mie_data = []
for s in marian_mie_series:
    marian_mie_data.append(np.loadtxt(marian_file(s)))

#%% GET MAX WAVELENGTH

vacuum_max = [vacuum_data[i][np.argmax(vacuum_data[i][:,1]), 0] for i in range(len(vacuum_data))]
water_max = [water_data[i][np.argmax(water_data[i][:,1]), 0] for i in range(len(water_data))]
glassnwater_max = [glassnwater_data[i][np.argmax(glassnwater_data[i][:,1]), 0] for i in range(len(glassnwater_data))]

marian_exp_max = [marian_exp_data[i][np.argmax(marian_exp_data[i][:,1]), 0] for i in range(len(marian_exp_data))]
marian_mie_max = [marian_mie_data[i][np.argmax(marian_mie_data[i][:,1]), 0] for i in range(len(marian_mie_data))]

#%% PLOT

diameters = [48, 64, 80, 103]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

plt.figure()
for vdt, d, c in zip(vacuum_data, diameters, colors):
    plt.plot(vdt[:,0], vdt[:,1] / max(vdt[:,1]), #* np.pi * (d**2) / 4
             linestyle='solid', color=c, label=f"Meep Vacío {d} nm")
for wdt, d, c in zip(water_data, diameters, colors):
    plt.plot( wdt[:,0], wdt[:,1] / max(wdt[:,1]), 
             linestyle='dashdot', color=c, label=f"Meep Agua {d} nm")
plt.style.use('default')
for mdt, d, c in zip(marian_exp_data, diameters, colors):
    plt.plot(mdt[:,0], mdt[:,1] / max(mdt[:,1]), 
             linestyle='dashed', color=c, label=f"Marian Exp {d} nm")
plt.style.use('default')
for mdt, d, c in zip(marian_mie_data, diameters, colors):
    plt.plot(mdt[:,0], mdt[:,1] / max(mdt[:,1]), 
             linestyle="dotted", color=c, label=f"Marian Mie {d} nm")
plt.xlabel("Wavelength [nm]")
plt.ylabel("Normalized Scattering [a.u.]")
plt.xlim([450, 650])
plt.legend(ncol=4, framealpha=1)
vs.saveplot(water_file("", "AllScatt.png"), overwrite=True)

#%% PLOT 103 nm

colors = (c for c in plab.cm.Reds(np.linspace(0,1,2+5))[2:])

plt.figure()
plt.title("Scattering of Au sphere (103 nm diameter)")
scatt = vacuum_data[-1][:,1] #* np.pi * (diameters[-1]**2) / 4
plt.plot(vacuum_data[-1][:,0], scatt/max(scatt),
         linestyle="solid", color=next(colors), label="Vacuum")
scatt = water_data[-1][:,1] #* np.pi * (diameters[-1]**2) / 4
plt.plot(water_data[-1][:,0], scatt/max(scatt),
         linestyle="dashdot", color=next(colors), label="Water")
scatt = glassnwater_data[-1][:,1] #* np.pi * (diameters[-1]**2) / 4
plt.plot(glassnwater_data[0][:,0], scatt/max(scatt),
         linestyle="dashed", color=next(colors), label="Glass+Water")
plt.plot(marian_exp_data[-1][:,0], 
         marian_exp_data[-1][:,1] / max(marian_exp_data[-1][:,1]), 
         linestyle="dotted", color=next(colors), label="Experimental")
plt.plot(marian_mie_data[-1][:,0], 
         marian_mie_data[-1][:,1] / max(marian_mie_data[-1][:,1]), 
         linestyle="dotted", color=next(colors), label="Mie Theory Water")
plt.xlabel("Wavelength [nm]")
plt.ylabel("Rescaled Scattering [a.u.]")
plt.xlim([450, 650])
plt.legend(framealpha=1)
vs.saveplot(water_file("", "Scatt103.png"), overwrite=True)
