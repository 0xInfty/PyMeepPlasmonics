#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 12:24:04 2020

@author: vall
"""

import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.pylab as plab
import os
import PyMieScatt as ps
import v_materials as vmt
import v_theory as vt
import v_save as vs
import v_utilities as vu

#%% PARAMETERS

# Saving directories
vacuum_folder = "Scattering/AuSphere/AllVacTest/7)Diameters/WLen4560"
water_folder = "Scattering/AuSphere/AllWatDiam"
glassnwater_folder = "Scattering/AuSphere/GlassNWater"
marian_folder = "Scattering/AuSphere/AllVacTest/7)Diameters/Marians"

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
water_series = vu.filter_by_string_must(water_series, "More", False)
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

#%% LOAD PYMIESCATT RAKIC'S DATA IN WATER

diameters = [48, 64, 80, 103]
from_um_factor = [wp["from_um_factor"] for wp in water_params]
index = 1.333

pms_r_mie_data = []
for wd, d, fuf in zip(marian_mie_data, diameters, from_um_factor):
    wlens = wd[:,0]
    freqs = 1e3*fuf/wlens
    medium = vmt.import_medium("Au", fuf)
    scatt_eff_theory = [ps.MieQ(np.sqrt(medium.epsilon(f)[0,0]*medium.mu(f)[0,0]), 
                                1e3*fuf/f,
                                d,
                                nMedium=index,
                                asDict=True)['Qsca'] 
                        for f in freqs]
    scatt_eff_theory = np.array(scatt_eff_theory)
    pms_r_mie_data.append(np.array([wlens, scatt_eff_theory]).T)
    
#%% LOAD PYMIESCATT JOHNSON AND CHRISTY'S MEEP DATA IN WATER

pms_jc_meep_mie_data = []
for wd, d, fuf in zip(marian_mie_data, diameters, from_um_factor):
    wlens = np.linspace(400, 800, len(wd[:,0]))
    epsilon_function = vmt.epsilon_function_from_meep(paper="JC")
    medium = vmt.import_medium("Au", fuf, paper="JC")
    scatt_eff_theory = [ps.MieQ(np.sqrt(epsilon_function(wl)), 
                                wl,
                                d,
                                nMedium=index,
                                asDict=True)['Qsca'] 
                        for wl in wlens]
    scatt_eff_theory = np.array(scatt_eff_theory)
    pms_jc_meep_mie_data.append(np.array([wlens, scatt_eff_theory]).T)
    
#%% LOAD PYMIESCATT JOHNSON AND CHRISTY'S RINFO DATA IN WATER

epsilon_function = vmt.epsilon_function_from_file(paper="JC")

pms_jc_rinfo_mie_data = []
for wd, d, fuf in zip(marian_mie_data, diameters, from_um_factor):
    wlens = wd[:,0]
    scatt_eff_theory = [ps.MieQ(np.sqrt(epsilon_function(wl)), 
                                wl,
                                d,
                                nMedium=index,
                                asDict=True)['Qsca'] 
                        for wl in wlens]
    scatt_eff_theory = np.array(scatt_eff_theory)
    pms_jc_rinfo_mie_data.append(np.array([wlens, scatt_eff_theory]).T)

#%% GET MAX WAVELENGTH

vacuum_max = [vacuum_data[i][np.argmax(vacuum_data[i][:,1]), 0] for i in range(len(vacuum_data))]
water_max = [water_data[i][np.argmax(water_data[i][:,1]), 0] for i in range(len(water_data))]
glassnwater_max = [glassnwater_data[i][np.argmax(glassnwater_data[i][:,1]), 0] for i in range(len(glassnwater_data))]

marian_exp_max = [marian_exp_data[i][np.argmax(marian_exp_data[i][:,1]), 0] for i in range(len(marian_exp_data))]
marian_mie_max = [marian_mie_data[i][np.argmax(marian_mie_data[i][:,1]), 0] for i in range(len(marian_mie_data))]
pms_r_mie_max = [pms_r_mie_data[i][np.argmax(pms_r_mie_data[i][:,1]), 0] for i in range(len(pms_r_mie_data))]
pms_jc_meep_mie_max = [pms_jc_meep_mie_data[i][np.argmax(pms_jc_meep_mie_data[i][:,1]), 0] for i in range(len(pms_jc_meep_mie_data))]
pms_jc_rinfo_mie_max = [pms_jc_rinfo_mie_data[i][np.argmax(pms_jc_rinfo_mie_data[i][:,1]), 0] for i in range(len(pms_jc_rinfo_mie_data))]

#%% 1st PLOT ALL: VACUUM, WATER, MARIAN EXPERIMENTAL, MARIAN MIE
# This makes up the first figures I've made, the ones that confused me

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

#%% 1st: PLOT 103 nm ALL: VACUUM, WATER, MARIAN EXPERIMENTAL, MARIAN MIE
# This makes up the first figures I've made, the ones that confused me

colors = (c for c in plab.cm.Reds(np.linspace(0,1,2+5))[2:])

plt.figure()
plt.title("Scattering of Au sphere (103 nm diameter)")
scatt = vacuum_data[-1][:,1] #* np.pi * (diameters[-1]**2) / 4
plt.plot(vacuum_data[-1][:,0], scatt/max(scatt),
         linestyle="solid", color=next(colors), label="Vacuum")
scatt = water_data[-1][:,1] #* np.pi * (diameters[-1]**2) / 4
plt.plot(water_data[-1][:,0], scatt/max(scatt),
         linestyle="dashdot", color=next(colors), label="Water")
# scatt = glassnwater_data[-1][:,1] #* np.pi * (diameters[-1]**2) / 4
# plt.plot(glassnwater_data[0][:,0], scatt/max(scatt),
#          linestyle="dashed", color=next(colors), label="Glass+Water")
plt.plot(marian_exp_data[-1][:,0], 
         marian_exp_data[-1][:,1] / max(marian_exp_data[-1][:,1]), 
         linestyle="dotted", color=next(colors), label="Experimental")
plt.plot(marian_mie_data[-1][:,0], 
         marian_mie_data[-1][:,1] / max(marian_mie_data[-1][:,1]), 
         linestyle="dotted", color=next(colors), label="Marian Mie Water")
plt.xlabel("Wavelength [nm]")
plt.ylabel("Normalized Scattering [a.u.]")
plt.xlim([450, 650])
plt.legend(framealpha=1)
vs.saveplot(os.path.join(home, "DataAnalysis/AllWater103Comparison1st.png"), overwrite=True)

#%% 2nd: WATER VS MARIAN THEORY 103 nm
# This makes up the figure I used when posting on Meep's Github

plt.figure()
plt.title("Scattering of Au sphere with 103 nm diameter submerged in water")
scatt = water_data[-1][:,1] #* np.pi * (diameters[-1]**2) / 4
plt.plot(water_data[-1][:,0], scatt/max(scatt),
         linestyle="solid", color='#1f77b4', label="Meep")
scatt = glassnwater_data[-1][:,1] #* np.pi * (diameters[-1]**2) / 4
plt.plot(marian_mie_data[-1][:,0], 
         marian_mie_data[-1][:,1] / max(marian_mie_data[-1][:,1]), 
         linestyle="dashed", color='#1f77b4', label="Mie")
plt.xlabel("Wavelength [nm]")
plt.ylabel("Normalized Scattering Cross Section [a.u.]")
plt.xlim([500, 650])
plt.legend(framealpha=1)
plt.grid()
vs.saveplot(os.path.join(home, "DataAnalysis/AllWater103Comparison.png"), 
            overwrite=True)

#%% 3rd: WATER VS ALL WATER MIE THEORIES 1 DIAMETER

for d in diameters:
    index = np.argmin(np.abs(np.array(diameters)-d))
    
    fig = plt.figure()
    plt.title(f"Scattering of Au sphere with {d} nm diameter submerged in water")
    scatt = water_data[index][:,1] #* np.pi * (diameters[index]**2) / 4
    plt.plot(water_data[index][:,0], scatt/max(scatt),
             linestyle="solid", color='r', label="Meep's R Simulated Data")
    plt.plot(marian_mie_data[index][:,0], 
             marian_mie_data[index][:,1] / max(marian_mie_data[index][:,1]), 
             linestyle="solid", color='#e377c2', label="Marian's JC Mie")
    plt.plot(pms_jc_meep_mie_data[index][:,0], 
             pms_jc_meep_mie_data[index][:,1] / max(pms_jc_meep_mie_data[index][:,1]), 
             linestyle="dashed", color='#e377c2', label="PyMieScatt+Meep's JC Mie")
    plt.plot(pms_jc_rinfo_mie_data[index][:,0], 
             pms_jc_rinfo_mie_data[index][:,1] / max(pms_jc_rinfo_mie_data[index][:,1]), 
             linestyle="dotted", color='k', label="PyMieScatt+RI.info's JC Mie")
    plt.plot(pms_r_mie_data[index][:,0], 
             pms_r_mie_data[index][:,1] / max(pms_r_mie_data[index][:,1]), 
             linestyle="dashed", color='#1f77b4', label="PyMieScatt+Meep's R Mie")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Normalized Scattering Cross Section [a.u.]")
    plt.xlim([400, 800])
    plt.legend(framealpha=1)
    plt.grid()
    fig.set_size_inches([7.71, 4.8 ])
    vs.saveplot(os.path.join(home, f"DataAnalysis/AllWater{d}Comparison.png"), 
                overwrite=True)

#%% 3rd: WATER VS WATER MIE THEORIES 1 DIAMETER NOT NORMALIZED

for d in diameters:
    index = np.argmin(np.abs(np.array(diameters)-d))
    
    fig = plt.figure()
    plt.title(f"Scattering of Au sphere with {d} nm diameter submerged in water")
    scatt = water_data[index][:,1]
    plt.plot(water_data[index][:,0], scatt,
             linestyle="solid", color='r', label="Meep's R Simulated Data")
    plt.plot(pms_jc_meep_mie_data[index][:,0], 
             pms_jc_meep_mie_data[index][:,1], 
             linestyle="dashed", color='#e377c2', label="PyMieScatt+Meep's JC Mie")
    plt.plot(pms_jc_rinfo_mie_data[index][:,0], 
             pms_jc_rinfo_mie_data[index][:,1], 
             linestyle="dotted", color='k', label="PyMieScatt+RI.info's JC Mie")
    plt.plot(pms_r_mie_data[index][:,0], 
             pms_r_mie_data[index][:,1], 
             linestyle="dashed", color='#1f77b4', label="PyMieScatt+Meep's R Mie")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Scattering Effienciency [a.u.]")
    plt.xlim([400, 800])
    plt.legend(framealpha=1)
    plt.grid()
    fig.set_size_inches([7.71, 4.8 ])
    vs.saveplot(os.path.join(home, "DataAnalysis/AllWater103Comparison.png"), 
                overwrite=True)

# CANNOT PLOT MARIAN'S WITHOUT NORMALIZATION
# Marian's data is already normalized :(

#%% 3rd: ONE HUGE PLOT

fig = plt.figure()
plot_grid = gridspec.GridSpec(ncols=2, nrows=2, hspace=0.3, wspace=0.1, figure=fig)

plot_list = [plot_grid[0,0], plot_grid[0,1], plot_grid[1,0], plot_grid[1,1]]
axes_list = []
for pl in plot_list:
    axes_list.append(fig.add_subplot(pl))

for d, ax in zip(diameters, axes_list):
    index = np.argmin(np.abs(np.array(diameters)-d))
    ax.set_title(f"Diámetro {d} nm")
    ax.plot(water_data[index][:,0], 
            water_data[index][:,1]/max(water_data[index][:,1]),
             linestyle="solid", color='r', label="Meep's R Simulated Data")
    ax.plot(pms_jc_meep_mie_data[index][:,0], 
             pms_jc_meep_mie_data[index][:,1]/max(pms_jc_meep_mie_data[index][:,1]), 
             linestyle="dashed", color='#e377c2', label="PyMieScatt+Meep's JC Mie")
    ax.plot(pms_jc_rinfo_mie_data[index][:,0], 
             pms_jc_rinfo_mie_data[index][:,1]/max(pms_jc_rinfo_mie_data[index][:,1]), 
             linestyle="dotted", color='k', label="PyMieScatt+RI.info's JC Mie")
    ax.plot(pms_r_mie_data[index][:,0], 
             pms_r_mie_data[index][:,1]/max(pms_r_mie_data[index][:,1]), 
             linestyle="dashed", color='#1f77b4', label="PyMieScatt+Meep's R Mie")
    ax.xaxis.set_label_text("Wavelength [nm]")
    ax.yaxis.set_label_text("Scattering Effienciency [a.u.]")
    ax.set_xlim([400, 800])
    ax.grid()
    
fig.set_size_inches([19.2 ,  8.39])
plt.legend(framealpha=1)
vs.saveplot(os.path.join(home, "DataAnalysis/AllWaterFullComparison.png"), 
            overwrite=True)

#%% 3rd: ONE HUGE PLOT NOT NORMALIZED

fig = plt.figure()
plot_grid = gridspec.GridSpec(ncols=2, nrows=2, hspace=0.3, wspace=0.1, figure=fig)

plot_list = [plot_grid[0,0], plot_grid[0,1], plot_grid[1,0], plot_grid[1,1]]
axes_list = []
for pl in plot_list:
    axes_list.append(fig.add_subplot(pl))

for d, ax in zip(diameters, axes_list):
    i = np.argmin(np.abs(np.array(diameters)-d))
    ax.set_title(f"Diámetro {d} nm")
    ax.plot(water_data[i][:,0], 
            water_data[i][:,1],
             linestyle="solid", color='r', label="Meep's R Simulated Data")
    ax.plot(pms_jc_meep_mie_data[i][:,0], 
             pms_jc_meep_mie_data[i][:,1], 
             linestyle="dashed", color='#e377c2', label="PyMieScatt+Meep's JC Mie")
    ax.plot(pms_jc_rinfo_mie_data[i][:,0], 
             pms_jc_rinfo_mie_data[i][:,1], 
             linestyle="dotted", color='k', label="PyMieScatt+RI.info's JC Mie")
    ax.plot(pms_r_mie_data[i][:,0], 
             pms_r_mie_data[i][:,1], 
             linestyle="dashed", color='#1f77b4', label="PyMieScatt+Meep's R Mie")
    ax.xaxis.set_label_text("Wavelength [nm]")
    ax.yaxis.set_label_text("Scattering Effienciency [a.u.]")
    ax.set_xlim([400, 800])
    ax.grid()
    
fig.set_size_inches([19.2 ,  8.39])
plt.legend(framealpha=1)
vs.saveplot(os.path.join(home, "DataAnalysis/AllWaterFullComparisonEff.png"), 
            overwrite=True)