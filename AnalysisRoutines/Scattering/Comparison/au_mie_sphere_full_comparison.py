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
from matplotlib import use as use_backend
import os
import PyMieScatt as ps
import vmp_materials as vml
import v_save as vs
import v_utilities as vu
import v_plot as vp

vp.set_style()

#%% PARAMETERS

# Saving directories
vacuum_folder = "Scattering/AuSphere/AllVacTest/7)Diameters/WLen4560"
water_folder = "Scattering/AuSphere/AllWatDiam"
glassnwater_folder = "Scattering/AuSphere/GlassNWater"
marian_folder = "Scattering/AuSphere/AllVacTest/7)Diameters/Marians"

home = vs.get_home()

plot_folder = "DataAnalysis/Scattering/AuSphere/VacWatFullComparison"
plot_make_big = True
plot_for_display = True
plot_english = False

#%% PLOT CONFIGURATION

plot_file = lambda n : os.path.join(home, plot_folder, n)
if not os.path.isdir(plot_file("")): os.mkdir(plot_file(""))

trs = vu.BilingualManager(english=plot_english)

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
    medium = vml.import_medium("Au", from_um_factor=fuf)
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
    epsilon_function = vml.epsilon_function_from_meep(paper="JC")
    medium = vml.import_medium("Au", from_um_factor=fuf, paper="JC")
    scatt_eff_theory = [ps.MieQ(np.sqrt(epsilon_function(wl)), 
                                wl,
                                d,
                                nMedium=index,
                                asDict=True)['Qsca'] 
                        for wl in wlens]
    scatt_eff_theory = np.array(scatt_eff_theory)
    pms_jc_meep_mie_data.append(np.array([wlens, scatt_eff_theory]).T)
    
#%% LOAD PYMIESCATT JOHNSON AND CHRISTY'S RINFO DATA IN WATER

epsilon_function = vml.epsilon_function_from_file(paper="JC")

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

colors = ["C0", "C1", "C2", "C3"]

fig = plt.figure()
for vdt, d, c in zip(vacuum_data, diameters, colors):
    plt.plot(vdt[:,0], vdt[:,1] / max(vdt[:,1]), #* np.pi * (d**2) / 4
             linestyle='solid', color=c, 
             label=trs.choose(f"MEEP Vacuum {d} nm",f"MEEP Vacío {d} nm"))
for wdt, d, c in zip(water_data, diameters, colors):
    plt.plot( wdt[:,0], wdt[:,1] / max(wdt[:,1]), 
             linestyle='dashdot', color=c,
             label=trs.choose(f"MEEP Water {d} nm",f"MEEP Agua {d} nm"))
for mdt, d, c in zip(marian_exp_data, diameters, colors):
    plt.plot(mdt[:,0], mdt[:,1] / max(mdt[:,1]), 
             linestyle='dashed', color=c, label=f"Barella Experimental {d} nm")
for mdt, d, c in zip(marian_mie_data, diameters, colors):
    plt.plot(mdt[:,0], mdt[:,1] / max(mdt[:,1]), 
             linestyle="dotted", color=c,
             label=trs.choose(f"Barella Mie Water {d} nm",f"Barella Mie Agua {d} nm"))
plt.xlabel(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
plt.ylabel(trs.choose("Normalized Scattering Cross Section " + 
                      r"$\sigma_{scatt}/\sigma_{scatt}^{max}$",
                      "Sección eficaz de dispersión normalizada " +
                      r"$\sigma_{disp}/\sigma_{scatt}^{disp}$"))
plt.xlim([450, 650])
plt.legend(ncol=4)

if plot_make_big: fig.set_size_inches([19.2 ,  9.61])

vs.saveplot(plot_file("AllScatt.png"), overwrite=True)

#%% 1st: PLOT 103 nm ALL: VACUUM, WATER, MARIAN EXPERIMENTAL, MARIAN MIE
# This makes up the first figures I've made, the ones that confused me

colors = (c for c in plab.cm.Reds(np.linspace(0,1,2+5))[2:])

plt.figure()
plt.title(trs.choose("Scattering of Au sphere (103 nm diameter)",
                     "Dispersión de NP esférica de Au (103 nm de diámetro)"))
scatt = vacuum_data[-1][:,1] #* np.pi * (diameters[-1]**2) / 4
plt.plot(vacuum_data[-1][:,0], scatt/max(scatt),
         linestyle="solid", color=next(colors), 
         label=trs.choose("Vacuum", "Vacío"))
scatt = water_data[-1][:,1] #* np.pi * (diameters[-1]**2) / 4
plt.plot(water_data[-1][:,0], scatt/max(scatt),
         linestyle="dashdot", color=next(colors), 
         label=trs.choose("Water", "Agua"))
# scatt = glassnwater_data[-1][:,1] #* np.pi * (diameters[-1]**2) / 4
# plt.plot(glassnwater_data[0][:,0], scatt/max(scatt),
#          linestyle="dashed", color=next(colors), label="Glass+Water")
plt.plot(marian_exp_data[-1][:,0], 
         marian_exp_data[-1][:,1] / max(marian_exp_data[-1][:,1]), 
         linestyle="dotted", color=next(colors), 
         label="Experimental")
plt.plot(marian_mie_data[-1][:,0], 
         marian_mie_data[-1][:,1] / max(marian_mie_data[-1][:,1]), 
         linestyle="dotted", color=next(colors), 
         label=trs.choose("Barella Mie Water", "Barella Mie Agua"))
plt.xlabel(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
plt.ylabel(trs.choose("Normalized Scattering Cross Section " + 
                      r"$\sigma_{scatt}/\sigma_{scatt}^{max}$",
                      "Sección eficaz de dispersión normalizada " +
                      r"$\sigma_{disp}/\sigma_{scatt}^{disp}$"))
plt.xlim([450, 650])
plt.legend()
vs.saveplot(plot_file("AllWater103Comparison1st.png"), overwrite=True)

#%% 2nd: WATER VS MARIAN THEORY 103 nm
# This makes up the figure I used when posting on Meep's Github

plt.figure()
plt.title(trs.choose("Scattering of Au sphere with 103 nm diameter submerged in water",
                     "Dispersión de nanoesfera de Au con diámetro 103 nm en agua"))
scatt = water_data[-1][:,1] #* np.pi * (diameters[-1]**2) / 4
plt.plot(water_data[-1][:,0], scatt/max(scatt),
         linestyle="solid", color='C0', label="MEEP")
scatt = glassnwater_data[-1][:,1] #* np.pi * (diameters[-1]**2) / 4
plt.plot(marian_mie_data[-1][:,0], 
         marian_mie_data[-1][:,1] / max(marian_mie_data[-1][:,1]), 
         linestyle="dashed", color='C0', label="Mie")
plt.xlabel(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
plt.ylabel(trs.choose("Normalized Scattering Cross Section " + 
                      r"$\sigma_{scatt}/\sigma_{scatt}^{max}$",
                      "Sección eficaz de dispersión normalizada " +
                      r"$\sigma_{disp}/\sigma_{scatt}^{disp}$"))
plt.xlim([500, 650])
plt.legend()
vs.saveplot(plot_file("AllWater103Comparison.png"), overwrite=True)

#%% 3rd: WATER VS WATER MIE THEORIES 1 DIAMETER

fig = plt.figure()
plot_grid = gridspec.GridSpec(ncols=2, nrows=2, hspace=0.4, wspace=0.2, figure=fig)

plot_list = [plot_grid[0,0], plot_grid[0,1], plot_grid[1,0], plot_grid[1,1]]
axes_list = []
for pl in plot_list:
    axes_list.append(fig.add_subplot(pl))

for d, ax in zip(diameters, axes_list):
    index = np.argmin(np.abs(np.array(diameters)-d))
    ax.set_title(f"Diámetro {d} nm")
    ax.plot(water_data[index][:,0], 
            water_data[index][:,1]/max(water_data[index][:,1]),
             linestyle="solid", color='r', 
             label="MEEP R " + trs.choose("Simulated Data", "Simulación"))
    ax.plot(pms_jc_meep_mie_data[index][:,0], 
             pms_jc_meep_mie_data[index][:,1]/max(pms_jc_meep_mie_data[index][:,1]), 
             linestyle="dashed", color='#e377c2', 
             label="PyMieScatt + MEEP JC Mie")
    ax.plot(pms_jc_rinfo_mie_data[index][:,0], 
             pms_jc_rinfo_mie_data[index][:,1]/max(pms_jc_rinfo_mie_data[index][:,1]), 
             linestyle="dotted", color='k', label="PyMieScatt+RIInfo JC Mie")
    ax.plot(pms_r_mie_data[index][:,0], 
             pms_r_mie_data[index][:,1]/max(pms_r_mie_data[index][:,1]), 
             linestyle="dashed", color='#1f77b4', label="PyMieScatt+MEEP R Mie")
    ax.xaxis.set_label_text(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
    ax.yaxis.set_label_text(trs.choose("Normalized Scattering Cross Section\n" +
                                       r"$\sigma_{scatt}/\sigma_{scatt}^{max}$",
                                       "Sección eficaz de dispersión normalizada\n" +
                                       r"$\sigma_{disp}/\sigma_{disp}^{max}$"))
    ax.set_xlim([400, 800])
    
fig.set_size_inches([19.2 ,  8.39])
plt.legend()
vs.saveplot(plot_file("AllWaterFullComparison.png"), overwrite=True)

#%% 3rd: WATER VS WATER MIE THEORIES 1 DIAMETER NOT NORMALIZED

fig = plt.figure()
plot_grid = gridspec.GridSpec(ncols=2, nrows=2, hspace=0.4, wspace=0.1, figure=fig)

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
    ax.xaxis.set_label_text(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
    ax.yaxis.set_label_text(trs.choose(r"Scattering Efficiency $C_{scatt}$", 
                                       "Eficiencia de dispersión $C_{disp}$"))
    ax.set_xlim([400, 800])
    
fig.set_size_inches([19.2 ,  8.39])
plt.legend()
vs.saveplot(plot_file("AllWaterFullComparisonEff.png"), overwrite=True)

# CANNOT PLOT MARIAN'S WITHOUT NORMALIZATION
# Marian's data is already normalized :(

#%% JUST MARIAN'S DATA, TO REPRODUCE HIS PAPER'S DATA

if plot_for_display: use_backend("Agg")

colors = ["C0", "C1", "C2", "C3"]

fig = plt.figure()
if plot_for_display: fig.dpi = 200
for mdt, d, c in zip(marian_exp_data, diameters, colors):
    plt.plot(mdt[:,0], mdt[:,1] / max(mdt[:,1]), 
             linestyle='solid', color=c, 
             label=trs.choose(f"Experimental {d} nm",
                              f"Experimental {d} nm"))
for mdt, d, c in zip(marian_mie_data, diameters, colors):
    plt.plot(mdt[:,0], mdt[:,1] / max(mdt[:,1]), 
             linestyle="dashed", color=c,
             label=trs.choose(f"Mie Theory {d} nm",
                              f"Teoría de Mie {d} nm"))
plt.xlabel(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
plt.ylabel(trs.choose("Normalized Scattering Cross Section " + 
                      r"$\sigma_{scatt}/\sigma_{scatt}^{max}$", 
                      "Sección eficaz de dispersión normalizada " + 
                      r"$\sigma_{disp}/\sigma_{disp}^{max}$"))
plt.xlim([500, 720])
plt.legend(ncol=2)

if plot_make_big: fig.set_size_inches([10.6,  5.5])

vs.saveplot(plot_file("Barella.png"), overwrite=True)

if plot_for_display: use_backend("Qt5Agg")