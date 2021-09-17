#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 23:42:47 2021

@author: vall
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plab
import matplotlib.gridspec as gridspec
import os
import PyMieScatt as ps
from v_materials import import_medium
import v_save as vs
import v_utilities as vu
import v_materials as vmt
import h5py as h5

english = False
trs = vu.BilingualManager(english=english)

#%% PARAMETERS

# Saving directories
folder = "AngularPattern/DipoleStanding/DipoleStandingTest/VacStanding" # 50 nm
# folder = "AngularPattern/DipoleStanding/DipoleStandingTest/TestSpyderEvenBettDisc" # 50 nm but 0.5 displacement, shhh x_D
# folder = "AngularPattern/DipoleStanding/DipoleStandingTest/TestSerial" # 200 nm
# folder = "AngularPattern/DipoleStanding/DipoleStandingTest/TestParallel" # 100 nm
# folder = "AngularPattern/DipoleStanding/DipoleStandingTest/TestSpyderCheckWidth" # 200 nm
home = vs.get_home()

# Scattering plot options
plot_title = trs.choose("Angular pattern of point dipole molecule at 650 nm",
                        "Patrón angular de dipolo puntual como molécula a 650 nm")

plot_make_big = True
plot_file = lambda n : os.path.join(home, "DataAnalysis/AngularPattern/Dipole", "VacStanding"+n)

#%% LOAD DATA

path = os.path.join(home, folder)
file = lambda n : os.path.join(path, n)
data_file = h5.File(file("Results.h5"), "r")

poynting_x = data_file["Px"]
poynting_y = data_file["Py"]
poynting_z = data_file["Pz"]
poynting_r = data_file["Pr"]
params = dict(data_file["Pr"].attrs)

try:
    ram_file = h5.File(file("RAM.h5"))
    params["used_ram"] = np.array(ram_file["RAM"])
except FileNotFoundError:
    ram_file = h5.File(file("Resources.h5"))
    params["used_ram"] = np.array(ram_file["RAM"])
    params["elapsed"] = np.array(ram_file["ElapsedTime"])

resolution = params["resolution"]
from_um_factor = params["from_um_factor"]
submerged_index = params["submerged_index"]
wlen_width = params["wlen_width"]
wlen_center = params["wlen_center"]
npolar = params["npolar"]
nazimuthal = params["nazimuthal"]
nfreq = params["nfreq"]

minor_division = from_um_factor * 1e3 / resolution
width_points = params["cell_width"] * resolution
effective_width_points = (params["cell_width"] - 2 * params["pml_width"]) * resolution
grid_points = width_points**3
max_occupied_ram = np.max(params["used_ram"]) / (1024)**2
memory_B = [[2 * 12 * gp * 32 for p, gp in zip(par, gpoints)] for par, gpoints in zip(params, grid_points)] # in bytes

#%%

# Frequency and wavelength
freq_center = 1/wlen_center # Hz center frequency in Meep units
freq_width = 1/wlen_width # Hz range in Meep units from highest to lowest

freqs = np.linspace(freq_center - freq_center/2, 
                    freq_center + freq_width/2, 
                    nfreq)
wlens = from_um_factor * 1e3 / freqs

azimuthal_angle = np.arange(0, 2 + 2/nazimuthal, 2/nazimuthal) # in multiples of pi
polar_angle = np.arange(0, 1 + 1/npolar, 1/npolar)

#%% PLOT ANGULAR PATTERN IN 3D

wlen_chosen = wlen_center
freq_index = np.argmin(np.abs(freqs - 1/wlen_chosen))

fig = plt.figure()
plt.suptitle(trs.choose(
    f'Angular Pattern of Point Dipole at {from_um_factor * 1e3 * wlen_center:.0f} nm with a ' + 
    f'{from_um_factor * 1e3 * wlen_width:.0f} nm bandwidth',
    f'Patrón angular de dipolo puntual en {from_um_factor * 1e3 * wlen_center:.0f} nm con ') + 
    f'ancho de banda {from_um_factor * 1e3 * wlen_width:.0f} nm')

ax = fig.add_subplot(1,1,1, projection='3d')
ax.set_title(trs.choose(f"evaluated at {from_um_factor * 1e3 * wlen_chosen:.0f} nm",
                        f"evaluado en {from_um_factor * 1e3 * wlen_chosen:.0f} nm"))
ax.plot_surface(
    poynting_x[:,:,freq_index], 
    poynting_y[:,:,freq_index], 
    poynting_z[:,:,freq_index], cmap=plt.get_cmap('jet'), 
    linewidth=1, antialiased=False, alpha=0.5)
ax.set_xlabel(trs.choose(r"Poynting Vector $P_x$ [a.u.]",
                         r"Vector de Poynting $P_x$ [u.a.]"))
ax.set_ylabel(trs.choose(r"Poynting Vector $P_y$ [a.u.]",
                         r"Vector de Poynting $P_y$ [u.a.]"))
ax.set_zlabel(trs.choose(r"Poynting Vector $P_z$ [a.u.]",
                         r"Vector de Poynting $P_z$ [u.a.]"))

plt.savefig(plot_file("AngularPattern.png"))
    
#%% PLOT ANGULAR PATTERN PROFILE FOR DIFFERENT POLAR ANGLES
       
polar_plot = [0, .25, .5, .75, 1]
colors = plt.get_cmap("winter")( np.linspace(0,1,len(polar_plot)) )

wlen_chosen = wlen_center
freq_index = np.argmin(np.abs(freqs - 1/wlen_chosen))
index = [list(polar_angle).index(alpha) for alpha in polar_plot]

fig = plt.figure()
plt.suptitle(trs.choose(
    f'Angular Pattern of Point Dipole at {from_um_factor * 1e3 * wlen_center:.0f} nm for ',
    f'Patrón angular de dipolo puntual en {from_um_factor * 1e3 * wlen_center:.0f} nm para ') + 
    f"{wlen_chosen * 1e3 * from_um_factor:.0f} nm")
ax_plain = plt.axes()
for k, i in enumerate(index):
    ax_plain.plot(poynting_x[:,i,freq_index], 
                  poynting_y[:,i,freq_index], 
                  ".-", color=colors[k],
                  label=rf"$\theta$ = {polar_angle[i]:.2f} $\pi$")
plt.legend(bbox_to_anchor=(1.4, 0.5), #(2.5, 1.4), 
           loc="center right", frameon=False)
ax_plain.set_xlabel(trs.choose(r"Poynting Vector $P_x$ [a.u.]",
                               r"Vector de Poynting $P_x$ [u.a.]"))
ax_plain.set_ylabel(trs.choose(r"Poynting Vector $P_y$ [a.u.]",
                               r"Vector de Poynting $P_y$ [u.a.]"))
ax_plain.set_aspect("equal")

fig.set_size_inches(np.array([7.13, 4.8 ]))
plt.savefig(plot_file("AngularPolar.png"))

#%% PLOT ANGULAR PATTERN PROFILE FOR DIFFERENT AZIMUTHAL ANGLES
       
azimuthal_plot = [0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2]
colors = plt.get_cmap("spring")( np.linspace(0,1,len(azimuthal_plot)) )

wlen_chosen = wlen_center
freq_index = np.argmin(np.abs(freqs - 1/wlen_chosen))
index = [list(azimuthal_angle).index(alpha) for alpha in azimuthal_plot]

fig = plt.figure()
plt.suptitle(trs.choose(
    f'Angular Pattern of Point Dipole at {from_um_factor * 1e3 * wlen_center:.0f} nm for ',
    f'Patrón angular de dipolo puntual en {from_um_factor * 1e3 * wlen_center:.0f} nm para ') + 
    f"{wlen_chosen * 1e3 * from_um_factor:.0f} nm")

ax_plain = plt.subplot()
for k, i in enumerate(index):
    ax_plain.plot(poynting_x[i,:,freq_index], 
                  poynting_z[i,:,freq_index], 
                  ".-", color=colors[k],
                  label=rf"$\phi$ = {azimuthal_angle[i]:.2f} $\pi$")
# ax_plain.legend(bbox_to_anchor=(1.4, 0.5), #(2.5, 1.4), 
#            loc="center right", frameon=False)

for ax in fig.axes:
    box = ax.get_position()
    box.x1 = box.x1 - .15 * (box.x1 - box.x0)
    ax.set_position(box)
leg = ax_plain.legend(bbox_to_anchor=(1.3, 0.5), loc="center right", frameon=False)

ax_plain.set_xlabel(trs.choose(r"Poynting Vector $P_x$ [a.u.]",
                               r"Vector de Poynting $P_x$ [u.a.]"))
ax_plain.set_ylabel(trs.choose(r"Poynting Vector $P_z$ [a.u.]",
                               r"Vector de Poynting $P_z$ [u.a.]"))

fig.set_size_inches([7.13, 4.8 ])
plt.savefig(plot_file("AngularAzimuthal.png"))
    
#%%

azimuthal_plot = [0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2]
colors = plt.get_cmap("spring")( np.linspace(0,1,len(azimuthal_plot)) )

wlen_chosen = wlen_center
freq_index = np.argmin(np.abs(freqs - 1/wlen_chosen))
index = [list(azimuthal_angle).index(alpha) for alpha in azimuthal_plot]

fig = plt.figure()
plt.suptitle(trs.choose(
    f'Angular Pattern of Point Dipole at {from_um_factor * 1e3 * wlen_center:.0f} nm for ',
    f'Patrón angular de dipolo puntual en {from_um_factor * 1e3 * wlen_center:.0f} nm para ') + 
    f"{wlen_chosen * 1e3 * from_um_factor:.0f} nm")

ax_plain = plt.subplot()
for k, i in enumerate(index):
    ax_plain.plot(np.sqrt(np.square(poynting_x[i,:,freq_index]) + np.square(poynting_y[i,:,freq_index])), 
                          poynting_z[i,:,freq_index], 
                          ".-", color=colors[k],
                          label=rf"$\phi$ = {azimuthal_angle[i]:.2f} $\pi$")
# ax_plain.legend(bbox_to_anchor=(1.4, 0.5), #(2.5, 1.4), 
#            loc="center right", frameon=False)

for ax in fig.axes:
    box = ax.get_position()
    box.x1 = box.x1 - .15 * (box.x1 - box.x0)
    ax.set_position(box)
leg = ax_plain.legend(bbox_to_anchor=(1.3, 0.5), loc="center right", frameon=False)

ax_plain.set_xlabel(trs.choose(r"Poynting Vector $P_\rho$ [a.u.]",
                                 r"Vector de Poynting $P_\rho$ [u.a.]"))
ax_plain.set_ylabel(trs.choose(r"Poynting Vector $P_z$ [a.u.]",
                               r"Vector de Poynting $P_z$ [u.a.]"))

fig.set_size_inches([7.13, 4.8 ])
plt.savefig(plot_file("AngularAzimuthalAbs.png"))

#%%

azimuthal_plot = [0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2]
colors = plt.get_cmap("spring")( np.linspace(0,1,len(azimuthal_plot)) )

wlen_chosen = wlen_center
freq_index = np.argmin(np.abs(freqs - 1/wlen_chosen))
index = [list(azimuthal_angle).index(alpha) for alpha in azimuthal_plot]

fig = plt.figure()
plt.suptitle(trs.choose(
    f'Angular Pattern of Point Dipole at {from_um_factor * 1e3 * wlen_center:.0f} nm for ',
    f'Patrón angular de dipolo puntual en {from_um_factor * 1e3 * wlen_center:.0f} nm para ') + 
    f"{wlen_chosen * 1e3 * from_um_factor:.0f} nm")

ax_plain, ax_plain_2 = fig.subplots(nrows=2, sharey=True,
                                    gridspec_kw={"hspace":0.3})
for k, i in enumerate(index):
    ax_plain.axvline(0, linewidth=0.5, color="k")
    ax_plain.axhline(0, linewidth=0.5, color="k")
    ax_plain.plot(poynting_x[i,:,freq_index], 
                  poynting_z[i,:,freq_index], 
                  ".-", color=colors[k],
                  label=rf"$\phi$ = {azimuthal_angle[i]:.2f} $\pi$")
    ax_plain_2.axvline(0, linewidth=0.5, color="k")
    ax_plain_2.axhline(0, linewidth=0.5, color="k")
    ax_plain_2.plot(np.sqrt(np.square(poynting_x[i,:,freq_index]) + np.square(poynting_y[i,:,freq_index])), 
                    poynting_z[i,:,freq_index], 
                    ".-", color=colors[k],
                    label=rf"$\phi$ = {azimuthal_angle[i]:.2f} $\pi$")
# ax_plain.legend(bbox_to_anchor=(1.4, 0.5), #(2.5, 1.4), 
#            loc="center right", frameon=False)

for ax in fig.axes:
    box = ax.get_position()
    box.x1 = box.x1 - .15 * (box.x1 - box.x0)
    ax.set_position(box)
leg = ax_plain.legend(bbox_to_anchor=(1.3, -0.2), loc="center right", frameon=False)

ax_plain.set_xlabel(trs.choose(r"Poynting Vector $P_x$ [a.u.]",
                               r"Vector de Poynting $P_x$ [u.a.]"))
ax_plain.set_ylabel(trs.choose(r"Poynting Vector $P_z$ [a.u.]",
                               r"Vector de Poynting $P_z$ [u.a.]"))

ax_plain_2.set_xlabel(trs.choose(r"Poynting Vector $P_\rho$ [a.u.]",
                                 r"Vector de Poynting $P_\rho$ [u.a.]"))
ax_plain_2.set_ylabel(trs.choose(r"Poynting Vector $P_z$ [a.u.]",
                                 r"Vector de Poynting $P_z$ [u.a.]"))

fig.set_size_inches([7.13, 4.8 ])
plt.savefig(plot_file("AngularAzimuthalBoth.png"))

#%%

polar_plot = [0, .25, .5, .75, 1]
polar_colors = plt.get_cmap("winter")( np.linspace(0,1,len(polar_plot)+1) )[:-1]

azimuthal_plot = [0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2]
azimuthal_colors = plt.get_cmap("spring")( np.linspace(0,1,len(azimuthal_plot)+1) )[:-1]

wlen_chosen = wlen_center
freq_index = np.argmin(np.abs(freqs - 1/wlen_chosen))
polar_index = [list(polar_angle).index(alpha) for alpha in polar_plot]
azimuthal_index = [list(azimuthal_angle).index(alpha) for alpha in azimuthal_plot]

fig = plt.figure()
# plt.suptitle(trs.choose(
#     f'Angular Pattern of Point Dipole at {from_um_factor * 1e3 * wlen_center:.0f} nm for ',
#     f'Patrón angular de dipolo puntual en {from_um_factor * 1e3 * wlen_center:.0f} nm para ') + 
#     f"{wlen_chosen * 1e3 * from_um_factor:.0f} nm")

plot_grid = gridspec.GridSpec(ncols=5, nrows=2, hspace=0.25, wspace=0.65, figure=fig)

pol_axis = fig.add_subplot(plot_grid[:,0:2])
azi_axis = fig.add_subplot(plot_grid[0,2:])
azi_axis_abs = fig.add_subplot(plot_grid[1,2:])
fig.set_size_inches(np.array([8.94, 4.8 ])*1.5)

for k, i in enumerate(polar_index):
    pol_axis.axvline(0, linewidth=0.5, color="k")
    pol_axis.axhline(0, linewidth=0.5, color="k")
    pol_axis.plot(poynting_x[:,i,freq_index], 
                  poynting_y[:,i,freq_index], 
                  ".-", color=polar_colors[k],
                  label=rf"$\theta$ = {polar_angle[i]:.2f} $\pi$")
# plt.legend(bbox_to_anchor=(1.4, 0.5), #(2.5, 1.4), 
#            loc="center right", frameon=False)
pol_axis.set_xlabel(trs.choose(r"Poynting Vector $P_x$ [a.u.]",
                               r"Vector de Poynting $P_x$ [u.a.]"),
                    fontsize=15)
pol_axis.set_ylabel(trs.choose(r"Poynting Vector $P_y$ [a.u.]",
                               r"Vector de Poynting $P_y$ [u.a.]"),
                    fontsize=15)
pol_axis.xaxis.set_tick_params(labelsize=11)
pol_axis.yaxis.set_tick_params(labelsize=11)
pol_axis.set_aspect("equal")

for k, i in enumerate(azimuthal_index):
    azi_axis.axvline(0, linewidth=0.5, color="k")
    azi_axis.axhline(0, linewidth=0.5, color="k")
    azi_axis.plot(poynting_x[i,:,freq_index], 
                  poynting_z[i,:,freq_index], 
                  ".-", color=colors[k],
                  label=rf"$\phi$ = {azimuthal_angle[i]:.2f} $\pi$")
    azi_axis_abs.axvline(0, linewidth=0.5, color="k")
    azi_axis_abs.axhline(0, linewidth=0.5, color="k")
    azi_axis_abs.plot(np.sqrt(np.square(poynting_x[i,:,freq_index]) + np.square(poynting_y[i,:,freq_index])), 
                    poynting_z[i,:,freq_index], 
                    ".-", color=colors[k],
                    label=rf"$\phi$ = {azimuthal_angle[i]:.2f} $\pi$")
# ax_plain.legend(bbox_to_anchor=(1.4, 0.5), #(2.5, 1.4), 
#            loc="center right", frameon=False)

azi_axis.set_xlabel(trs.choose(r"Poynting Vector $P_x$ [a.u.]",
                               r"Vector de Poynting $P_x$ [u.a.]"),
                    fontsize=15)
azi_axis.set_ylabel(trs.choose(r"Poynting Vector $P_z$ [a.u.]",
                               r"Vector de Poynting $P_z$ [u.a.]"),
                    fontsize=15)

azi_axis_abs.set_xlabel(trs.choose(r"Poynting Vector $P_\rho$ [a.u.]",
                                   r"Vector de Poynting $P_\rho$ [u.a.]"),
                        fontsize=15)
azi_axis_abs.set_ylabel(trs.choose(r"Poynting Vector $P_z$ [a.u.]",
                                   r"Vector de Poynting $P_z$ [u.a.]"),
                        fontsize=15)
azi_axis.xaxis.set_tick_params(labelsize=11)
azi_axis.yaxis.set_tick_params(labelsize=11)
azi_axis_abs.xaxis.set_tick_params(labelsize=11)
azi_axis_abs.yaxis.set_tick_params(labelsize=11)

for ax in [azi_axis, azi_axis_abs]:
    box = ax.get_position()
    box.x1 = box.x1 - .15 * (box.x1 - box.x0)
    ax.set_position(box)
# leg = azi_axis.legend(bbox_to_anchor=(1.33, -.1), loc="center right", frameon=False)
leg = azi_axis.legend(bbox_to_anchor=(-.18, -1.14), loc="center right", ncol=2, frameon=False, fontsize=12)

box = pol_axis.get_position()
box.y0, box.y1 = box.y0 + .275 * (box.y1 - box.y0), box.y1 + .275 * (box.y1 - box.y0)
box.x0, box.x1 = box.x0 - .1 * (box.x1 - box.x0), box.x1 - .1 * (box.x1 - box.x0)
pol_axis.set_position(box)
# leg = pol_axis.legend(bbox_to_anchor=(0.7, -0.45), loc="center right", frameon=False)
leg = pol_axis.legend(bbox_to_anchor=(0.15, -0.45), loc="center right", frameon=False, fontsize=12)

plt.savefig(plot_file("AngularAzimuthalAll.png"))
