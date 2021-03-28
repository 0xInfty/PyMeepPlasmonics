#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 22:51:24 2021

@author: vall
"""

from socket import gethostname
if "Nano" in gethostname():
    syshome = "/home/nanofisica/Documents/Vale/ThesisPython"
elif "vall" in gethostname():
    syshome = "/home/vall/Documents/Thesis/ThesisPython"
else:
    raise ValueError("Your PC must be registered at the top of this code")

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
import v_materials as vm
import v_theory as vt
import v_save as vs

#%% PARAMETERS

from_um_factor = 10e-3
r = 6
wlen_range = np.array([35, 70])
wlen_chosen = [40, 57, 75]
epsilon_ext = 1 # Vacuum
medium = vm.import_medium("Au", from_um_factor)
# E0_func = lambda n : np.array([[0,0,1] for i in range(n)])
E0 = np.array([0,0,1])

npoints = 500

data_folder = "AuMieSphere/AuSphereField"
data_file = "RefractiveIndexJohnson.csv"
plot_folder = "DataAnalysis"

#%% SAVING CONFIGURATION

home = vs.get_home()
plot_file = lambda n : os.path.join(
    home, plot_folder, f"VacuumField{2*r*from_um_factor*1e3:1.0f}" + n + '.png')
data_file = os.path.join(home, data_folder, data_file)

#%% INNER MEDIUM EPSILON

wlen_meep = np.linspace(*wlen_range, npoints)

freq_meep = 1/wlen_meep
epsilon_meep = np.array([medium.epsilon(f) for f in freq_meep]) # Tensor
epsilon_meep = np.array([et[0,0] for et in epsilon_meep]) # Isotropic

data_juli = np.loadtxt(data_file, delimiter=',', skiprows=1)
wlen_juli = data_juli[:,0] * 1e3 # Wavelength in nm
n_juli = data_juli[:,1] + 1j * data_juli[:,2] # Refractive Index
epsilon_juli = np.power(n_juli, 2) # Because permeability is mu = 1

wlen_john = wlen_meep
n_real_john = interp1d(wlen_juli, data_juli[:,1], kind="cubic")
n_imag_john = interp1d(wlen_juli, data_juli[:,2], kind="cubic")
n_func_john = lambda wl : n_real_john(wl) + 1j * n_imag_john(wl)
epsilon_func_john = lambda wl : np.power(n_func_john(wl), 2)
epsilon_john = epsilon_func_john(wlen_meep * from_um_factor * 1e3)

epsilon = [epsilon_meep, epsilon_john]
wlen = [*[wlen_meep * from_um_factor * 1e3]*2]
# epsilon = [epsilon_meep, epsilon_juli]
# wlen = [wlen_meep * from_um_factor * 1e3, wlen_juli]

#%% CLAUSIUS-MOSETTI: POLARIZABILITY

alpha_cm_meep = vt.alpha_Clausius_Mosotti(epsilon_meep, 
                                          r * from_um_factor * 1e3,
                                          epsilon_ext=epsilon_ext)
alpha_cm_john = vt.alpha_Clausius_Mosotti(epsilon_john, 
                                          r * from_um_factor * 1e3,
                                          epsilon_ext=epsilon_ext)
alpha_cm = [alpha_cm_meep, alpha_cm_john]

#%% KUWATA: POLARIZABILITY
   
alpha_k_meep = vt.alpha_Kuwata(epsilon_meep, 
                               wlen_meep * from_um_factor * 1e3, 
                               r * from_um_factor * 1e3,
                               epsilon_ext=epsilon_ext)
alpha_k_john = vt.alpha_Kuwata(epsilon_john, 
                               wlen_john * from_um_factor * 1e3, 
                               r * from_um_factor * 1e3,
                               epsilon_ext=epsilon_ext)
alpha_k = [alpha_k_meep, alpha_k_john]

#%% PLOT POLARIZABILITY

functions = [np.abs, np.real, np.imag]
titles = ["Absolute value", "Real part", "Imaginary part"]
ylabels = [r"|$\alpha$| [nm$^3$]", r"Re($\alpha$) [nm$^3$]", r"Im($\alpha$) [nm$^3$]"]
wlen_alpha = [*[*wlen]*2]
alpha = [*alpha_cm, *alpha_k]
labels = ["Meep: Claussius-Mosotti", "Jhonson: Claussius-Mosotti", 
          "Meep: Kuwata", "Jhonson: Kuwata"]

nplots = len(functions)
fig = plt.figure(figsize=(nplots*6.4, 6.4))
axes = fig.subplots(ncols=nplots)

max_value = []
for ax, f, t, y in zip(axes, functions, titles, ylabels):
    for wl, a, l in zip(wlen_alpha, alpha, labels):
        ax.set_title(t)
        ax.plot(wl, f(a), label=l)
        ax.xaxis.set_label_text("Wavelength [nm]")
        ax.yaxis.set_label_text(y)
        ax.legend()
        ax.set_xlim(*(wlen_range * from_um_factor * 1e3))
        max_value.append(max(f(a)))
        
for ax in axes: ax.set_ylim([0, 1.1*max(max_value)])
    
plt.savefig(plot_file("Polarizability.png"))

#%% DIPOLAR APROXIMATION: ELECTRIC FIELD EZ IN Z

rvec = np.zeros((npoints, 3))
rvec[:,2] = np.linspace(-3*r, 3*r, npoints)

E_cm_meep_chosen = []
E_k_meep_chosen = []
E_cm_john_chosen = []
E_k_john_chosen = []
for wl in wlen_chosen:
    e_meep_chosen = medium.epsilon(1/wl)[0,0]
    a_cm_meep_chosen = vt.alpha_Clausius_Mosotti(e_meep_chosen, r, 
                                                 epsilon_ext=epsilon_ext)
    a_k_meep_chosen = vt.alpha_Kuwata(e_meep_chosen, wl*from_um_factor*1e3, r,
                                      epsilon_ext=epsilon_ext)
    e_john_chosen = epsilon_func_john(wl * from_um_factor * 1e3)
    a_cm_john_chosen = vt.alpha_Clausius_Mosotti(e_john_chosen, r,
                                                 epsilon_ext=epsilon_ext)
    a_k_john_chosen = vt.alpha_Kuwata(e_john_chosen, wl*from_um_factor*1e3, r,
                                      epsilon_ext=epsilon_ext)
    E_cm_meep_chosen.append(
        np.array([vt.E(e_meep_chosen, a_cm_meep_chosen, 
                       E0, rv, r, epsilon_ext=epsilon_ext) for rv in rvec]))
    E_k_meep_chosen.append(
        np.array([vt.E(e_meep_chosen, a_k_meep_chosen, 
                       E0, rv, r, epsilon_ext=epsilon_ext) for rv in rvec]))
    E_cm_john_chosen.append(
        np.array([vt.E(e_john_chosen, a_cm_john_chosen, 
                       E0, rv, r, epsilon_ext=epsilon_ext) for rv in rvec]))
    E_k_john_chosen.append(
        np.array([vt.E(e_john_chosen, a_k_john_chosen, 
                       E0, rv, r, epsilon_ext=epsilon_ext) for rv in rvec]))

E_chosen = [E_cm_meep_chosen, E_k_meep_chosen,
            E_cm_john_chosen, E_k_john_chosen]
E_labels = ["Meep + Claussius-Mosotti",
            "Meep + Kuwata",
            "Johnson + Claussius-Mossetti",
            "Johnson + Kuwata"]
# E_plane_cm_meep = np.array([E(epsilon_meep, alpha_cm_meep, E0, rv) for rv in rvec])
# First index: position
# Second index: wavelength
# Third index: direction

#%%

n = len(wlen_chosen)
fig = plt.figure(figsize=(n*6.4, 6.4))
axes = fig.subplots(ncols=n)

E_max = []
E_min = []
for i in range(n):
    axes[i].set_title(f'$\lambda$={wlen_chosen[i]*from_um_factor*1e3:1.0f} nm')
    for j in range(len(E_labels)):
        axes[i].plot(rvec[:,2], np.real(E_chosen[j][i][:,2]))
        E_max.append(np.max(np.real(E_chosen[j][i][:,2])))
        E_min.append(np.min(np.real(E_chosen[j][i][:,2])))
    axes[i].set_xlabel("Distance in z [u.a.]")
    axes[i].set_ylabel("Electric field Ez [u.a.]")
    axes[i].legend(E_labels)
lims = [min(E_min), max(E_max)]
lims = [lims[0] - 0.1*(lims[1]-lims[0]), lims[1] + 0.1*(lims[1]-lims[0])]
for ax in axes: ax.set_ylim(*lims)

plt.savefig(plot_file("FieldProfile.png"))

#%%

# n = len(wlen_chosen)
# fig = plt.figure(figsize=(n*6.4, 6.4))
# axes = fig.subplots(ncols=n)
# lims = [abs(np.min(np.real(E_cm_chosen))), abs(np.max(np.real(E_cm_chosen)))]
# lims = [-max(lims), max(lims)]

# for j in range(n):
#     axes[j].imshow(np.real(E_cm_chosen[j]), 
#                    interpolation='spline36', cmap='RdBu', 
#                    vmin=lims[0], vmax=lims[1])
#     axes[j].set_xlabel("Distancia en x (u.a.)", fontsize=18)
#     axes[j].set_ylabel("Distancia en y (u.a.)", fontsize=18)
#     axes[j].set_title("$\lambda$={} nm".format(wlen_chosen[j]*10), fontsize=22)
#     plt.setp(axes[j].get_xticklabels(), fontsize=16)
#     plt.setp(axes[j].get_yticklabels(), fontsize=16)
# plt.savefig(file("MaxFieldPlane.png"))