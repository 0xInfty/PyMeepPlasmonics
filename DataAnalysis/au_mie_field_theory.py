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
import v_theory as vt
import v_save as vs
import v_materials as vmt

#%% PARAMETERS

r = 30 # nm
wlen_range = np.array([400, 800]) # nm
wlen_chosen = [405, 532, 642]
epsilon_ext = (1.33)**2 # Water
# E0_func = lambda n : np.array([[0,0,1] for i in range(n)])
E0 = np.array([0,0,1])

npoints = 500

folder = "AllWaterField2"

#%% SAVING CONFIGURATION

series = f"{folder}{2*r:1.0f}"
plot_folder = f"DataAnalysis/{folder}/{series}"
home = vs.get_home()
if not os.path.isdir(os.path.join(home, plot_folder)): 
    vs.new_dir(os.path.join(home, plot_folder))
plot_file = lambda n : os.path.join(
    home, plot_folder, f"AllWaterField{2*r:1.0f}" + n)

#%% INNER MEDIUM EPSILON

wlen = np.linspace(*wlen_range, npoints)

epsilon_func_meep_jc = vmt.epsilon_function(material="Au", 
                                           paper="JC", 
                                           reference="Meep")
epsilon_func_meep_r = vmt.epsilon_function(material="Au", 
                                          paper="R", 
                                          reference="Meep")
epsilon_func_exp_jc = vmt.epsilon_function(material="Au", 
                                          paper="JC", 
                                          reference="RIinfo")
epsilon_func_exp_r = vmt.epsilon_function(material="Au", 
                                          paper="R", 
                                          reference="RIinfo")

epsilon_meep_jc = np.array([epsilon_func_meep_jc(wl) for wl in wlen])
epsilon_meep_r = np.array([epsilon_func_meep_r(wl) for wl in wlen])
epsilon_exp_jc = np.array([epsilon_func_exp_jc(wl) for wl in wlen])
epsilon_exp_r = np.array([epsilon_func_exp_r(wl) for wl in wlen])

epsilon = np.array([epsilon_meep_jc, 
                    epsilon_meep_r,
                    epsilon_exp_jc,
                    epsilon_exp_r])
wlen_long = np.array([*[wlen]*4])

functions = [np.abs, np.real, np.imag]
titles = ["Absolute value", "Real part", "Imaginary part"]
ylabels = [r"|$\alpha$| [nm$^3$]", r"Re($\alpha$) [nm$^3$]", r"Im($\alpha$) [nm$^3$]"]
labels = ["Meep: Johnson & Christy", "Meep: Rakic", 
          "RIinfo: Johnson & Christy", "RIinfo: Rakic"]

nplots = len(functions)
fig = plt.figure(figsize=(nplots*6.4, 6.4))
axes = fig.subplots(ncols=nplots)

max_value = []
min_value = []
for ax, f, t, y in zip(axes, functions, titles, ylabels):
    for wl, eps, l in zip(wlen_long, epsilon, labels):
        ax.set_title(t)
        print(l, eps[:2])
        ax.plot(wl, f(eps), label=l)
        ax.xaxis.set_label_text("Wavelength [nm]")
        ax.yaxis.set_label_text(y)
        ax.legend()
        ax.set_xlim(*wlen_range)
        max_value.append(max(f(eps)))
        min_value.append(min(f(eps)))
        
for ax in axes: ax.set_ylim([min(min_value)-.1*(max(max_value)-min(min_value)), 
                             max(max_value)+.1*(max(max_value)-min(min_value))])
    
plt.savefig(plot_file("Epsilon.png"))

#%% CLAUSIUS-MOSETTI: POLARIZABILITY

alpha_cm_meep_r = vt.alpha_Clausius_Mosotti(epsilon_meep_r, r,
                                           epsilon_ext=epsilon_ext)
alpha_cm_exp_jc = vt.alpha_Clausius_Mosotti(epsilon_exp_jc, r,
                                            epsilon_ext=epsilon_ext)

alpha_cm = np.array([alpha_cm_meep_r,
                     alpha_cm_exp_jc])

#%% KUWATA: POLARIZABILITY
   
alpha_k_meep_r = vt.alpha_Kuwata(epsilon_meep_r, wlen, r,
                                  epsilon_ext=epsilon_ext)
alpha_k_exp_jc = vt.alpha_Kuwata(epsilon_exp_jc, wlen, r,
                                  epsilon_ext=epsilon_ext)

alpha_k = np.array([alpha_k_meep_r,
                    alpha_k_exp_jc])

#%% PLOT POLARIZABILITY

functions = [np.abs, np.real, np.imag]
titles = ["Absolute value", "Real part", "Imaginary part"]
ylabels = [r"|$\alpha$| [nm$^3$]", r"Re($\alpha$) [nm$^3$]", r"Im($\alpha$) [nm$^3$]"]
wlen_alpha = [*[wlen]*4]
alpha = [*alpha_cm, *alpha_k]
labels = ["Meep: Claussius-Mosotti", "Jhonson: Claussius-Mosotti", 
          "Meep: Kuwata", "Jhonson: Kuwata"]

nplots = len(functions)
fig = plt.figure(figsize=(nplots*6.4, 6.4))
axes = fig.subplots(ncols=nplots)

max_value = []
min_value = []
for ax, f, t, y in zip(axes, functions, titles, ylabels):
    for wl, al, l in zip(wlen_alpha, alpha, labels):
        ax.set_title(t)
        print(l, eps[:10])
        ax.plot(wl, f(al), label=l)
        ax.xaxis.set_label_text("Wavelength [nm]")
        ax.yaxis.set_label_text(y)
        ax.legend()
        ax.set_xlim(*wlen_range)
        max_value.append(max(f(al)))
        min_value.append(min(f(al)))
        
for ax in axes: ax.set_ylim([min(min_value)-.1*(max(max_value)-min(min_value)), 
                             max(max_value)+.1*(max(max_value)-min(min_value))])
    
plt.savefig(plot_file("Polarizability.png"))

#%% DIPOLAR APROXIMATION: ELECTRIC FIELD EZ IN Z

rvec = np.zeros((npoints, 3))
rvec[:,2] = np.linspace(-3*r, 3*r, npoints)

E_cm_meep_r_chosen = []
E_k_meep_r_chosen = []
E_cm_exp_jc_chosen = []
E_k_exp_jc_chosen = []
for wl in wlen_chosen:
    e_meep_r_chosen = epsilon_func_meep_r(wl)
    a_cm_meep_r_chosen = vt.alpha_Clausius_Mosotti(e_meep_r_chosen, r, 
                                                   epsilon_ext=epsilon_ext)
    a_k_meep_r_chosen = vt.alpha_Kuwata(e_meep_r_chosen, wl, r,
                                        epsilon_ext=epsilon_ext)
    e_exp_jc_chosen = epsilon_func_exp_jc(wl)
    a_cm_exp_jc_chosen = vt.alpha_Clausius_Mosotti(e_exp_jc_chosen, r,
                                                   epsilon_ext=epsilon_ext)
    a_k_exp_jc_chosen = vt.alpha_Kuwata(e_exp_jc_chosen, wl, r,
                                        epsilon_ext=epsilon_ext)
    E_cm_meep_r_chosen.append(
        np.array([vt.E(e_meep_r_chosen, a_cm_meep_r_chosen, 
                       E0, rv, r, epsilon_ext=epsilon_ext) for rv in rvec]))
    E_k_meep_r_chosen.append(
        np.array([vt.E(e_meep_r_chosen, a_k_meep_r_chosen, 
                       E0, rv, r, epsilon_ext=epsilon_ext) for rv in rvec]))
    E_cm_exp_jc_chosen.append(
        np.array([vt.E(e_exp_jc_chosen, a_cm_exp_jc_chosen, 
                       E0, rv, r, epsilon_ext=epsilon_ext) for rv in rvec]))
    E_k_exp_jc_chosen.append(
        np.array([vt.E(e_exp_jc_chosen, a_k_exp_jc_chosen, 
                       E0, rv, r, epsilon_ext=epsilon_ext) for rv in rvec]))

E_chosen = [E_cm_meep_r_chosen, E_k_meep_r_chosen,
            E_cm_exp_jc_chosen, E_k_exp_jc_chosen]
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
    axes[i].set_title(f'$\lambda$={wlen_chosen[i]:1.0f} nm')
    for j in range(len(E_labels)):
        axes[i].plot(rvec[:,2], np.real(E_chosen[j][i][:,2]))
        E_max.append(np.max(np.real(E_chosen[j][i][:,2])))
        E_min.append(np.min(np.real(E_chosen[j][i][:,2])))
    axes[i].set_xlabel("Distance in z [u.a.]")
    axes[i].set_ylabel("Electric field Ez [u.a.]")
    axes[i].legend(E_labels)
lims = [min(E_min), max(E_max)]
lims = [lims[0] - 0.25*(lims[1]-lims[0]), lims[1] + 0.25*(lims[1]-lims[0])]
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