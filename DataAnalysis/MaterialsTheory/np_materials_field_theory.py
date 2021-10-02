#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 22:51:24 2021

@author: vall
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import v_theory as vt
import v_plot as vp
import v_save as vs
import v_materials as vmt
import v_utilities as vu

english = False
trs = vu.BilingualManager(english=english)
vp.set_style()

syshome = vs.get_sys_home()

#%% PARAMETERS

r = 30 # nm
material = "Au"
wlen_range = np.array([300, 800])
wlen_chosen = [405, 532, 642]
submerged_index = 1.33 # Water

# E0_func = lambda n : np.array([[0,0,1] for i in range(n)])
E0 = np.array([0,0,1])

npoints = 500

folder = "DataAnalysis/MaterialsPaper"

#%% MORE PARAMETERS

def get_wlen_range(material, surrounding_index):
    if material=="Au":
        if surrounding_index == 1: return (450, 600) # (450, 650)
        elif surrounding_index == 1.33: return (500, 650) # (500, 650)
    else:
        raise ValueError("Please, expand this function.")
    
if wlen_range is None:
    wlen_range = get_wlen_range(material, submerged_index)
epsilon_ext = (submerged_index)**2

#%% SAVING CONFIGURATION

series = f"All{vmt.recognize_material(submerged_index)}{material}Sphere{2*r:1.0f}"
home = vs.get_home()
if not os.path.isdir(os.path.join(home, folder)): 
    vs.new_dir(os.path.join(home, folder))
plot_file = lambda n : os.path.join(
    home, folder, f"AllWaterField{2*r:1.0f}" + n)

plot_title = trs.choose(f"{material} NP of {2*r:1.0f} nm diameter in {vmt.recognize_material(submerged_index).lower()}",
                        f"NP de {material} de {2*r:1.0f} nm de diámetro en {vmt.recognize_material(submerged_index, english).lower()}")

#%% INNER MEDIUM EPSILON

wlen = np.linspace(*wlen_range, npoints)

epsilon_func_meep_jc = vmt.epsilon_function(material=material, 
                                           paper="JC", 
                                           reference="Meep")
epsilon_func_meep_r = vmt.epsilon_function(material=material, 
                                          paper="R", 
                                          reference="Meep")
epsilon_func_exp_jc = vmt.epsilon_function(material=material, 
                                          paper="JC", 
                                          reference="RIinfo")
epsilon_func_exp_r = vmt.epsilon_function(material=material, 
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
titles = trs.choose([r"Absolute value |$\epsilon$|", r"Real part $\Re$e$(\epsilon$)", r"Imaginary part $\Im$m$(\epsilon$)$"],
                    [r"Valor absoluto |$\epsilon$|", r"Parte real $\Re$e$(\epsilon)$", r"Parte imaginaria $\Im$m$(\epsilon$)"])
labels = ["Meep: Johnson & Christy", "Meep: Rakic", 
          "RIinfo: Johnson & Christy", "RIinfo: Rakic"]

nplots = len(functions)
fig = plt.figure(figsize=(nplots*6.4, 6.4))
axes = fig.subplots(ncols=nplots, sharex=True, sharey=True, 
                    gridspec_kw={"wspace":0})

colors = ["C2", "C1"]*2
linestyles = [*["solid"]*2, *["dotted"]*2]
transparencies = [*[.5]*2, *[1]*2]

plt.suptitle(plot_title)
for ax, f, t in zip(axes, functions, titles):
    for wl, eps, l, col, lsy, tps in zip(wlen_long, epsilon, labels, 
                                         colors, linestyles, transparencies):
        ax.set_title(t)
        print(l, eps[:2])
        ax.plot(wl, f(eps), label=l, linewidth=2, alpha=tps, color=col, linestyle=lsy)
        ax.xaxis.set_label_text(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
        if ax == axes[0]:
            ax.yaxis.set_label_text(trs.choose(r"Electric Permitivity $\epsilon$", 
                                               r"Permitividad eléctrica $\epsilon$"))
        ax.legend()
        ax.set_xlim(*wlen_range)
    
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
titles = trs.choose([r"Absolute value |$\alpha$|", r"Real part $\Re$e$(\alpha$)", r"Imaginary part $\Im$m$(\alpha$)$"],
                    [r"Valor absoluto |$\alpha$|", r"Parte real $\Re$e$(\alpha)$", r"Parte imaginaria $\Im$m$(\alpha$)"])
wlen_alpha = [*[wlen]*4]
alpha = [*alpha_cm, *alpha_k]
labels = ["Meep R: Claussius-Mosotti", "RIinfo J&C: Claussius-Mosotti", 
          "Meep R: Kuwata", "RIinfo J&C: Kuwata"]

nplots = len(functions)
fig, axes = plt.subplots(ncols=nplots, sharex=True, sharey=True, 
                         gridspec_kw={"wspace":0})
fig.set_size_inches((nplots*6.4, 6.4))

colors = ["C0", "C3"]*2
linestyles = [*["solid"]*2, *["dotted"]*2]
transparencies = [*[.5]*2, *[1]*2]

plt.suptitle(plot_title)
for ax, f, t in zip(axes, functions, titles):
    for wl, al, l, col, lsy, trp in zip(wlen_alpha, alpha, labels,
                                        colors, linestyles, transparencies):
        ax.set_title(t)
        print(l, eps[:10])
        ax.plot(wl, f(al), label=l, color=col, linestyle=lsy, alpha=trp, linewidth=2)
        ax.xaxis.set_label_text(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
        if ax == axes[0]:
            ax.yaxis.set_label_text(trs.choose("Polarizability", "Polarizabilidad") + r" $\alpha$ [nm$^3$]")
        ax.legend()
        ax.set_xlim(*wlen_range)
    
plt.savefig(plot_file("Polarizability.png"))

#%% DIPOLAR APROXIMATION: ELECTRIC FIELD EZ IN Z

rvec = np.zeros((npoints, 3))
rvec[:,2] = np.linspace(-4*r, 4*r, npoints)

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

E_chosen = [E_cm_meep_r_chosen, E_cm_exp_jc_chosen,
            E_k_meep_r_chosen, E_k_exp_jc_chosen]
E_labels = ["Meep R: Claussius-Mosotti", "RIinfo J&C: Claussius-Mosotti", 
            "Meep R: Kuwata", "RIinfo J&C: Kuwata"]
# E_plane_cm_meep = np.array([E(epsilon_meep, alpha_cm_meep, E0, rv) for rv in rvec])
# First index: position
# Second index: wavelength
# Third index: direction

#%%

n = len(wlen_chosen)
fig = plt.figure(figsize=(n*6.4, 6.4))
axes = fig.subplots(ncols=n, sharex=True, sharey=True, 
                    gridspec_kw={"wspace":0})

colors = ["C0", "C3"]*2
linestyles = [*["solid"]*2, *["dotted"]*2]
transparencies = [*[.5]*2, *[1]*2]

sample_r_factor = 3

plt.suptitle(plot_title)
for i in range(n):
    axes[i].set_title(f'$\lambda$={wlen_chosen[i]:1.0f} nm')
    for j in range(len(E_labels)):
        axes[i].axvline(-r, color="grey", linestyle="dotted", label="", alpha=0.5)
        axes[i].axvline(r, color="grey", linestyle="dotted", label="", alpha=0.5)
        axes[i].axvline(-sample_r_factor*r, color="grey", linestyle="dotted", label="", alpha=0.5)
        axes[i].axvline(sample_r_factor*r, color="grey", linestyle="dotted", label="", alpha=0.5)
        axes[i].axhline(np.abs(E_chosen[j][i][np.argmin(np.abs( rvec[:,2] + sample_r_factor*r ) ),2]), 
                        color=colors[j], linestyle=linestyles[j], label="", 
                        alpha=0.2)
        # axes[i].axhline(np.abs(E_chosen[j][i][np.argmin(np.abs( rvec[:,2] - 2.5*r ) ),2]), 
        #                 color="grey", linestyle="dotted", label="", alpha=0.5)
        axes[i].plot(rvec[:,2], np.abs(E_chosen[j][i][:,2]),
                     color=colors[j], linestyle=linestyles[j],
                     alpha=transparencies[j], linewidth=2, label=E_labels[j])
    axes[i].set_xlabel(trs.choose(r"Position $Z$ [nm]", r"Posición $Z$ [nm]"))
    if i==0:
        axes[i].set_ylabel(trs.choose(r"Electric field $E_z$", "Campo eléctrico $E_z$"))
    if (material=="Au" and i==0) or (material=="Ag" and i==n-1):
        axes[i].legend()
    # axes[i].set_xlim(np.min(rvec), np.max(rvec))

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