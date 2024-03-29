#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 22:51:24 2021

@author: vall
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import use as use_backend
# from matplotlib.ticker import FormatStrFormatter
import matplotlib.pylab as plab
import os
import vmp_theory as vmt
import v_plot as vp
import v_save as vs
import vmp_materials as vml
import v_utilities as vu

english = False
trs = vu.BilingualManager(english=english)
vp.set_style()

syshome = vs.get_sys_home()

#%% PARAMETERS

r = 60 # nm
material = "Au"
wlen_range = np.array([300, 800])
wlen_chosen = [405, 532, 642]
submerged_index = 1.33 # Water

# E0_func = lambda n : np.array([[0,0,1] for i in range(n)])
E0 = np.array([0,0,1])

npoints = 500
nminpoints = 75

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

series = f"All{vml.recognize_material(submerged_index)}{material}Sphere{2*r:1.0f}"
home = vs.get_home()
if not os.path.isdir(os.path.join(home, folder)): 
    vs.new_dir(os.path.join(home, folder))
plot_file = lambda n : os.path.join(
    home, folder, f"All{vml.recognize_material(submerged_index)}Field{2*r:1.0f}" + n)
short_plot_file = lambda n : os.path.join(
        home, folder, f"All{vml.recognize_material(submerged_index)}" + n)

plot_title = trs.choose(f"{material} NP of {2*r:1.0f} nm diameter in {vml.recognize_material(submerged_index).lower()}",
                        f"NP de {material} de {2*r:1.0f} nm de diámetro en {vml.recognize_material(submerged_index, english).lower()}")

#%% INNER MEDIUM EPSILON

wlen = np.linspace(*wlen_range, npoints)

epsilon_func_meep_jc = vml.epsilon_function(material=material, 
                                            paper="JC", 
                                            reference="Meep")
epsilon_func_meep_r = vml.epsilon_function(material=material, 
                                           paper="R", 
                                           reference="Meep")
epsilon_func_exp_jc = vml.epsilon_function(material=material, 
                                           paper="JC", 
                                           reference="RIinfo")
epsilon_func_exp_r = vml.epsilon_function(material=material, 
                                          paper="R", 
                                          reference="RIinfo")

epsilon_meep_jc = np.array([epsilon_func_meep_jc(wl) for wl in wlen])
epsilon_meep_r = np.array([epsilon_func_meep_r(wl) for wl in wlen])
epsilon_exp_jc = np.array([epsilon_func_exp_jc(wl) for wl in wlen])
epsilon_exp_r = np.array([epsilon_func_exp_r(wl) for wl in wlen])

wlen_exp_jc_data, epsilon_exp_jc_data = vml.epsilon_data_from_file(material=material, 
                                                                   paper="JC", 
                                                                   reference="RIinfo")
wlen_exp_r_data, epsilon_exp_r_data = vml.epsilon_data_from_file(material=material, 
                                                                 paper="R", 
                                                                 reference="RIinfo")

epsilon_exp_jc_data = epsilon_exp_jc_data[min(wlen_range) <= wlen_exp_jc_data]
wlen_exp_jc_data = wlen_exp_jc_data[min(wlen_range) <= wlen_exp_jc_data]
epsilon_exp_jc_data = epsilon_exp_jc_data[wlen_exp_jc_data <= max(wlen_range)]
wlen_exp_jc_data = wlen_exp_jc_data[wlen_exp_jc_data <= max(wlen_range)]

epsilon_exp_r_data = epsilon_exp_r_data[min(wlen_range) <= wlen_exp_r_data]
wlen_exp_r_data = wlen_exp_r_data[min(wlen_range) <= wlen_exp_r_data]
epsilon_exp_r_data = epsilon_exp_r_data[wlen_exp_r_data <= max(wlen_range)]
wlen_exp_r_data = wlen_exp_r_data[wlen_exp_r_data <= max(wlen_range)]

#%% INNER MEDIUM REFRACTIVE INDEX

n_func_meep_jc = vml.n_function(material=material, 
                                      paper="JC", 
                                      reference="Meep")
n_func_meep_r = vml.n_function(material=material, 
                               paper="R", 
                               reference="Meep")
n_func_exp_jc = vml.n_function(material=material, 
                               paper="JC", 
                               reference="RIinfo")
n_func_exp_r = vml.n_function(material=material, 
                              paper="R", 
                              reference="RIinfo")

n_meep_jc = np.array([n_func_meep_jc(wl) for wl in wlen])
n_meep_r = np.array([n_func_meep_r(wl) for wl in wlen])
n_exp_jc = np.array([n_func_exp_jc(wl) for wl in wlen])
n_exp_r = np.array([n_func_exp_r(wl) for wl in wlen])

wlen_exp_jc_data, n_exp_jc_data = vml.n_data_from_file(material=material, 
                                                       paper="JC", 
                                                       reference="RIinfo")
wlen_exp_r_data, n_exp_r_data = vml.n_data_from_file(material=material, 
                                                     paper="R", 
                                                     reference="RIinfo")

n_exp_jc_data = n_exp_jc_data[min(wlen_range) <= wlen_exp_jc_data]
wlen_exp_jc_data = wlen_exp_jc_data[min(wlen_range) <= wlen_exp_jc_data]
n_exp_jc_data = n_exp_jc_data[wlen_exp_jc_data <= max(wlen_range)]
wlen_exp_jc_data = wlen_exp_jc_data[wlen_exp_jc_data <= max(wlen_range)]

n_exp_r_data = n_exp_r_data[min(wlen_range) <= wlen_exp_r_data]
wlen_exp_r_data = wlen_exp_r_data[min(wlen_range) <= wlen_exp_r_data]
n_exp_r_data = n_exp_r_data[wlen_exp_r_data <= max(wlen_range)]
wlen_exp_r_data = wlen_exp_r_data[wlen_exp_r_data <= max(wlen_range)]

#%% EPSILON PLOT

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
axes = fig.subplots(ncols=nplots, sharex=True, sharey=True, gridspec_kw={"wspace":0})

colors = ["C2", "C1"]*2
linestyles = [*["solid"]*2, *["dotted"]*2]
transparencies = [*[.5]*2, *[1]*2]

plt.suptitle(plot_title)
for ax, f, t in zip(axes, functions, titles):
    for wl, eps, l, col, lsy, tps in zip(wlen_long, epsilon, labels, 
                                         colors, linestyles, transparencies):
        ax.set_title(t)
        print(l, eps[:2])
        ax.axhline(linewidth=0.5, color="k")
        ax.plot(wl, f(eps), label=l, linewidth=2, alpha=tps, color=col, linestyle=lsy)
        ax.xaxis.set_label_text(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
        if ax == axes[0]:
            ax.set_ylabel(trs.choose(r"Electric Permitivity $\epsilon$", 
                                     r"Permitividad eléctrica $\epsilon$"))
        ax.legend()
        ax.set_xlim(*wlen_range)
   
plt.savefig(plot_file("Epsilon.png"))

#%% N PLOT

index = np.array([n_meep_jc, n_meep_r, n_exp_jc, n_exp_r])
wlen_long = np.array([*[wlen]*4])

functions = [np.abs, np.real, np.imag]
titles = trs.choose([r"Absolute value |$N$|", r"Real part $\Re$e$(N)$", r"Imaginary part $\Im$m$(N)$$"],
                    [r"Valor absoluto |$N$|", r"Parte real $\Re$e$(N)$", r"Parte imaginaria $\Im$m$(N)$"])
labels = ["Meep: Johnson & Christy", "Meep: Rakic", 
          "RIinfo: Johnson & Christy", "RIinfo: Rakic"]

nplots = len(functions)
fig = plt.figure(figsize=(nplots*6.4, 6.4))
axes = fig.subplots(ncols=nplots, sharex=True, sharey=True, gridspec_kw={"wspace":0})

colors = ["C2", "C1"]*2
linestyles = [*["solid"]*2, *["dotted"]*2]
transparencies = [*[.5]*2, *[1]*2]

plt.suptitle(plot_title)
for ax, f, t in zip(axes, functions, titles):
    for wl, ni, l, col, lsy, tps in zip(wlen_long, index, labels, 
                                         colors, linestyles, transparencies):
        ax.set_title(t)
        print(l, ni[:2])
        ax.axhline(linewidth=0.5, color="k")
        ax.plot(wl, f(ni), label=l, linewidth=2, alpha=tps, color=col, linestyle=lsy)
        ax.xaxis.set_label_text(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
        if ax == axes[0]:
            ax.yaxis.set_label_text(trs.choose(r"Electric Permitivity $\epsilon$", 
                                               r"Permitividad eléctrica $\epsilon$"))
        ax.legend()
        ax.set_xlim(*wlen_range)
            
plt.savefig(plot_file("N.png"))

#%% EPSILON AND N PLOT

plot_for_display = True

if plot_for_display: use_backend("Agg")

data = [[epsilon_exp_jc_data, epsilon_exp_jc,
         epsilon_exp_r_data, epsilon_exp_r],
        [n_exp_jc_data, n_exp_jc,
         n_exp_r_data, n_exp_r]]
wlen_long = [[wlen_exp_jc_data, wlen, wlen_exp_r_data, wlen]]*2

functions = [np.real, np.imag]
titles = trs.choose([r"Electric Permitivity $\epsilon = \epsilon_1 + i \epsilon_2$", 
                     "Refractive Complex Index $N = n + i \kappa$"], 
                    [r"Permitividad eléctrica $\epsilon = \epsilon_1 + i \epsilon_2$", 
                     "Índice de refracción complejo $N = n + i \kappa$"])
short_titles = trs.choose([r"Electric Permitivity $\epsilon$", 
                           "Refractive Complex Index $N$"], 
                          [r"Permitividad eléctrica $\epsilon$",  
                           "Índice de refracción complejo $N$"])
axis_labels = trs.choose([[r"Real part $\Re$e$(\epsilon) = \epsilon_1$", 
                           r"Imaginary part $\Im$m$(\epsilon) = \epsilon_2$"],
                          [r"Real part $\Re$e$(N) = n$", 
                           r"Imaginary part $\Im$m$(N) = \kappa$"]],
                         [[r"Parte real $\Re$e$(\epsilon) = \epsilon_1$", 
                           r"Parte imaginaria $\Im$m$(\epsilon) = \epsilon_2$"],
                          [r"Parte real $\Re$e$(N) = n$", 
                           r"Parte imaginaria $\Im$m$(N) = \kappa$"]])
labels = trs.choose(["JC Data", "JC Interpolation", 
                     "R Data", "R Interpolation"],
                    ["JC Datos", "JC Interpolación", 
                     "R Datos", "R Interpolación"])

nplots = len(functions)
fig, axes = plt.subplots(nrows=2, ncols=2, gridspec_kw={"hspace":0.25, "wspace":0})
if plot_for_display: fig.dpi = 200

colors = [*["C2"]*2, *["C1"]*2]
linestyles = ["none", "solid"]*2
markers = ["o", ""]*2
transparencies = [.4, 1, .4, 1]
zorders = [10, 10, 0, 0]

if not plot_for_display: plt.suptitle(plot_title)

for i in range(len(axes)):
    for j in range(len(axes)):
        axes[i,j].set_title(axis_labels[i][j])
        if i==0:
            axes[i,j].axhline(linewidth=1, color="k")
        for k in range(len(data[i])):
            axes[i,j].plot(wlen_long[i][k], 
                           functions[j](data[i][k]), 
                           label=labels[k], linewidth=2, markersize=7,
                           alpha=transparencies[k], color=colors[k], 
                           linestyle=linestyles[k], marker=markers[k],
                           markeredgewidth=0, zorder=zorders[k])
        if j==0: axes[i,j].set_ylabel(titles[i])
        else: axes[i,j].yaxis.set_ticklabels([])
        axes[i,j].set_xlim(*wlen_range)
        axes[i,j].set_xlabel(trs.choose(r"Wavelength $\lambda$ [nm]", 
                                        r"Longitud de onda $\lambda$ [nm]"))
    
ylims = ( np.min([axes[0,j].get_ylim() for j in range(len(axes[0,:]))]),
          np.max([axes[0,j].get_ylim() for j in range(len(axes[0,:]))])*1.2 )
for j in range(len(axes[0,:])): axes[0,j].set_ylim(ylims)
ylims = ( 0, np.max([np.abs(axes[1,j].get_ylim()) for j in range(len(axes[1,:]))]) )
for j in range(len(axes[1,:])): axes[1,j].set_ylim(ylims)

for j in range(len(axes[0,:])):
    box = axes[0,j].get_position()
    height = box.y1 - box.y0
    box.y0 = box.y0 + .10 * height
    box.y1 = box.y1 + .10 * height
    axes[0,j].set_position(box)

axes[-1,-1].legend()
fig.set_size_inches([12.02,  9.61])

plt.savefig(plot_file("EpsilonN.png"))

if plot_for_display: use_backend("Qt5Agg")

#%% CLAUSIUS-MOSETTI: POLARIZABILITY

alpha_cm_meep_r = vmt.alpha_Clausius_Mosotti(epsilon_meep_r, r,
                                            epsilon_ext=epsilon_ext)
alpha_cm_exp_jc = vmt.alpha_Clausius_Mosotti(epsilon_exp_jc, r,
                                            epsilon_ext=epsilon_ext)

alpha_cm = np.array([alpha_cm_exp_jc, alpha_cm_meep_r])

#%% KUWATA: POLARIZABILITY
   
alpha_k_meep_r = vmt.alpha_Kuwata(epsilon_meep_r, wlen, r,
                                  epsilon_ext=epsilon_ext)
alpha_k_exp_jc = vmt.alpha_Kuwata(epsilon_exp_jc, wlen, r,
                                  epsilon_ext=epsilon_ext)

alpha_k = np.array([alpha_k_exp_jc, alpha_k_meep_r])

#%% PLOT POLARIZABILITY

plot_for_display = True
with_legend = False

functions = [np.abs, np.real, np.imag]
titles = trs.choose([r"Absolute value |$\alpha$|", r"Real part $\Re$e$(\alpha$)", r"Imaginary part $\Im$m$(\alpha$)$"],
                    [r"Valor absoluto |$\alpha$|", r"Parte real $\Re$e$(\alpha)$", r"Parte imaginaria $\Im$m$(\alpha$)"])
wlen_alpha = [*[wlen]*4]
alpha = [*alpha_cm, *alpha_k]
labels = ["J&C Claussius-Mosotti", "R Claussius-Mosotti", "J&C Kuwata", "R Kuwata"]
# labels = ["Meep R: Claussius-Mosotti", "RIinfo J&C: Claussius-Mosotti", 
          # "Meep R: Kuwata", "RIinfo J&C: Kuwata"]

nplots = len(functions)
        
if plot_for_display: use_backend("Agg")

fig = plt.figure()
subplot_grid = fig.add_gridspec(1, 7, wspace=0)
axes = [fig.add_subplot(subplot_grid[0,:2]),
          fig.add_subplot(subplot_grid[0,3:5]),
        fig.add_subplot(subplot_grid[0,5:7])]
if plot_for_display: fig.dpi = 200

colors = ["C2", "C1"]*2
linestyles = [*["solid"]*2, *["dotted"]*2]
transparencies = [*[.5]*2, *[1]*2]

if plot_for_display: plt.suptitle(plot_title)
for ax, f, t in zip(axes, functions, titles):
    for wl, al, l, col, lsy, trp in zip(wlen_alpha, alpha, labels,
                                        colors, linestyles, transparencies):
        ax.set_title(t)
        ax.axhline(linewidth=0.5, color="k")
        # if submerged_index==1.33:
        #     ax.plot(wl, f(al), label=l, color="cornflowerblue", linestyle="solid", alpha=0.3, linewidth=4)
        ax.plot(wl, f(al), label=l, color=col, linestyle=lsy, alpha=trp, linewidth=2)
        ax.set_xlabel(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
        ax.set_ylabel(trs.choose("Polarizability", "Polarizabilidad") + r" $\alpha$ [nm$^3$]")
        if ax == axes[-1]:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
        ax.set_xlim(*wlen_range)

if with_legend: axes[0].legend()

# ylims = ( -np.max([np.abs(axes[-1].get_ylim()), np.abs(axes[-2].get_ylim())]), 
#            np.max([np.abs(axes[-1].get_ylim()), np.abs(axes[-2].get_ylim())]) )
ylims = ( 0, np.max([np.abs(axes[-1].get_ylim()), np.abs(axes[-2].get_ylim())]) )
axes[-1].set_ylim(ylims)
axes[-2].set_ylim(ylims)
axes[0].set_ylim( 0 , axes[0].set_ylim()[-1] )
    
fig.set_size_inches([12.8, 4.95])

plt.savefig(plot_file("Polarizability.png"))

if plot_for_display: use_backend("Qt5Agg")

#%% DIPOLAR APROXIMATION: ELECTRIC FIELD EZ IN Z

rvec = np.zeros((npoints, 3))
rvec[:,2] = np.linspace(-4*r, 4*r, npoints)

E_cm_meep_r_chosen = []
E_k_meep_r_chosen = []
E_cm_exp_jc_chosen = []
E_k_exp_jc_chosen = []
for wl in wlen_chosen:
    e_meep_r_chosen = epsilon_func_meep_r(wl)
    a_cm_meep_r_chosen = vmt.alpha_Clausius_Mosotti(e_meep_r_chosen, r, 
                                                   epsilon_ext=epsilon_ext)
    a_k_meep_r_chosen = vmt.alpha_Kuwata(e_meep_r_chosen, wl, r,
                                        epsilon_ext=epsilon_ext)
    e_exp_jc_chosen = epsilon_func_exp_jc(wl)
    a_cm_exp_jc_chosen = vmt.alpha_Clausius_Mosotti(e_exp_jc_chosen, r,
                                                   epsilon_ext=epsilon_ext)
    a_k_exp_jc_chosen = vmt.alpha_Kuwata(e_exp_jc_chosen, wl, r,
                                        epsilon_ext=epsilon_ext)
    E_cm_meep_r_chosen.append(
        np.array([vmt.E(e_meep_r_chosen, a_cm_meep_r_chosen, 
                       E0, rv, r, epsilon_ext=epsilon_ext) for rv in rvec]))
    E_k_meep_r_chosen.append(
        np.array([vmt.E(e_meep_r_chosen, a_k_meep_r_chosen, 
                       E0, rv, r, epsilon_ext=epsilon_ext) for rv in rvec]))
    E_cm_exp_jc_chosen.append(
        np.array([vmt.E(e_exp_jc_chosen, a_cm_exp_jc_chosen, 
                       E0, rv, r, epsilon_ext=epsilon_ext) for rv in rvec]))
    E_k_exp_jc_chosen.append(
        np.array([vmt.E(e_exp_jc_chosen, a_k_exp_jc_chosen, 
                       E0, rv, r, epsilon_ext=epsilon_ext) for rv in rvec]))

E_chosen = [E_cm_exp_jc_chosen, E_cm_meep_r_chosen,
            E_k_exp_jc_chosen, E_k_meep_r_chosen]
E_labels = ["J&C Claussius-Mosotti", "R Claussius-Mosotti", "J&C Kuwata", "R Kuwata"]
# E_labels = ["Meep R: Claussius-Mosotti", "RIinfo J&C: Claussius-Mosotti", 
#             "Meep R: Kuwata", "RIinfo J&C: Kuwata"]

# E_plane_cm_meep = np.array([E(epsilon_meep, alpha_cm_meep, E0, rv) for rv in rvec])
# First index: position
# Second index: wavelength
# Third index: direction

#%% FIELD VS WAVELENGTH

plot_for_display = False

if plot_for_display: use_backend("Agg")

n = len(wlen_chosen)
fig = plt.figure(figsize=(n*6.4, 6.4))
axes = fig.subplots(ncols=n, sharex=True, sharey=True, 
                    gridspec_kw={"wspace":0})
if plot_for_display: fig.dpi = 200

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

if plot_for_display: use_backend("Qt5Agg")

#%% COMPLEX FIELD FOR SINGLE WAVELENGTH

plot_for_display = True
with_legend = False

wlen_single = 532
wlen_index = wlen_chosen.index(wlen_single)

sample_r_factor = 3

E_single = [E_cm_exp_jc_chosen[wlen_index], E_cm_meep_r_chosen[wlen_index], 
            E_k_exp_jc_chosen[wlen_index], E_k_meep_r_chosen[wlen_index]]

colors = ["C2", "C1"]*2
linestyles = [*["solid"]*2, *["dotted"]*2]
transparencies = [*[.5]*2, *[1]*2]

functions = [np.abs, np.real, np.imag]
titles = trs.choose([r"Absolute value |$E_z$|", r"Real part $\Re$e$(E_z)$", r"Imaginary part $\Im$m$(E_z)$"],
                    [r"Valor absoluto |$E_z$|", r"Parte real $\Re$e$(E_z)$", r"Parte imaginaria $\Im$m$(E_z)$"])
wlen_alpha = [*[wlen]*4]
alpha = [*alpha_cm, *alpha_k]
labels = ["J&C Claussius-Mosotti", "R Claussius-Mosotti", "J&C Kuwata", "R Kuwata"]
# labels = ["Meep R: Claussius-Mosotti", "RIinfo J&C: Claussius-Mosotti", 
#           "Meep R: Kuwata", "RIinfo J&C: Kuwata"]

if plot_for_display: use_backend("Agg")

nplots = len(functions)
fig = plt.figure()
subplot_grid = fig.add_gridspec(1, 7, wspace=0)
axes = [fig.add_subplot(subplot_grid[0,:2]),
          fig.add_subplot(subplot_grid[0,3:5]),
        fig.add_subplot(subplot_grid[0,5:7])]
if plot_for_display: fig.dpi = 200

plt.suptitle(plot_title)
lines = []
for ax, f, t in zip(axes, functions, titles):
    for field, l, col, lsy, trp in zip(E_single, labels,
                                        colors, linestyles, transparencies):
        ax.set_title(t)
        ax.axvline(-r, color="grey", linestyle="dotted", alpha=0.5)
        ax.axvline(r, color="grey", linestyle="dotted", alpha=0.5)
        if ax != axes[0]: ax.axhline(color="k", linewidth=.5)
        line, = ax.plot(rvec[:,2], f(field[:,2]), label=l, color=col, 
                        linestyle=lsy, alpha=trp, linewidth=2)
        if ax == axes[0]: 
            lines.append(line)
            ax.axhline(np.max(f(field[:,2])), color=col, linestyle=lsy, 
                       alpha=trp-.15, linewidth=2)
        ax.set_xlabel(trs.choose(r"Position $Z$ [nm]", r"Posición $Z$ [nm]"))
        ax.yaxis.set_label_text(trs.choose(r"Electric field $E_z(x=y=0)$", "Campo eléctrico $E_z(x=y=0)$"))
        if ax == axes[-1]:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
        ax.set_xlim(np.min(rvec), np.max(rvec))
if with_legend and plot_for_display: 
    ax.legend(lines, [l.get_label() for l in lines], frameon=True, framealpha=1)
elif with_legend: ax.legend(lines, [l.get_label() for l in lines])

ylims = ( np.min([axes[-1].get_ylim(), axes[-2].get_ylim()]),
          np.max([axes[-1].get_ylim(), axes[-2].get_ylim()]) )
axes[-1].set_ylim(ylims)
axes[-2].set_ylim(ylims)
axes[0].set_ylim( 0 , axes[0].set_ylim()[-1] )
    
fig.set_size_inches([12.8, 4.95])
if with_legend: plt.savefig(plot_file("FieldComplexLegend.png"))
else: plt.savefig(plot_file("FieldComplex.png"))

if plot_for_display: use_backend("Qt5Agg")

#%% DIPOLAR APROXIMATION: ELECTRIC FIELD EZ IN YZ PLANE

# x = np.zeros(npoints)
y = np.linspace(-2.5*r, 2.5*r, nminpoints)
z = np.linspace(-2.5*r, 2.5*r, nminpoints)
# x, y, z = np.meshgrid(x, y, z)
Y, Z = np.meshgrid(y, z)

rvec = np.zeros((len(y) * len(z), 3))
rvec[:,1] = Y.reshape( len(y) * len(z) )
rvec[:,2] = Z.reshape( len(y) * len(z) )

E_cm_meep_r_chosen = []
E_k_meep_r_chosen = []
E_cm_exp_jc_chosen = []
E_k_exp_jc_chosen = []
for wl in wlen_chosen:
    e_meep_r_chosen = epsilon_func_meep_r(wl)
    a_cm_meep_r_chosen = vmt.alpha_Clausius_Mosotti(e_meep_r_chosen, r, 
                                                   epsilon_ext=epsilon_ext)
    a_k_meep_r_chosen = vmt.alpha_Kuwata(e_meep_r_chosen, wl, r,
                                        epsilon_ext=epsilon_ext)
    e_exp_jc_chosen = epsilon_func_exp_jc(wl)
    a_cm_exp_jc_chosen = vmt.alpha_Clausius_Mosotti(e_exp_jc_chosen, r,
                                                   epsilon_ext=epsilon_ext)
    a_k_exp_jc_chosen = vmt.alpha_Kuwata(e_exp_jc_chosen, wl, r,
                                        epsilon_ext=epsilon_ext)
    E_cm_meep_r_chosen.append(
        np.array([vmt.E(e_meep_r_chosen, a_cm_meep_r_chosen, 
                       E0, rv, r, epsilon_ext=epsilon_ext)[-1] 
                  for rv in rvec]).reshape(len(y), len(z)))
    E_k_meep_r_chosen.append(
        np.array([vmt.E(e_meep_r_chosen, a_k_meep_r_chosen, 
                       E0, rv, r, epsilon_ext=epsilon_ext)[-1]
                  for rv in rvec]).reshape(len(y), len(z)))
    E_cm_exp_jc_chosen.append(
        np.array([vmt.E(e_exp_jc_chosen, a_cm_exp_jc_chosen, 
                       E0, rv, r, epsilon_ext=epsilon_ext)[-1]
                  for rv in rvec]).reshape(len(y), len(z)))
    E_k_exp_jc_chosen.append(
        np.array([vmt.E(e_exp_jc_chosen, a_k_exp_jc_chosen, 
                       E0, rv, r, epsilon_ext=epsilon_ext)[-1]
                  for rv in rvec]).reshape(len(y), len(z)))
# First index: wavelength
# Second index: position
# Third index: direction


# E_chosen = [E_cm_exp_jc_chosen, E_cm_meep_r_chosen,
#             E_k_exp_jc_chosen, E_k_meep_r_chosen]
# E_labels = ["J&C Claussius-Mosotti", "R Claussius-Mosotti", "J&C Kuwata", "R Kuwata"]

E_chosen = [E_cm_exp_jc_chosen, E_cm_meep_r_chosen,
            E_k_exp_jc_chosen, E_k_meep_r_chosen]
E_labels = ["J&C Claussius-Mosotti", "R Claussius-Mosotti", "J&C Kuwata", "R Kuwata"]

# E_plane_cm_meep = np.array([E(epsilon_meep, alpha_cm_meep, E0, rv) for rv in rvec])
# First index: position
# Second index: wavelength
# Third index: direction

#%% FIELD VS WAVELENGTH

plot_for_display = True

n = len(wlen_chosen)
if n <= 3:
    subfig_size = 6
if n <= 6:
    subfig_size = 4.15
else:
    subfig_size = 3.2
    
if plot_for_display: use_backend("Agg")

fig = plt.figure(figsize=(n*subfig_size, subfig_size))
axes = fig.subplots(ncols=n, nrows=1)
if plot_for_display: fig.dpi = 200

# lims = [0, np.max([np.max([ np.max( np.power( np.abs(E_chosen[j][i]) , 2)/2 ) for j in range(len(E_chosen))]) for i in range(len(wlen_chosen))])]
# [0, 15.310595163068891]
lims = [0, np.max([ np.max( np.power( np.abs(E_k_meep_r_chosen[i]) , 2)/2 ) for i in range(len(wlen_chosen))])]
# lims = [0, 11.203576619970478]

plt.suptitle(plot_title)
for i in range(len(wlen_chosen)):
    axes[i].set_title(f'$\lambda$={wlen_chosen[i]:1.0f} nm')
    ims = axes[i].imshow(np.power(np.abs(E_cm_meep_r_chosen[i]), 2).T / 2,
                         cmap=plab.cm.jet, interpolation='spline36', 
                         vmin=lims[0], vmax=lims[1],
                         extent=[np.min(rvec), np.max(rvec),
                                 np.min(rvec), np.max(rvec)])
    # for j in range(len(E_chosen)):
    #     ims = axes[i].imshow(np.power(np.abs(E_cm_meep_r_chosen[j][i]), 2).T / 2,
    #                          cmap=plab.cm.jet, #interpolation='spline36', 
    #                          vmin=lims[0], vmax=lims[1],
    #                          extent=[np.min(rvec), np.max(rvec),
    #                                  np.min(rvec), np.max(rvec)])
    axes[i].grid(False)
    axes[i].set_xlabel(trs.choose("Position $Z$ [nm]", "Posición $Z$ [nm]"))
    if i==0:
        axes[i].set_ylabel(trs.choose("Position $Y$ [nm]", "Posición $Y$ [nm]"))
    if i==len(wlen_chosen)-1:
        cax = axes[i].inset_axes([1.04, 0, 0.07, 1], #[1.04, 0.2, 0.05, 0.6], 
                                 transform=axes[i].transAxes)
        cbar = fig.colorbar(ims, ax=axes[i], cax=cax)
        cbar.set_label(trs.choose(r"Electric field intensity $\langle\,|E_z|^2\,\rangle(y=z=0)$",
                                  r"Intensidad del campo eléctrico $\langle\,|E_z|^2\,\rangle(y=z=0)$"))
        
plt.savefig(plot_file("FieldIntensity.png"))

if plot_for_display: use_backend("Qt5Agg")

#%% SCATTERING CROSS SECTION

scatt_meep_jc = vml.sigma_scatt_meep(r, material, "JC", 
                                      wlen, surrounding_index=submerged_index)
scatt_meep_r = vml.sigma_scatt_meep(r, material, "R", 
                                    wlen, surrounding_index=submerged_index)
scatt_exp_jc = vmt.sigma_scatt_Mie(r, wlen, inner_N=n_exp_jc,
                                  surrounding_N=submerged_index)

scatt_k_exp_jc = vmt.sigma_scatt_dipolar(r, wlen, alpha_k_exp_jc, surrounding_N=submerged_index)
scatt_cm_exp_jc = vmt.sigma_scatt_dipolar(r, wlen, alpha_cm_exp_jc, surrounding_N=submerged_index)
scatt_k_meep_r = vmt.sigma_scatt_dipolar(r, wlen, alpha_k_meep_r, surrounding_N=submerged_index)
scatt_cm_meep_r = vmt.sigma_scatt_dipolar(r, wlen, alpha_cm_meep_r, surrounding_N=submerged_index)

#%% PLOT SCATTERING THEORY: FULL COMPARISON

scatt = [scatt_exp_jc, scatt_cm_exp_jc, scatt_k_exp_jc,
         scatt_meep_r, scatt_cm_meep_r, scatt_k_meep_r]
wlen_long = [wlen]*6
scatt_labels = ["JC Mie", "JC CM Dipolar", "JC K Dipolar",
                "R Mie", "R CM Dipolar", "R K Dipolar"]

plot_for_display = True
split_plot = False

colors = [*["C2"]*3, *["C1"]*3]
linestyles = ["solid", "dashed", "dotted"]*2
transparencies = [.5, .75, 1]*2

if plot_for_display: use_backend("Agg")

fig = plt.figure()
if split_plot:
    axes = fig.subplots(nrows=2, sharex=True, gridspec_kw={"wspace":0, "hspace":0})
    second_axes = [plt.twinx(ax) for ax in axes]
else:
    first_ax = fig.add_subplot()
    second_ax = plt.twinx(first_ax)
    axes = [first_ax]*2
    second_axes = [second_ax]*2
def get_ax(k):
    if k<len(wlen_long)/2: return axes[0]
    else: return axes[1]
if plot_for_display: fig.dpi = 200

plt.suptitle(plot_title)
for k in range(len(wlen_long)):
    ax = get_ax(k)
    ax.plot(wlen_long[k], scatt[k], linewidth=2, linestyle=linestyles[k],
            color=colors[k], alpha=transparencies[k], label=scatt_labels[k])
ylims = (0, np.max([axes[0].get_ylim()[1], axes[0].get_ylim()[1]]))
for ax, second_ax in zip(axes, second_axes):
    ax.set_ylim(ylims)
    second_ax.set_ylim(np.array(ax.get_ylim()) / (np.pi * r**2))
    ax.legend()
    ax.set_ylabel(trs.choose("Scattering Cross Section\n"+r"$\sigma_{disp}$ [nm$^2$]",
                             "Sección eficaz\nde dispersión\n"+r"$\sigma_{disp}$ [nm$^2$]"))
    second_ax.set_ylabel(trs.choose("Scattering Efficiency\n"+r"$C_{disp}$",
                                    "Eficiencia\nde dispersión\n"+r"$C_{disp}$"))
ax.set_xlabel(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))

# fig.set_size_inches([9.06, 4.8])
if split_plot: 
    fig.set_size_inches([10.01,  5.68])
    plt.savefig(plot_file("SplitScattering.png"))
else: 
    fig.set_size_inches([10.01,  4.99])
    plt.savefig(plot_file("Scattering.png"))

if plot_for_display: use_backend("Qt5Agg")

#%% PLOT SCATTERING THEORY: MINIMUM COMPARISON

# scatt = [scatt_exp_jc, scatt_meep_jc, scatt_meep_r]
# wlen_long = [wlen]*3
# scatt_labels = [trs.choose("JC Data", "JC Datos"), 
#                 "JC Drude-Lorentz", "R Drude-Lorentz"]

scatt = [scatt_exp_jc, scatt_meep_r]
wlen_long = [wlen]*2
scatt_labels = [trs.choose("JC Data", "JC Datos"), 
                "R Drude-Lorentz"]

# colors = ["C2", "C2", "C1"]
# linestyles = ["dashdot", "solid", "solid"]
# transparencies = [1, .5, .5]

colors = ["C2", "C1"]
linestyles = ["solid", "solid"]
transparencies = [.5, .5]

plot_for_display = True

if plot_for_display: use_backend("Agg")

fig = plt.figure()
ax = fig.add_subplot()
second_ax = plt.twinx(ax)
if plot_for_display: fig.dpi = 200

plt.suptitle(plot_title)
for k in range(len(wlen_long)):
    ax.plot(wlen_long[k], scatt[k], linewidth=2, linestyle=linestyles[k],
            color=colors[k], alpha=transparencies[k], label=scatt_labels[k])
ax.set_ylim( (0, ax.get_ylim()[1]) )
second_ax.set_ylim(np.array(ax.get_ylim()) / (np.pi * r**2))

ax.set_ylabel(trs.choose("Scattering Cross Section\n"+r"$\sigma_{disp}$ [nm$^2$]",
                         "Sección eficaz\nde dispersión\n"+r"$\sigma_{disp}$ [nm$^2$]"))
second_ax.set_ylabel(trs.choose("Scattering Efficiency\n"+r"$C_{disp}$",
                                "Eficiencia\nde dispersión\n"+r"$C_{disp}$"))
ax.set_xlabel(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))

ax.legend()

# fig.set_size_inches([9.06, 4.8])

fig.set_size_inches([10.01,  3.83])
plt.tight_layout()
plt.savefig(plot_file("ScatteringSimple.png"))

if plot_for_display: use_backend("Qt5Agg")