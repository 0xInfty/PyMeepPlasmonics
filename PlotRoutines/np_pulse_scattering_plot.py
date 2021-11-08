#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis of scattering of a single Au sphere under incidence of a visible planewave pulse.

See also
--------
Routines/u_np_scattering
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import vmp_utilities as vmu
import v_plot as vp
import v_save as vs
import v_utilities as vu

vp.set_style()

#%% PARAMETERS

"""
series = "AllVacRes14"
folder = "Scattering/AuSphere/AlgorithmTest/RAM/TestRAMRes/AllVac"

series = "AllVacNear2FarTrueRes2"
folder = "Scattering/AuSphere/AlgorithmTest/Near2Far/AllVac"

near2far = False
english = False
"""

#%% 

def mid_plots_np_scattering(series, folder, english=False):
    
    #%% SETUP
    
    # Computation
    pm = vmu.ParallelManager()
    
    # Saving directories
    sa = vmu.SavingAssistant(series, folder)
    
    trs = vu.BilingualManager(english=english)
    
    #%% LOAD DATA
    
    flux_data = np.loadtxt(sa.file("MidFlux.txt"))
    params = vs.retrieve_footer(sa.file("MidFlux.txt"))
    
    #%% EXTRACT SOME PARAMETERS
    
    from_um_factor = params["from_um_factor"]
    r = params["r"]
    material = params["material"]
    
    #%% PLOT CONFIGURATION
    
    plot_title_ending = trs.choose(
        f" of {2 * 1e3 * from_um_factor * r:.0f} nm {material} Spherical NP",
        f" de NP esférica de {material} y diámetro {2 * 1e3 * from_um_factor * r:.0f} nm")
    
    #%% PLOT FLUX FOURIER MID DATA
    
    header = trs.choose([r"Wavelength $\lambda$ [nm]", 
                         "Flux $X_{10}$ [ua]", "Flux $X_{20}$ [ua]",
                         "Flux $Y_{10}$ [ua]", "Flux $Y_{20}$ [ua]",
                         "Flux $Z_{10}$ [ua]", "Flux $Z_{20}$ [ua]"],
                        [r"Longitud de onda $\lambda$ [nm]", 
                         "Flujo $X_{10}$ [ua]", "Flujo $X_{20}$ [ua]",
                         "Flujo $Y_{10}$ [ua]", "Flujo $Y_{20}$ [ua]",
                         "Flujo $Z_{10}$ [ua]", "Flujo $Z_{20}$ [ua]"])
    
    if pm.assign(1):
        ylims = (np.min(flux_data[:,1:]), np.max(flux_data[:,1:]))
        ylims = (ylims[0]-.1*(ylims[1]-ylims[0]),
                  ylims[1]+.1*(ylims[1]-ylims[0]))
        
        fig, ax = plt.subplots(3, 2, sharex=True, gridspec_kw={"hspace":0, "wspace":0})
        fig.subplots_adjust(hspace=0, wspace=.05)
        plt.suptitle(trs.choose("Incident Flux", "Flujo incidente") + plot_title_ending)
        for a in ax[:,1]:
            a.yaxis.tick_right()
            a.yaxis.set_label_position("right")
        for a, h in zip(np.reshape(ax, 6), header[1:]):
            a.set_ylabel(h)
        
        for d, a in zip(flux_data[:,1:].T, np.reshape(ax, 6)):
            a.plot(flux_data[:,0], d)
            a.set_ylim(*ylims)
        ax[-1,0].set_xlabel(trs.choose(r"Wavelength $\lambda$ [nm]",
                                       r"Longitud de onda $\lambda$ [nm]"))
        ax[-1,1].set_xlabel(trs.choose(r"Wavelength $\lambda$ [nm]",
                                       r"Longitud de onda $\lambda$ [nm]"))
    
        plt.tight_layout()
        plt.savefig(sa.file("MidFlux.png"))

#%%

def plots_np_scattering(series, folder, near2far=False, 
                        separate_simulations_needed=False, english=False):
    
    #%% SETUP
    
    # Computation
    pm = vmu.ParallelManager()
    
    # Saving directories
    sa = vmu.SavingAssistant(series, folder)
    
    trs = vu.BilingualManager(english=english)
    
    #%% LOAD DATA
    
    flux_data = np.loadtxt(sa.file("BaseResults.txt"))
    data = np.loadtxt(sa.file("Results.txt"))
    if near2far: 
        near2far_data = np.loadtxt(sa.file("Near2FarResults.txt"))
        if near2far_data.shape[0] < near2far_data.shape[1]:
            near2far_data = near2far_data.T
    
    params = vs.retrieve_footer(sa.file("Results.txt"))
    
    #%% EXTRACT SOME PARAMETERS
    
    from_um_factor = params["from_um_factor"]
    r = params["r"]
    material = params["material"]
    
    submerged_index = params["submerged_index"]
    surface_index = params["surface_index"]
    if submerged_index!=1 and surface_index==1: surface_index = submerged_index
    
    wlen_range = params["wlen_range"]
    wlens = data[:,0]
    
    if near2far:
        nazimuthal = params["nazimuthal"]
        npolar = params["npolar"]
        nfreq = params["nfreq"]
        
        azimuthal_angle = np.arange(0, 2 + 2/nazimuthal, 2/nazimuthal) # in multiples of pi
        polar_angle = np.arange(0, 1 + 1/npolar, 1/npolar)
        
        near2far_shape = (azimuthal_angle.shape[0], polar_angle.shape[0], nfreq)
        poynting_x = near2far_data[:,0].reshape(near2far_shape)
        poynting_y = near2far_data[:,1].reshape(near2far_shape)
        poynting_z = near2far_data[:,2].reshape(near2far_shape)
        poynting_r = near2far_data[:,3].reshape(near2far_shape)
    
    #%% PLOT CONFIGURATION
    
    plot_title_ending = trs.choose(
        f" of {2 * 1e3 * from_um_factor * r:.0f} nm {material} Spherical NP",
        f" de NP esférica de {material} y diámetro {2 * 1e3 * from_um_factor * r:.0f} nm")
    
    #%% PLOT FLUX FOURIER FINAL DATA
    
    header = trs.choose([r"Wavelength $\lambda$ [nm]", 
                         "Flux $X_{1}$ [au]", "Flux $X_{2}$ [au]",
                         "Flux $Y_{1}$ [au]", "Flux $Y_{2}$ [au]",
                         "Flux $Z_{1}$ [au]", "Flux $Z_{2}$ [au]"],
                        [r"Longitud de onda $\lambda$ [nm]", 
                         "Flujo $X_{1}$ [ua]", "Flujo $X_{2}$ [ua]",
                         "Flujo $Y_{1}$ [ua]", "Flujo $Y_{2}$ [ua]",
                         "Flujo $Z_{1}$ [ua]", "Flujo $Z_{2}$ [ua]"])
    
    if pm.assign(1):
        ylims = (np.min(flux_data[:,1:]), np.max(flux_data[:,1:]))
        ylims = (ylims[0]-.1*(ylims[1]-ylims[0]),
                  ylims[1]+.1*(ylims[1]-ylims[0]))
        
        fig, ax = plt.subplots(3, 2, sharex=True, gridspec_kw={"hspace":0, "wspace":0})
        fig.subplots_adjust(hspace=0, wspace=.05)
        plt.suptitle(trs.choose("Scattered Flux", "Flujo dispersado") + plot_title_ending)
        for a in ax[:,1]:
            a.yaxis.tick_right()
            a.yaxis.set_label_position("right")
        for a, h in zip(np.reshape(ax, 6), header[1:]):
            a.set_ylabel(h)
        
        for d, a in zip(flux_data[:,1:].T, np.reshape(ax, 6)):
            a.plot(flux_data[:,0], d)
            a.set_ylim(*ylims)
        ax[-1,0].set_xlabel(trs.choose(r"Wavelength $\lambda$ [nm]",
                                       r"Longitud de onda $\lambda$ [nm]"))
        ax[-1,1].set_xlabel(trs.choose(r"Wavelength $\lambda$ [nm]",
                                       r"Longitud de onda $\lambda$ [nm]"))
        
        plt.tight_layout()
        plt.savefig(sa.file("FinalFlux.png"))
        
    #%% PLOT SCATTERING: ALL TOGETHER
    
    if pm.assign(0) and surface_index==submerged_index:
        plt.figure()
        plt.title(trs.choose('Scattering', 'Dispersión') + plot_title_ending)
        plt.plot(data[:,0], data[:,1], 'b.-', label='MEEP', alpha=.6)
        plt.plot(data[:,0], data[:,-1], 'r.-', alpha=.6,
                 label=trs.choose('Mie Theory', 'Teoría de Mie'))
        plt.xlabel(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
        plt.ylabel(trs.choose(r'Scattering efficiency $C_{scat}$ [σ/πr$^{2}$]', 
                              r'Eficiencia de scattering $C_{disp}$ [σ/πr$^{2}$]'))
        plt.legend()
        plt.tight_layout()
        plt.savefig(sa.file("Comparison.png"))
    
    #%% PLOT SCATTERING: SEPARATE
    
    if pm.assign(1):
        plt.figure()
        plt.title(trs.choose('Scattering', 'Dispersión') + plot_title_ending)
        plt.plot(data[:,0], data[:,1],'b.-',label='Meep',alpha=.6)
        plt.xlabel(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
        plt.ylabel(trs.choose(r'Scattering efficiency $C_{scat}$ [σ/πr$^{2}$]', 
                              r'Eficiencia de scattering $C_{disp}$ [σ/πr$^{2}$]'))
        plt.legend()
        plt.tight_layout()
        plt.savefig(sa.file("Meep.png"))
        
    if pm.assign(0):
        plt.figure()
        plt.title(trs.choose('Scattering', 'Dispersión') + plot_title_ending)
        plt.plot(data[:,0], data[:,-1],'r.-',alpha=.6,
                 label=trs.choose('Theory', 'Teoría'))
        plt.xlabel(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
        plt.ylabel(trs.choose(r'Scattering efficiency $C_{scat}$ [σ/πr$^{2}$]', 
                              r'Eficiencia de scattering $C_{disp}$ [σ/πr$^{2}$]'))
        plt.legend()
        plt.tight_layout()
        plt.savefig(sa.file("Theory.png"))
        
    #%% PLOT ANGULAR PATTERN IN 3D
        
    if near2far and pm.assign(1) and separate_simulations_needed:
    
        freq_index = np.argmin(np.abs(wlens - np.mean(wlen_range) * 1e3 * from_um_factor))
    
        fig = plt.figure()
        plt.suptitle(trs.choose('Angular Pattern', 'Patrón angular') + plot_title_ending)
        ax = fig.add_subplot(1,1,1, projection='3d')
        ax.plot_surface(
            poynting_x[:,:,freq_index], 
            poynting_y[:,:,freq_index], 
            poynting_z[:,:,freq_index], cmap=plt.get_cmap('jet'), 
            linewidth=1, antialiased=False, alpha=0.5)
        ax.set_xlabel(trs.choose("Poynting Vector\n"+r"$P_x$ [au]",
                                 "Vector de Poynting\n"+r"$P_x$ [ua]"))
        ax.set_ylabel(trs.choose("Poynting Vector\n"+r"$P_y$ [au]",
                                 "Vector de Poynting\n"+r"$P_y$ [ua]"))
        ax.set_zlabel(trs.choose("Poynting Vector\n"+r"$P_z$ [au]",
                                 "Vector de Poynting\n"+r"$P_z$ [ua]"))
        
        plt.savefig(sa.file("AngularPattern.png"))
        
    #%% PLOT ANGULAR PATTERN PROFILE FOR DIFFERENT POLAR ANGLES
        
    if near2far and pm.assign(0) and separate_simulations_needed:
        
        freq_index = np.argmin(np.abs(wlens - np.mean(wlen_range) * 1e3 * from_um_factor))
        index = [list(polar_angle).index(alpha) for alpha in [0, .25, .5, .75, 1]]
        
        fig = plt.figure()
        plt.suptitle(trs.choose('Angular Pattern', 'Patrón angular') + plot_title_ending)
        ax_plain = plt.axes()
        for i in index:
            ax_plain.plot(poynting_x[:,i,freq_index], 
                          poynting_y[:,i,freq_index], 
                          ".-", label=rf"$\theta$ = {polar_angle[i]:.2f} $\pi$")
        plt.legend()
        ax_plain.set_xlabel(trs.choose("Poynting Vector "+r"$P_x$ [au]",
                                       "Vector de Poynting "+r"$P_x$ [ua]"))
        ax_plain.set_ylabel(trs.choose("Poynting Vector "+r"$P_y$ [au]",
                                       "Vector de Poynting "+r"$P_y$ [ua]"))
        ax_plain.set_aspect("equal")
        
        for ax in fig.axes:
            box = ax.get_position()
            width = box.x1 - box.x0
            box.x0, box.x1 = (box.x0 - .15*width, box.x1 - .15*width)
            ax.set_position(box)
        plt.legend(bbox_to_anchor=(1.4, 0.5), loc="center right", frameon=False)
        
        fig.set_size_inches([7.13, 4.8 ])
        plt.savefig(sa.file("AngularPolar.png"))
        
    #%% PLOT ANGULAR PATTERN PROFILE FOR DIFFERENT AZIMUTHAL ANGLES
        
    if near2far and pm.assign(1) and separate_simulations_needed:
        
        freq_index = np.argmin(np.abs(wlens - np.mean(wlen_range) * 1e3 * from_um_factor))
        index = [list(azimuthal_angle).index(alpha) for alpha in [0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2]]
        colors = plt.get_cmap("spring")( np.linspace(0,1,len(azimuthal_angle)) )
        
        fig = plt.figure()
        plt.suptitle(trs.choose('Angular Pattern', 'Patrón angular') + plot_title_ending)
                     
        ax_plain = plt.axes()
        for i in index:
            ax_plain.plot(poynting_x[i,:,freq_index], 
                          poynting_z[i,:,freq_index], 
                          ".-", color=colors[i], alpha=0.7,
                          label=rf"$\phi$ = {azimuthal_angle[i]:.2f} $\pi$")
        # plt.legend()
        ax_plain.set_xlabel(trs.choose("Poynting Vector "+r"$P_x$ [au]",
                                       "Vector de Poynting "+r"$P_x$ [ua]"))
        ax_plain.set_ylabel(trs.choose("Poynting Vector "+r"$P_z$ [au]",
                                       "Vector de Poynting "+r"$P_z$ [ua]"))
        
        for ax in fig.axes:
            box = ax.get_position()
            box.x1 = box.x1 - .15 * (box.x1 - box.x0)
            ax.set_position(box)
        plt.legend(bbox_to_anchor=(1.32, 0.5), loc="center right", frameon=False)
        
        fig.set_size_inches([7.13, 4.8 ])
        plt.savefig(sa.file("AngularAzimuthal.png"))
        
    if near2far and pm.assign(0) and separate_simulations_needed:
        
        freq_index = np.argmin(np.abs(wlens - np.mean(wlen_range) * 1e3 * from_um_factor))
        index = [list(azimuthal_angle).index(alpha) for alpha in [0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2]]
        colors = plt.get_cmap("spring")( np.linspace(0,1,len(azimuthal_angle)) )
        
        fig = plt.figure()
        plt.suptitle(trs.choose('Angular Pattern', 'Patrón angular') + plot_title_ending)
        
        ax_plain = plt.axes()
        for i in index:
            ax_plain.plot(np.sqrt(np.square(poynting_x[i,:,freq_index]) + np.square(poynting_y[i,:,freq_index])), 
                                  poynting_z[i,:,freq_index], ".-", color=colors[i], alpha=0.7,
                                  label=rf"$\phi$ = {azimuthal_angle[i]:.2f} $\pi$")
        plt.legend()
        ax_plain.set_xlabel(trs.choose("Poynting Vector "+r"$P_\rho$ [au]",
                                       "Vector de Poynting "+r"$P_\rho$ [ua]"))
        ax_plain.set_ylabel(trs.choose("Poynting Vector "+r"$P_z$ [au]",
                                       "Vector de Poynting "+r"$P_z$ [ua]"))
        
        for ax in fig.axes:
            box = ax.get_position()
            box.x1 = box.x1 - .15 * (box.x1 - box.x0)
            ax.set_position(box)
        plt.legend(bbox_to_anchor=(1.32, 0.5), loc="center right", frameon=False)
        
        fig.set_size_inches([7.13, 4.8 ])
        
        plt.savefig(sa.file("AngularAzimuthalAbs.png"))