#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis of single run of Au sphere field for a visible monochromatic incident wave.

See also
--------
Routines/np_monoch_field
"""

import imageio as mim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import use as use_backend
import numpy as np
import os
import vmp_utilities as vmu
import vmp_analysis as vma
import v_plot as vp
import v_save as vs
import v_utilities as vu

vp.set_style()

#%% PARAMETERS

"""
series = "AllVacRes14"
folder = "Scattering/AuSphere/AlgorithmTest/RAM/TestRAMRes/AllVac"

with_line = False
with_plane = False
with_flux_box = True
with_nanoparticle = True

english = False
"""

#%%

def plot_np_planewave_cell(params, series, folder, 
                           with_line=False, with_plane=False,
                           with_flux_box=False, with_flux_walls=False,
                           with_nanoparticle=False, 
                           english=False):
    
    #%% SETUP
    
    # Computation
    pm = vmu.ParallelManager()
    n_processes, n_cores, n_nodes = pm.specs
    parallel = pm.parallel
    
    # Saving directories
    sa = vmu.SavingAssistant(series, folder)
    home = sa.home
    sysname = sa.sysname
    path = sa.path
    
    trs = vu.BilingualManager(english=english)
    
    plot_for_display = True
    
    """
    params = vs.retrieve_footer(sa.file("Results.txt"))
    """
    
    #%% DATA EXTRACTION
    
    from_um_factor = params["from_um_factor"]
    try:
        units = params["units"]
    except:
        units = True
    
    cell_width = params["cell_width"]
    pml_width = params["pml_width"]
    
    try:
        r = params["r"]
        material = params["material"]
    except:
        r = 0
        material = "none"
        with_nanoparticle = False
    
    try:
        wlen = params["wlen"]
    except:
        wlen_range = params["wlen_range"]
        wlen_center = np.mean(wlen_range)
        wlen_width = float(np.diff(wlen_range))
    source_center = params["source_center"]
    
    submerged_index = params["submerged_index"]
    surface_index = params["surface_index"]
    try:
        overlap = params["overlap"]
    except:
        overlap = 0

    try:        
        flux_box_size = params["flux_box_size"]
    except:
        flux_box_size = 0
        with_flux_box = False
        
    try:
        flux_wall_positions = params["flux_wall_positions"]
    except:
        flux_wall_positions = []
        with_flux_walls = False
    
    #%% PLOT

    if plot_for_display: use_backend("Agg")

    fig, ax = plt.subplots()
    ax.grid(False)
    if plot_for_display: fig.dpi = 200
    
    # PML borders
    pml_out_square = plt.Rectangle((-cell_width/2, -cell_width/2), 
                                   cell_width, cell_width,
                                   fill=False, edgecolor="m", linestyle="dashed",
                                   hatch='/', 
                                   zorder=-20,
                                   label=trs.choose("PML borders", "Bordes PML"))
    pml_inn_square = plt.Rectangle((-cell_width/2+pml_width,
                                    -cell_width/2+pml_width), 
                                   cell_width - 2*pml_width, cell_width - 2*pml_width,
                                   facecolor="white", edgecolor="m", 
                                   linestyle="dashed", linewidth=1, zorder=-10)
   
    # Surrounding medium
    if submerged_index != 1:
        surrounding_square = plt.Rectangle((-cell_width/2, -cell_width/2),
                                           cell_width, cell_width,
                                           color="blue", alpha=.1, zorder=-6,
                                           label=trs.choose(fr"Medium $n$={submerged_index}",
                                                            fr"Medio $n$={submerged_index}"))
        
    # Surface medium
    if surface_index != submerged_index:
        surface_square = plt.Rectangle((r - overlap, -cell_width/2),
                                       cell_width/2 - r + overlap, 
                                       cell_width,
                                       edgecolor="navy", hatch=r"\\", 
                                       fill=False, zorder=-3,
                                       label=trs.choose(fr"Surface $n$={surface_index}",
                                                        fr"Superficie $n$={surface_index}"))
    # Nanoparticle
    if with_nanoparticle:
        if material=="Au":
            circle_color = "gold"
        elif material=="Ag":
            circle_color="darkgrey"
        else:
            circle_color="peru"
        circle = plt.Circle((0,0), r, color=circle_color, linewidth=1, alpha=.4, 
                            zorder=0, label=trs.choose(f"{material} Nanoparticle",
                                                          f"Nanopartícula de {material}"))
    
    # Source
    ax.vlines(source_center, -cell_width/2, cell_width/2,
              color="r", linestyle="dashed", zorder=5, 
              label=trs.choose("Planewave Source", "Fuente de ondas plana"))
    
    # Sampling line
    if with_line:
        ax.hlines(0, -cell_width/2, cell_width/2,
                  color="blue", linestyle=":", zorder=7, # limegreen
                  label=trs.choose("Sampling Line", "Línea de muestreo"))
    
    # Sampling plane
    if with_plane:
        ax.vlines(0, -cell_width/2, cell_width/2,
                  color="blue", linestyle="dashed", zorder=7, 
                  label=trs.choose("Sampling Plane", "Plano de muestreo"))
        
    # Flux box
    if with_flux_box:
        flux_square = plt.Rectangle((-flux_box_size/2, -flux_box_size/2), 
                                    flux_box_size, flux_box_size,
                                    linewidth=1, edgecolor="limegreen", linestyle="dashed",
                                    fill=False, zorder=10, 
                                    label=trs.choose("Flux box", "Caja de flujo"))

    # Flux wall
    if with_flux_walls:
        for flux_x in flux_wall_positions:
            l = ax.vlines(flux_x, -cell_width/2+pml_width, cell_width/2-pml_width,
                          color="limegreen", linestyle="dashed", zorder=10)
        l.set_label(trs.choose("Flux walls", "Paredes de flujo"))
    
    if with_nanoparticle: ax.add_patch(circle)
    if submerged_index!=1: ax.add_patch(surrounding_square)
    if surface_index!=submerged_index and surface_index!=1: ax.add_patch(surface_square)
    if with_flux_box: ax.add_patch(flux_square)
    ax.add_patch(pml_out_square)
    ax.add_patch(pml_inn_square)
    
    # General configuration
    box = ax.get_position()
    width = box.x1 - box.x0
    height = box.y1 - box.y0
    box.x0 = box.x0 - .2 * width
    box.x1 = box.x0 + width
    box.y0 = box.y0 + .12 * height
    box.y1 = box.y0 + height
    ax.set_position(box)           
    plt.legend(bbox_to_anchor=trs.choose( (1.8, 0.5), (1.87, 0.5) ), 
               loc="center right", frameon=False)
    
    fig.set_size_inches(7.5, 4.8)
    ax.set_aspect("equal")
    plt.xlim(-cell_width/2, cell_width/2)
    plt.ylim(-cell_width/2, cell_width/2)
    plt.xlabel(trs.choose(r"Position $X$ [MPu]", r"Posición $X$ [uMP]"))
    plt.ylabel(trs.choose(r"Position $Z$ [MPu]", r"Posición $Z$ [uMP]"))
    
    if units:
        plt.annotate(trs.choose(f"1 MPu = {from_um_factor * 1e3:.0f} nm",
                                f"1 uMP = {from_um_factor * 1e3:.0f} nm"),
                     (5, 5), xycoords='figure points')
        try:
            plt.annotate(fr"$\lambda$ = {wlen * from_um_factor * 1e3:.0f} nm",
                         (350, 5), xycoords='figure points', color="r")
        except:
            plt.annotate(fr"$\lambda_0$ = {wlen_center * from_um_factor * 1e3:.0f} nm" + 
                         ", " +
                         fr"$\Delta\lambda$ = {wlen_width * from_um_factor * 1e3:.0f} nm",
                         (345, 5), xycoords='figure points', color="r")
    else:
        plt.annotate(trs.choose(r"1 MPu = $\lambda$",
                                r"1 uMP = $\lambda$"),
                     (5, 5), xycoords='figure points')
        try:
            plt.annotate(fr"$\lambda$ = {wlen * from_um_factor * 1e3:.2f} " + 
                         trs.choose("MPu", "uMP"),
                         (350, 5), xycoords='figure points', color="r")
        except:
            plt.annotate(fr"$\lambda_0$ = {wlen_center * from_um_factor * 1e3:.2f} " + 
                         trs.choose("MPu", "uMP") + ", " +
                         fr"$\Delta\lambda$ = {wlen_width * from_um_factor * 1e3:.2f} " + 
                         trs.choose("MPu", "uMP"),
                         (345, 5), xycoords='figure points', color="r")
        
    fig.set_size_inches([6.68, 3.68])
    
    #%%
        
    if with_nanoparticle:
        plt.savefig(sa.file("MiniSimBox.png"))
    else:
        plt.savefig(sa.file("MiniSimBoxNorm.png"))
        
    if plot_for_display: use_backend("Qt5Agg")
            
#%% 

def mid_plots_np_scattering(series, folder, english=False):
    
    #%% SETUP
    
    # Computation
    pm = vmu.ParallelManager()
    
    # Saving directories
    sa = vmu.SavingAssistant(series, folder)
    
    trs = vu.BilingualManager(english=english)
    
    plot_for_display = False
    
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
    
    #%% PLOT FLUX FOURIER FINAL DATA
    
    header = trs.choose([r"Wavelength $\lambda$ [nm]", 
                         "Flux\n"+r"$X_{10}$ [ua]", "Flux\n"+r"$X_{20}$ [ua]",
                         "Flux\n"+r"$Y_{10}$ [ua]", "Flux\n"+r"$Y_{20}$ [ua]",
                         "Flux\n"+r"$Z_{10}$ [ua]", "Flux\n"+r"$Z_{20}$ [ua]"],
                        [r"Longitud de onda $\lambda$ [nm]", 
                         "Flujo\n"+r"$X_{10}$ [ua]", "Flujo\n"+r"$X_{20}$ [ua]",
                         "Flujo\n"+r"$Y_{10}$ [ua]", "Flujo\n"+r"$Y_{20}$ [ua]",
                         "Flujo\n"+r"$Z_{10}$ [ua]", "Flujo\n"+r"$Z_{20}$ [ua]"])
    
    if pm.assign(1):
        ylims = (np.min(flux_data[:,1:]), np.max(flux_data[:,1:]))
        ylims = (ylims[0]-.1*(ylims[1]-ylims[0]),
                  ylims[1]+.1*(ylims[1]-ylims[0]))
        
        if plot_for_display: use_backend("Agg")
        
        fig, ax = plt.subplots(3, 2, sharex=True, gridspec_kw={"hspace":0, "wspace":0})
        fig.subplots_adjust(hspace=0, wspace=.05)
        if plot_for_display: fig.dpi = 200
        
        plt.suptitle(trs.choose("Incident Flux", "Flujo incidente") + plot_title_ending)
        for a in ax[:,1]:
            a.yaxis.tick_right()
            a.yaxis.set_label_position("right")
        for a, h in zip(np.reshape(ax, 6), header[1:]):
            a.set_ylabel(h)
        
        for d, a in zip(flux_data[:,1:].T, np.reshape(ax, 6)):
            a.plot(flux_data[:,0], d)
            a.set_ylim(*ylims)
        ax[-1,0].set_xlabel(trs.choose("Wavelength\n"+r"$\lambda$ [nm]",
                                       "Longitud de onda\n"+r"$\lambda$ [nm]"))
        ax[-1,1].set_xlabel(trs.choose("Wavelength\n"+r"$\lambda$ [nm]",
                                       "Longitud de onda\n"+r"$\lambda$ [nm]"))
    
        fig.set_size_inches([6.09, 3.95])
        plt.tight_layout()
        plt.savefig(sa.file("MiniMidFlux.png"))
        
        if plot_for_display: use_backend("Qt5Agg")

#%%

def plots_np_scattering(series, folder, near2far=False, 
                        separate_simulations_needed=False, english=False):
    
    #%% SETUP
    
    # Computation
    pm = vmu.ParallelManager()
    
    # Saving directories
    sa = vmu.SavingAssistant(series, folder)
    
    trs = vu.BilingualManager(english=english)
    
    plot_for_display = True
    
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
                         "Flux\n"+r"$X_{1}$ [ua]", "Flux\n"+r"$X_{2}$ [ua]",
                         "Flux\n"+r"$Y_{1}$ [ua]", "Flux\n"+r"$Y_{2}$ [ua]",
                         "Flux\n"+r"$Z_{1}$ [ua]", "Flux\n"+r"$Z_{2}$ [ua]"],
                        [r"Longitud de onda $\lambda$ [nm]", 
                         "Flujo\n"+r"$X_{1}$ [ua]", "Flujo\n"+r"$X_{2}$ [ua]",
                         "Flujo\n"+r"$Y_{1}$ [ua]", "Flujo\n"+r"$Y_{2}$ [ua]",
                         "Flujo\n"+r"$Z_{1}$ [ua]", "Flujo\n"+r"$Z_{2}$ [ua]"])
    
    if pm.assign(1):
        if plot_for_display: use_backend("Agg")
        
        ylims = (np.min(flux_data[:,1:]), np.max(flux_data[:,1:]))
        ylims = (ylims[0]-.1*(ylims[1]-ylims[0]),
                  ylims[1]+.1*(ylims[1]-ylims[0]))
        
        fig, ax = plt.subplots(3, 2, sharex=True, gridspec_kw={"hspace":0, "wspace":0})
        fig.subplots_adjust(hspace=0, wspace=.05)
        if plot_for_display: fig.dpi = 200
        
        plt.suptitle(trs.choose("Scattered Flux", "Flujo dispersado") + plot_title_ending)
        for a in ax[:,1]:
            a.yaxis.tick_right()
            a.yaxis.set_label_position("right")
        for a, h in zip(np.reshape(ax, 6), header[1:]):
            a.set_ylabel(h)
        
        for d, a in zip(flux_data[:,1:].T, np.reshape(ax, 6)):
            a.plot(flux_data[:,0], d)
            a.set_ylim(*ylims)
        ax[-1,0].set_xlabel(trs.choose("Wavelength\n"+r"$\lambda$ [nm]",
                                       "Longitud de onda\n"+r"$\lambda$ [nm]"))
        ax[-1,1].set_xlabel(trs.choose("Wavelength\n"+r"$\lambda$ [nm]",
                                       "Longitud de onda\n"+r"$\lambda$ [nm]"))
        
        fig.set_size_inches([6.09, 3.95])
        plt.tight_layout()
        plt.savefig(sa.file("MiniFinalFlux.png"))
        
        if plot_for_display: use_backend("Qt5Agg")
            
    #%% PLOT SCATTERING: SEPARATE
    
    if pm.assign(1):
        if plot_for_display: use_backend("Agg")
        fig = plt.figure()
        if plot_for_display: fig.dpi = 200
        plt.title(trs.choose('Scattering', 'Dispersión') + plot_title_ending)
        plt.plot(data[:,0], data[:,1],'b-',label='Meep',alpha=.6)
        plt.xlabel(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
        plt.ylabel(trs.choose("Scattering\nefficiency "+r"$C_{scat}$", 
                              "Eficiencia de\ndispersión "+r"$C_{disp}$"))
    
        fig.set_size_inches([4.2 , 2.47])
        plt.tight_layout()
        plt.savefig(sa.file("MiniMeep.png"))
        if plot_for_display: use_backend("Qt5Agg")
        