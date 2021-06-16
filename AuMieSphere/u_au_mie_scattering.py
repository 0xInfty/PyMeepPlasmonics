#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Scattering efficiency in visible spectrum of 120nm-diameter Au sphere.

Adapted from Meep Tutorial: Mie Scattering of a Lossless Dielectric Sphere
"""

script = "au_mie_scattering"

from socket import gethostname
if "Nano" in gethostname():
    syshome = "/home/nanofisica/Documents/Vale/ThesisPython"
    sysname = "SC"
elif "vall" in gethostname():
    syshome = "/home/vall/Documents/Thesis/ThesisPython"
    sysname = "MC"
else:
    raise ValueError("Your PC must be registered at the top of this code")

import sys
sys.path.append(syshome)

import click as cli
# import h5py as h5
import meep as mp
# try: 
#     from mpi4py import MPI
# except:
#     print("No mpi4py module found!")
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time#, sleep
import PyMieScatt as ps
import v_materials as vmt
import v_meep as vm
import v_save as vs
import v_utilities as vu

#%% COMMAND LINE FORMATTER

@cli.command()
@cli.option("--from-um-factor", "-fum", "from_um_factor", 
            default=10e-3, type=float,
            help="Conversion of 1 μm to my length unit (i.e. 10e-3=10nm/1μm)")
@cli.option("--resolution", "-res", required=True, type=int,
            help="Spatial resolution. Number of divisions of each Meep unit")
# >=8 pixels per smallest wavelength, i.e. np.floor(8/wvl_min)
@cli.option("--courant", "-c", "courant", 
            type=float, default=0.5,
            help="Courant factor: time discretization from space discretization")
@cli.option("--radius", "-r", "r", default=60, type=float,
            help="Radius of sphere expressed in nm")
@cli.option("--paper", "-pp", "paper", type=str, default="R",
            help="Source of inner material experimental data. Options: 'JC'/'R'/'P'")
@cli.option("--reference", "-ref", "reference", type=str, default="Meep",
            help="Reference from which the data was extracted. Options: 'Meep'/'RIinfo'")
@cli.option("--index", "-i", "submerged_index", 
            type=float, default=1,
            help="Reflective index of sourrounding medium")
@cli.option("--displacement", "-dis", "displacement", default=0, type=float,
            help="Overlap of sphere and surface in nm")
@cli.option("--surface-index", "-si", "surface_index", 
            type=float, default=1,
            help="Reflective index of surface medium")
@cli.option("--wlen-range", "-wr", "wlen_range", 
            type=vu.NUMPY_ARRAY, default="np.array([500,650])",
            help="Wavelength range expressed in nm")
@cli.option("--stime", "-st", "second_time_factor", 
            type=float, default=10,
            help="Second simulation total time expressed as multiples of 1st")
@cli.option("--series", "-s", type=str, default=None,
            help="Series name used to create a folder and save files")
@cli.option("--folder", "-f", type=str, default=None,
            help="Series folder used to save files")
@cli.option("--parallel", "-par", type=bool, default=False,
            help="Whether the program is being run in parallel or in serial")
@cli.option("--n-processes", "-np", "n_processes", type=int, default=1,
            help="Number of nuclei used to run the program in parallel")
def main(from_um_factor, resolution, courant, 
         r, paper, reference, submerged_index, 
         displacement, surface_index,
         wlen_range, second_time_factor,
         series, folder, parallel, n_processes):

    #%% CLASSIC INPUT PARAMETERS    
    """
    # Simulation size
    from_um_factor = 10e-3 # Conversion of 1 μm to my length unit (=10nm/1μm)
    resolution = 2 # >=8 pixels per smallest wavelength, i.e. np.floor(8/wvl_min)
    courant = 0.5
    
    # Nanoparticle specifications: Sphere in Vacuum :)
    r = 60  # Radius of sphere in nm
    paper = "R"
    reference = "Meep"
    displacement = 0 # Displacement of the surface from the bottom of the sphere in nm
    submerged_index = 1.33 # 1.33 for water
    surface_index = 1.54 # 1.54 for glass
    
    # Frequency and wavelength
    wlen_range = np.array([500,650]) # Wavelength range in nm
    
    # Simulation time
    second_time_factor = 10
    
    # Saving directories
    series = None
    folder = None
    
    # Configuration
    parallel = False
    n_processes = 1
    """

    #%% MORE INPUT PARAMETERS

    # Nanoparticle specifications: Sphere in Vacuum :)
    material = "Au" # Gold: "Au"    
    
    # Frequency and wavelength
    nfreq = 100 # Number of frequencies to discretize range
    cutoff = 3.2 # Gaussian planewave source's parameter of shape
    
    # Simulation time
    time_factor_cell = 1.2 # Defined in multiples of time required to go through the cell
    
    ### TREATED INPUT PARAMETERS
    
    # Nanoparticle specifications: Sphere in Vacuum :)
    r = r / ( from_um_factor * 1e3 ) # Now in Meep units
    if reference=="Meep": 
        medium = vmt.import_medium(material, from_um_factor, paper=paper)
        # Importing material constants dependant on frequency from Meep Library
    elif reference=="RIinfo":
        medium = vmt.MediumFromFile(material, paper=paper, reference=reference, from_um_factor=from_um_factor)
        # Importing material constants dependant on frequency from external file
    else:
        raise ValueError("Reference for medium not recognized. Sorry :/")
    displacement = displacement / ( from_um_factor * 1e3 ) # Now in Meep units
    
    # Frequency and wavelength
    wlen_range = wlen_range / ( from_um_factor * 1e3 ) # Now in Meep units
    freq_range = 1/wlen_range # Hz range in Meep units from highest to lowest
    freq_center = np.mean(freq_range)
    freq_width = max(freq_range) - min(freq_range)
    
    # Space configuration
    pml_width = 0.38 * max(wlen_range)
    air_width = r/2 # 0.5 * max(wlen_range)
    
    # Computation
    enlapsed = []
    if parallel:
        np_process = mp.count_processors()
    else:
        np_process = 1
    
    # Saving directories
    if series is None:
        series = "Test"
    if folder is None:
        folder = "Test"
    params_list = ["from_um_factor", "resolution", "courant",
                   "r", "paper", "reference", "submerged_index",
                   "wlen_range", "nfreq", "cutoff",
                   "cell_width", "pml_width", "air_width", "source_center",
                   "until_after_sources", "time_factor_cell", "second_time_factor",
                   "enlapsed", "parallel", "n_processes", "script", "sysname", "path"]
    
    #%% GENERAL GEOMETRY SETUP
    
    air_width = air_width - air_width%(1/resolution)
    
    pml_width = pml_width - pml_width%(1/resolution)
    pml_layers = [mp.PML(thickness=pml_width)]
    
    # symmetries = [mp.Mirror(mp.Y), 
    #               mp.Mirror(mp.Z, phase=-1)]
    # Two mirror planes reduce cell size to 1/4
    # Issue related that lead me to comment this lines:
    # https://github.com/NanoComp/meep/issues/1484

    cell_width = 2 * (pml_width + air_width + r)
    cell_width = cell_width - cell_width%(1/resolution)
    cell_size = mp.Vector3(cell_width, cell_width, cell_width)
    
    # surface_center = r/4 - displacement/2 + cell_width/4
    # surface_center = surface_center - surface_center%(1/resolution)
    # displacement = r/2 + cell_width/2 - 2*surface_center
    
    displacement = displacement - displacement%(1/resolution)

    source_center = -0.5*cell_width + pml_width
    sources = [mp.Source(mp.GaussianSource(freq_center,
                                            fwidth=freq_width,
                                            is_integrated=True,
                                            cutoff=cutoff),
                          center=mp.Vector3(source_center),
                          size=mp.Vector3(0, cell_width, cell_width),
                          component=mp.Ez)]
    # Ez-polarized planewave pulse 
    # (its size parameter fills the entire cell in 2d)
    # >> The planewave source extends into the PML 
    # ==> is_integrated=True must be specified
    
    until_after_sources = time_factor_cell * cell_width * submerged_index
    # Enough time for the pulse to pass through all the cell
    # Originally: Aprox 3 periods of lowest frequency, using T=λ/c=λ in Meep units 
    # Now: Aprox 3 periods of highest frequency, using T=λ/c=λ in Meep units 
    
    geometry = [mp.Sphere(material=medium,
                          center=mp.Vector3(),
                          radius=r)]
    # Au sphere with frequency-dependant characteristics imported from Meep.
    
    if surface_index != 1:
        geometry = [mp.Block(material=mp.Medium(index=surface_index),
                             center=mp.Vector3(
                                 r/2 - displacement/2 + cell_width/4,
                                 0, 0),
                             size=mp.Vector3(
                                 cell_width/2 - r + displacement,
                                 cell_width, cell_width)),
                    *geometry]
    # A certain material surface underneath it
    
    home = vs.get_home()
    sysname = vs.get_sys_name()
    path = os.path.join(home, folder, series)
    if not os.path.isdir(path) and vm.parallel_assign(0, n_processes, parallel):
        os.makedirs(path)
    file = lambda f : os.path.join(path, f)
    
    #%% FIRST RUN
    
    params = {}
    params["surface_index"] = 1
    params["displacement"] = 0
    for p in params_list: params[p] = eval(p)
    
    try:
        flux_path = vm.check_midflux(params)[0]
    
    except:
        #% FIRST RUN: SET UP
        
        sim = mp.Simulation(resolution=resolution,
                            cell_size=cell_size,
                            boundary_layers=pml_layers,
                            sources=sources,
                            k_point=mp.Vector3(),
                            Courant=courant,
                            default_material=mp.Medium(index=submerged_index),
                            output_single_precision=True)#,
                            # symmetries=symmetries)
        # >> k_point zero specifies boundary conditions needed
        # for the source to be infinitely extended
        
        # Scattered power --> Computed by surrounding it with closed DFT flux box 
        # (its size and orientation are irrelevant because of Poynting's theorem) 
        box_x1 = sim.add_flux(freq_center, freq_width, nfreq, 
                              mp.FluxRegion(center=mp.Vector3(x=-r),
                                            size=mp.Vector3(0,2*r,2*r)))
        box_x2 = sim.add_flux(freq_center, freq_width, nfreq, 
                              mp.FluxRegion(center=mp.Vector3(x=+r),
                                            size=mp.Vector3(0,2*r,2*r)))
        box_y1 = sim.add_flux(freq_center, freq_width, nfreq,
                              mp.FluxRegion(center=mp.Vector3(y=-r),
                                            size=mp.Vector3(2*r,0,2*r)))
        box_y2 = sim.add_flux(freq_center, freq_width, nfreq,
                              mp.FluxRegion(center=mp.Vector3(y=+r),
                                            size=mp.Vector3(2*r,0,2*r)))
        box_z1 = sim.add_flux(freq_center, freq_width, nfreq,
                              mp.FluxRegion(center=mp.Vector3(z=-r),
                                            size=mp.Vector3(2*r,2*r,0)))
        box_z2 = sim.add_flux(freq_center, freq_width, nfreq,
                              mp.FluxRegion(center=mp.Vector3(z=+r),
                                            size=mp.Vector3(2*r,2*r,0)))
        # Funny you can encase the sphere (r radius) so closely (2r-sided box)
        
        #% FIRST RUN: INITIALIZE
        
        temp = time()
        sim.init_sim()
        enlapsed.append( time() - temp )
        
        #% FIRST RUN: SIMULATION NEEDED TO NORMALIZE
        
        temp = time()
        sim.run(until_after_sources=until_after_sources)
            #     mp.stop_when_fields_decayed(
            # np.mean(wlen_range), # dT = mean period of source
            # mp.Ez, # Component of field to check
            # mp.Vector3(0.5*cell_width - pml_width, 0, 0), # Where to check
            # 1e-3)) # Factor to decay
        enlapsed.append( time() - temp )
        
        #% SAVE MID DATA
        
        for p in params_list: params[p] = eval(p)
        
        field = sim.get_array(center=mp.Vector3(), 
                              size=(cell_width, cell_width, cell_width), 
                              component=mp.Ez)
        
        # if parallel:
        #     f = h5.File(file("MidField.h5"), "w", driver='mpio', comm=MPI.COMM_WORLD)
        # else:
        #     f = h5.File(file("MidField.h5"), "w")
        # f.create_dataset("Ez", data=field)
        # for a in params: f["Ez"].attrs[a] = params[a]
        # f.close()
        # del f

        flux_path =  vm.save_midflux(sim, box_x1, box_x2, box_y1, 
                                     box_y2, box_z1, box_z2, params, path)
        
        freqs = np.asarray(mp.get_flux_freqs(box_x1))
        box_x1_flux0 = np.asarray(mp.get_fluxes(box_x1))
        box_x2_flux0 = np.asarray(mp.get_fluxes(box_x2))
        box_y1_flux0 = np.asarray(mp.get_fluxes(box_y1))
        box_y2_flux0 = np.asarray(mp.get_fluxes(box_y2))
        box_z1_flux0 = np.asarray(mp.get_fluxes(box_z1))
        box_z2_flux0 = np.asarray(mp.get_fluxes(box_z2))
        
        data_mid = np.array([1e3*from_um_factor/freqs, box_x1_flux0, box_x2_flux0, 
                             box_y1_flux0, box_y2_flux0, box_z1_flux0, box_z2_flux0]).T
        
        header_mid = ["Longitud de onda [nm]", 
                      "Flujo X10 [u.a.]",
                      "Flujo X20 [u.a]",
                      "Flujo Y10 [u.a]",
                      "Flujo Y20 [u.a]",
                      "Flujo Z10 [u.a]",
                      "Flujo Z20 [u.a]"]
        
        if vm.parallel_assign(0, n_processes, parallel):
            vs.savetxt(file("MidFlux.txt"), data_mid, 
                       header=header_mid, footer=params)

        #% PLOT FLUX FOURIER MID DATA
        
        if vm.parallel_assign(1, np_process, parallel):
            ylims = (np.min(data_mid[:,1:]), np.max(data_mid[:,1:]))
            ylims = (ylims[0]-.1*(ylims[1]-ylims[0]),
                      ylims[1]+.1*(ylims[1]-ylims[0]))
            
            fig, ax = plt.subplots(3, 2, sharex=True)
            fig.subplots_adjust(hspace=0, wspace=.05)
            for a in ax[:,1]:
                a.yaxis.tick_right()
                a.yaxis.set_label_position("right")
            for a, h in zip(np.reshape(ax, 6), header_mid[1:]):
                a.set_ylabel(h)
            
            for d, a in zip(data_mid[:,1:].T, np.reshape(ax, 6)):
                a.plot(1e3*from_um_factor/freqs, d)
                a.set_ylim(*ylims)
            ax[-1,0].set_xlabel("Wavelength [nm]")
            ax[-1,1].set_xlabel("Wavelength [nm]")
            
            plt.savefig(file("MidFlux.png"))
        
        #% PLOT FLUX WALLS FIELD
        
        if vm.parallel_assign(0, np_process, parallel):
            # index_to_space = lambda i : i/resolution - cell_width/2
            space_to_index = lambda x : round(resolution * (x + cell_width/2))
            
            field_walls = [field[space_to_index(-r),:,:],
                            field[space_to_index(r),:,:],
                            field[:,space_to_index(-r),:],
                            field[:,space_to_index(r),:],
                            field[:,:,space_to_index(-r)],
                            field[:,:,space_to_index(r)]]
            
            zlims = (np.min([np.min(f) for f in field_walls]), 
                      np.max([np.max(f) for f in field_walls]))
            
            fig, ax = plt.subplots(3, 2)
            fig.subplots_adjust(hspace=0.25)
            for a, h in zip(np.reshape(ax, 6), header_mid[1:]):
                a.set_title(h.split(" ")[1].split("0")[0])
            
            for f, a in zip(field_walls, np.reshape(ax, 6)):
                a.imshow(f.T, interpolation='spline36', cmap='RdBu', 
                          vmin=zlims[0], vmax=zlims[1])
                a.axis("off")
            
            plt.savefig(file("MidField.png"))
    
    #%% SECOND RUN: SETUP
    
    sim = mp.Simulation(resolution=resolution,
                        cell_size=cell_size,
                        boundary_layers=pml_layers,
                        sources=sources,
                        k_point=mp.Vector3(),
                        Courant=courant,
                        default_material=mp.Medium(index=submerged_index),
                        output_single_precision=True,
                        # symmetries=symmetries,
                        geometry=geometry,)
    
    
    box_x1 = sim.add_flux(freq_center, freq_width, nfreq, 
                          mp.FluxRegion(center=mp.Vector3(x=-r),
                                        size=mp.Vector3(0,2*r,2*r)))
    box_x2 = sim.add_flux(freq_center, freq_width, nfreq, 
                          mp.FluxRegion(center=mp.Vector3(x=+r),
                                        size=mp.Vector3(0,2*r,2*r)))
    box_y1 = sim.add_flux(freq_center, freq_width, nfreq, 
                          mp.FluxRegion(center=mp.Vector3(y=-r),
                                        size=mp.Vector3(2*r,0,2*r)))
    box_y2 = sim.add_flux(freq_center, freq_width, nfreq, 
                          mp.FluxRegion(center=mp.Vector3(y=+r),
                                        size=mp.Vector3(2*r,0,2*r)))
    box_z1 = sim.add_flux(freq_center, freq_width, nfreq, 
                          mp.FluxRegion(center=mp.Vector3(z=-r),
                                        size=mp.Vector3(2*r,2*r,0)))
    box_z2 = sim.add_flux(freq_center, freq_width, nfreq, 
                          mp.FluxRegion(center=mp.Vector3(z=+r),
                                        size=mp.Vector3(2*r,2*r,0)))
    
    #%% SECOND RUN: INITIALIZE
    
    temp = time()
    sim.init_sim()
    enlapsed.append( time() - temp )
    
    #%% LOAD FLUX FROM FILE
    
    vm.load_midflux(sim, box_x1, box_x2, box_y1, box_y2, box_z1, box_z2, flux_path)

    freqs = np.asarray(mp.get_flux_freqs(box_x1))
    box_x1_flux0 = np.asarray(mp.get_fluxes(box_x1))
    box_x1_data = sim.get_flux_data(box_x1)
    box_x2_data = sim.get_flux_data(box_x2)
    box_y1_data = sim.get_flux_data(box_y1)
    box_y2_data = sim.get_flux_data(box_y2)
    box_z1_data = sim.get_flux_data(box_z1)
    box_z2_data = sim.get_flux_data(box_z2)
    
    temp = time()
    sim.load_minus_flux_data(box_x1, box_x1_data)
    sim.load_minus_flux_data(box_x2, box_x2_data)
    sim.load_minus_flux_data(box_y1, box_y1_data)
    sim.load_minus_flux_data(box_y2, box_y2_data)
    sim.load_minus_flux_data(box_z1, box_z1_data)
    sim.load_minus_flux_data(box_z2, box_z2_data)
    enlapsed.append( time() - temp )
    del box_x1_data, box_x2_data, box_y1_data, box_y2_data
    del box_z1_data, box_z2_data

    #%% SECOND RUN: SIMULATION :D
    
    temp = time()
    sim.run(until_after_sources=second_time_factor*until_after_sources) 
        #     mp.stop_when_fields_decayed(
        # np.mean(wlen_range), # dT = mean period of source
        # mp.Ez, # Component of field to check
        # mp.Vector3(0.5*cell_width - pml_width, 0, 0), # Where to check
        # 1e-3)) # Factor to decay
    enlapsed.append( time() - temp )
    del temp
    # Aprox 30 periods of lowest frequency, using T=λ/c=λ in Meep units 
    
    box_x1_flux = np.asarray(mp.get_fluxes(box_x1))
    box_x2_flux = np.asarray(mp.get_fluxes(box_x2))
    box_y1_flux = np.asarray(mp.get_fluxes(box_y1))
    box_y2_flux = np.asarray(mp.get_fluxes(box_y2))
    box_z1_flux = np.asarray(mp.get_fluxes(box_z1))
    box_z2_flux = np.asarray(mp.get_fluxes(box_z2))
    
    #%% ANALYSIS
    
    scatt_flux = box_x1_flux - box_x2_flux
    scatt_flux = scatt_flux + box_y1_flux - box_y2_flux
    scatt_flux = scatt_flux + box_z1_flux - box_z2_flux
    
    intensity = box_x1_flux0/(2*r)**2
    # Flux of one of the six monitor planes / Área
    # (the closest one, facing the planewave source) 
    # This is why the six sides of the flux box are separated
    # (Otherwise, the box could've been one flux object with weights ±1 per side)
    
    scatt_cross_section = np.divide(scatt_flux, intensity)
    # Scattering cross section σ = 
    # = scattered power in all directions / incident intensity.
    
    scatt_eff_meep = -1 * scatt_cross_section / (np.pi*r**2)
    # Scattering efficiency =
    # = scattering cross section / cross sectional area of the sphere
    
    freqs = np.array(freqs)
    scatt_eff_theory = [ps.MieQ(np.sqrt(medium.epsilon(f)[0,0]*medium.mu(f)[0,0]), 
                                1e3*from_um_factor/f,
                                2*r*1e3*from_um_factor,
                                nMedium=submerged_index,
                                asDict=True)['Qsca'] 
                        for f in freqs]
    # The simulation results are validated by comparing with 
    # analytic theory of PyMieScatt module
    
    #%% SAVE FINAL DATA
    
    for p in params_list: params[p] = eval(p)
    
    data = np.array([1e3*from_um_factor/freqs, scatt_eff_meep, scatt_eff_theory]).T
    
    header = ["Longitud de onda [nm]", 
              "Sección eficaz efectiva (Meep) [u.a.]", 
              "Sección eficaz efectiva (Theory) [u.a.]"]
    
    data_base = np.array([1e3*from_um_factor/freqs, box_x1_flux0, box_x1_flux,
                          box_x2_flux, box_y1_flux, box_y2_flux, 
                          box_z1_flux, box_z2_flux, 
                          intensity, scatt_flux, scatt_cross_section]).T
    
    header_base = ["Longitud de onda [nm]", 
                    "Flujo X10 [u.a.]",
                    "Flujo X1 [u.a]",
                    "Flujo X2 [u.a]",
                    "Flujo Y1 [u.a]",
                    "Flujo Y2 [u.a]",
                    "Flujo Z1 [u.a]",
                    "Flujo Z2 [u.a]",
                    "Intensidad incidente [u.a.]", 
                    "Flujo scattereado [u.a.]",
                    "Sección eficaz de scattering [u.a.]"]
    
    if vm.parallel_assign(0, np_process, parallel):
        vs.savetxt(file("Results.txt"), data, 
                   header=header, footer=params)
        vs.savetxt(file("BaseResults.txt"), data_base, 
                   header=header_base, footer=params)
    
    #%% PLOT ALL TOGETHER
    
    if vm.parallel_assign(1, np_process, parallel):
        plt.figure()
        plt.plot(1e3*from_um_factor/freqs, scatt_eff_meep,'bo-',label='Meep')
        plt.plot(1e3*from_um_factor/freqs, scatt_eff_theory,'ro-',label='Theory')
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Scattering efficiency [σ/πr$^{2}$]')
        plt.legend()
        plt.title('Mie Scattering of Au Sphere With {} nm Radius'.format(r*10))
        plt.tight_layout()
        plt.savefig(file("Comparison.png"))
    
    #%% PLOT SEPARATE
    
    if vm.parallel_assign(0, np_process, parallel):
        plt.figure()
        plt.plot(1e3*from_um_factor/freqs, scatt_eff_meep,'bo-',label='Meep')
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Scattering efficiency [σ/πr$^{2}$]')
        plt.legend()
        plt.title('Mie Scattering of Au Sphere With {} nm Radius'.format(r*10))
        plt.tight_layout()
        plt.savefig(file("Meep.png"))
        
        plt.figure()
        plt.plot(1e3*from_um_factor/freqs, scatt_eff_theory,'bo-',label='Theory')
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Scattering efficiency [σ/πr$^{2}$]')
        plt.legend()
        plt.title('Mie Scattering of Au Sphere With {} nm Radius'.format(r*10))
        plt.tight_layout()
        plt.savefig(file("Theory.png"))
    
    #%% PLOT ONE ABOVE THE OTHER
    
    if vm.parallel_assign(1, np_process, parallel):
        fig, axes = plt.subplots(nrows=2, sharex=True)
        fig.subplots_adjust(hspace=0)
        
        axes[0].plot(1e3*from_um_factor/freqs, scatt_eff_meep,'bo-',label='Meep')
        axes[0].yaxis.tick_right()
        axes[0].set_ylabel('Scattering efficiency [σ/πr$^{2}$]')
        axes[0].legend()
        
        axes[1].plot(1e3*from_um_factor/freqs, scatt_eff_theory,'bo-',label='Theory')
        axes[1].set_xlabel('Wavelength [nm]')
        axes[1].set_ylabel('Scattering efficiency [σ/πr$^{2}$]')
        axes[1].legend()
        
        plt.savefig(file("SeparatedComparison.png"))
        
    #%% PLOT FLUX FOURIER FINAL DATA
    
    if vm.parallel_assign(0, np_process, parallel):
        
        ylims = (np.min(data_base[:,2:8]), np.max(data_base[:,2:8]))
        ylims = (ylims[0]-.1*(ylims[1]-ylims[0]),
                  ylims[1]+.1*(ylims[1]-ylims[0]))
        
        fig, ax = plt.subplots(3, 2, sharex=True)
        fig.subplots_adjust(hspace=0, wspace=.05)
        for a in ax[:,1]:
            a.yaxis.tick_right()
            a.yaxis.set_label_position("right")
        for a, h in zip(np.reshape(ax, 6), header_base[1:7]):
            a.set_ylabel(h)
        
        for d, a in zip(data_base[:,3:9].T, np.reshape(ax, 6)):
            a.plot(1e3*from_um_factor/freqs, d)
            a.set_ylim(*ylims)
        ax[-1,0].set_xlabel("Wavelength [nm]")
        ax[-1,1].set_xlabel("Wavelength [nm]")
        
        plt.savefig(file("FinalFlux.png"))
    
#%%

if __name__ == '__main__':
    main()
