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
elif "vall" in gethostname():
    syshome = "/home/vall/Documents/Thesis/ThesisPython"
else:
    syshome = "/nfs/home/vpais/ThesisPython"
    # raise ValueError("Your PC must be registered at the top of this code")

import sys
sys.path.append(syshome)

import click as cli
import h5py as h5
import meep as mp
try: 
    from mpi4py import MPI
except:
    print("No mpi4py module found!")
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time#, sleep
import PyMieScatt as ps
import v_materials as vmt
import v_meep as vm
import v_save as vs

used_ram, swapped_ram, measure_ram = vm.ram_manager()
measure_ram()

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
@cli.option("--radius", "-r", "r", default=51.5, type=float,
            help="Radius of sphere expressed in nm")
@cli.option("--material", "-mat", default="Au", type=str,
            help="Material of spherical nanoparticle")
@cli.option("--paper", "-pp", "paper", type=str, default="R",
            help="Source of inner material experimental data. Options: 'JC'/'R'/'P'")
@cli.option("--reference", "-ref", "reference", type=str, default="Meep",
            help="Reference from which the data was extracted. Options: 'Meep'/'RIinfo'")
@cli.option("--index", "-i", "submerged_index", 
            type=float, default=1,
            help="Reflective index of sourrounding medium")
@cli.option("--overlap", "-over", "overlap", default=0, type=float,
            help="Overlap of sphere and surface in nm")
@cli.option("--surface-index", "-si", "surface_index", 
            type=float, default=None,
            help="Reflective index of surface medium")
@cli.option("--wlen-range", "-wr", "wlen_range", 
            type=cli.Tuple([float, float]), default=(450,600),
            help="Wavelength range expressed in nm")
@cli.option("--nfreq", "-nfreq", "nfreq", 
            type=int, default=100,
            help="Quantity of frequencies to sample in wavelength range")
@cli.option("--air-r-factor", "-air", "air_r_factor", 
            type=float, default=0.5,
            help="Empty layer width expressed in multiples of radius")
@cli.option("--pml-wlen-factor", "-pml", "pml_wlen_factor", 
            type=float, default=0.38,
            help="PML layer width expressed in multiples of maximum wavelength")
@cli.option("--flux-r-factor", "-flux", "flux_r_factor", 
            type=float, default=0,
            help="Flux box padding expressed in multiples of radius")
@cli.option("--time-factor-cell", "-tfc", "time_factor_cell", 
            type=float, default=1.2,
            help="First simulation total time expressed as multiples of time \
                required to go through the cell")
@cli.option("--second-time-factor", "-stf", "second_time_factor", 
            type=float, default=10,
            help="Second simulation total time expressed as multiples of 1st")
@cli.option("--series", "-s", type=str, default=None,
            help="Series name used to create a folder and save files")
@cli.option("--folder", "-f", type=str, default=None,
            help="Series folder used to save files")
@cli.option("--parallel", "-par", type=bool, default=True,
            help="Whether the program is being run in parallel or in serial")
@cli.option("--split-chunks-evenly", "-chev", "split_chunks_evenly", 
            type=bool, default=True,
            help="Whether to split chunks evenly or not during parallel run")
@cli.option("--n-processes", "-np", "n_processes", type=int, default=0,
            help="Number of cores used to run the program in parallel")
@cli.option("--n-cores", "-nc", "n_cores", type=int, default=0,
            help="Number of cores used to run the program in parallel")
@cli.option("--n-nodes", "-nn", "n_nodes", type=int, default=0,
            help="Number of nodes used to run the program in parallel")
@cli.option("--load-flux", "-loadf", "load_flux", 
            type=bool, default=True,
            help="Whether to search the midflux database and load, if possible, or not.")
@cli.option("--load-chunks", "-loadc", "load_chunks", 
            type=bool, default=True,
            help="Whether to search the chunks layout database and load, if possible, or not.")
@cli.option("--near2far", "-n2f", "near2far", 
            type=bool, default=False,
            help="Whether to calculate angular pattern or not.")
@cli.option("--make-plots", "-plt", "make_plots", 
            type=bool, default=True,
            help="Whether to make plots while running or not.")
def main(from_um_factor, resolution, courant, 
         r, material, paper, reference, submerged_index, 
         overlap, surface_index, wlen_range, nfreq,
         air_r_factor, pml_wlen_factor, flux_r_factor,
         time_factor_cell, second_time_factor,
         series, folder, 
         parallel, n_processes, n_cores, n_nodes, split_chunks_evenly,
         load_flux, load_chunks, near2far, make_plots):

    #%% CLASSIC INPUT PARAMETERS    
    """
    # Simulation size
    from_um_factor = 10e-3 # Conversion of 1 μm to my length unit (=10nm/1μm)
    resolution = 2 # >=8 pixels per smallest wavelength, i.e. np.floor(8/wvl_min)
    courant = 0.5
    
    # Nanoparticle specifications: Sphere in Vacuum :)
    material = "Au"
    r = 51.5  # Radius of sphere in nm
    paper = "R"
    reference = "Meep"
    overlap = 0 # Displacement of the surface from the bottom of the sphere in nm
    submerged_index = 1 # 1.33 for water
    surface_index = None # 1.54 for glass
    
    # Frequency and wavelength
    wlen_range = np.array([450,600]) # Wavelength range in nm
    nfreq = 100
    
    # Box dimensions
    pml_wlen_factor = 0.38
    air_r_factor = 0.5
    flux_r_factor = 0 #0.1
    
    # Simulation time
    time_factor_cell = 1.2
    second_time_factor = 10
    
    # Saving directories
    series = None
    folder = None
    
    # Configuration
    parallel = False
    n_processes = 1
    n_cores = 1
    n_nodes = 1
    split_chunks_evenly = True
    load_flux = True
    load_chunks = True    
    near2far = False
    make_plots = True
    """

    #%% MORE INPUT PARAMETERS
    
    # Frequency and wavelength
    cutoff = 3.2 # Gaussian planewave source's parameter of shape
    nazimuthal = 16
    npolar = 20
    
    ### TREATED INPUT PARAMETERS
    
    # Nanoparticle specifications: Sphere in Vacuum :)
    r = r / ( from_um_factor * 1e3 ) # Now in Meep units
    if reference=="Meep": 
        medium = vmt.import_medium(material, from_um_factor=from_um_factor, paper=paper)
        # Importing material constants dependant on frequency from Meep Library
    elif reference=="RIinfo":
        medium = vmt.MediumFromFile(material, paper=paper, reference=reference, from_um_factor=from_um_factor)
        # Importing material constants dependant on frequency from external file
    else:
        raise ValueError("Reference for medium not recognized. Sorry :/")
    overlap = overlap / ( from_um_factor * 1e3 ) # Now in Meep units
    if surface_index is None:
        surface_index = submerged_index
    
    # Frequency and wavelength
    wlen_range = np.array(wlen_range)
    wlen_range = wlen_range / ( from_um_factor * 1e3 ) # Now in Meep units
    freq_range = 1/wlen_range # Hz range in Meep units from highest to lowest
    freq_center = np.mean(freq_range)
    freq_width = max(freq_range) - min(freq_range)
    
    # Space configuration
    pml_width = pml_wlen_factor * max(wlen_range) # 0.5 * max(wlen_range)
    air_width = air_r_factor * r # 0.5 * max(wlen_range)
    flux_box_size = 2 * ( 1 + flux_r_factor ) * r
       
    # Saving directories
    if series is None:
        series = "Test"
    if folder is None:
        folder = "Test"
    params_list = ["from_um_factor", "resolution", "courant",
                   "material", "r", "paper", "reference", "submerged_index",
                   "overlap", "surface_index",
                   "wlen_range", "nfreq", "nazimuthal", "npolar", "cutoff", "flux_box_size",
                   "cell_width", "pml_width", "air_width", "source_center",
                   "until_after_sources", "time_factor_cell", "second_time_factor",
                   "enlapsed", "parallel", "n_processes", "n_cores", "n_nodes",
                   "split_chunks_evenly", "near2far",
                   "script", "sysname", "path"]
    
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
    
    # surface_center = r/4 - overlap/2 + cell_width/4
    # surface_center = surface_center - surface_center%(1/resolution)
    # overlap = r/2 + cell_width/2 - 2*surface_center
    
    overlap = overlap - overlap%(1/resolution)
    
    flux_box_size = flux_box_size - flux_box_size%(1/resolution)

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
    
    until_after_sources = time_factor_cell * cell_width * max( submerged_index, surface_index )
    # Enough time for the pulse to pass through all the cell
    # Originally: Aprox 3 periods of lowest frequency, using T=λ/c=λ in Meep units 
    # Now: Aprox 3 periods of highest frequency, using T=λ/c=λ in Meep units 
    
    nanoparticle = mp.Sphere(material=medium,
                             center=mp.Vector3(),
                             radius=r)
    # Au sphere with frequency-dependant characteristics imported from Meep.
    
    if surface_index != submerged_index:
        initial_geometry = [mp.Block(material=mp.Medium(index=surface_index),
                                     center=mp.Vector3(
                                         r/2 - overlap/2 + cell_width/4,
                                         0, 0),
                                     size=mp.Vector3(
                                         cell_width/2 - r + overlap,
                                         cell_width, cell_width))]
    else:
        initial_geometry = []
    # If required, a certain material surface underneath it
    
    final_geometry = [*initial_geometry, nanoparticle]
          
    enlapsed = []
    
    parallel_specs = np.array([n_processes, n_cores, n_nodes], dtype=int)
    max_index = np.argmax(parallel_specs)
    for index, item in enumerate(parallel_specs): 
        if item == 0: parallel_specs[index] = 1
    parallel_specs[0:max_index] = np.full(parallel_specs[0:max_index].shape, 
                                          max(parallel_specs))
    n_processes, n_cores, n_nodes = parallel_specs
    parallel = max(parallel_specs) > 1
    del parallel_specs, max_index, index, item
            
    if parallel:
        np_process = mp.count_processors()
    else:
        np_process = 1
        
    home = vs.get_home()
    sysname = vs.get_sys_name()
    path = os.path.join(home, folder, series)
    if not os.path.isdir(path) and vm.parallel_assign(0, n_processes, parallel):
        os.makedirs(path)
    file = lambda f : os.path.join(path, f)
    
    #%% PLOT CELL

    if make_plots:
        fig, ax = plt.subplots()
        
        # PML borders
        pml_out_square = plt.Rectangle((-cell_width/2, -cell_width/2), 
                                       cell_width, cell_width,
                                       fill=False, edgecolor="m", linestyle="dashed",
                                       hatch='/', 
                                       zorder=-20,
                                       label="PML borders")
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
                                               label=fr"Medium $n$={submerged_index}") 
    
        # Surface medium
        if surface_index != submerged_index:
            surface_square = plt.Rectangle((r - overlap, -cell_width/2),
                                           cell_width/2 - r + overlap, 
                                           cell_width,
                                           edgecolor="navy", hatch=r"\\", 
                                           fill=False, zorder=-3,
                                           label=fr"Surface $n$={surface_index}") 
    
        # Nanoparticle
        if material=="Au":
            circle_color = "gold"
        elif material=="Ag":
            circle_color="silver"
        else:
            circle_color="peru"
        circle = plt.Circle((0,0), r, color=circle_color, linewidth=1, alpha=.4, 
                            zorder=0, label=f"{material} Nanoparticle")
        
        # Source
        ax.vlines(source_center, -cell_width/2, cell_width/2,
                  color="r", linestyle="dashed", zorder=5, label="Planewave Source")
        
        # Flux box
        flux_square = plt.Rectangle((-flux_box_size/2,-flux_box_size/2), 
                                    flux_box_size, flux_box_size,
                                    linewidth=1, edgecolor="limegreen", linestyle="dashed",
                                    fill=False, zorder=10, label="Flux box")
        
        ax.add_patch(circle)
        if submerged_index!=1: ax.add_patch(surrounding_square)
        if surface_index!=submerged_index: ax.add_patch(surface_square)
        ax.add_patch(flux_square)
        ax.add_patch(pml_out_square)
        ax.add_patch(pml_inn_square)
        
        # General configuration
        
        box = ax.get_position()
        box.x0 = box.x0 - .15 * (box.x1 - box.x0)
        # box.x1 = box.x1 - .05 * (box.x1 - box.x0)
        box.y1 = box.y1 + .10 * (box.y1 - box.y0)
        ax.set_position(box)
        plt.legend(bbox_to_anchor=(1.5, 0.5), loc="center right", frameon=False)
        
        fig.set_size_inches(7.5, 4.8)
        ax.set_aspect("equal")
        plt.xlim(-cell_width/2, cell_width/2)
        plt.ylim(-cell_width/2, cell_width/2)
        plt.xlabel("Position X [Meep Units]")
        plt.ylabel("Position Y [Meep Units]")
        
        plt.annotate(f"1 Meep Unit = {from_um_factor * 1e3:.0f} nm",
                (5, 5),
                xycoords='figure points')
        plt.show()
        
        plt.savefig(file("SimBox.png"))
        
    #%% FIRST RUN
    
    measure_ram()
    
    params = {}
    for p in params_list: params[p] = eval(p)
    
    stable, max_courant = vm.check_stability(params)
    if stable:
        print("As a whole, the simulation should be stable")
    else:
        print("As a whole, the simulation could not be stable")
        print(f"Recommended maximum courant factor is {max_courant}")
    del stable, max_courant
        
    if load_flux:
        try:
            flux_path = vm.check_midflux(params)[0]
            flux_needed = False
        except:
            flux_needed = True
    else:
        flux_needed = True
        
    if load_chunks and not split_chunks_evenly:
        try:
            chunks_path = vm.check_chunks(params)[0]
            chunk_layout = os.path.join(chunks_path, "Layout.h5")
            chunks_needed = False
        except:
            chunks_needed = True
            flux_needed = True
    else:
        if not split_chunks_evenly:
            chunks_needed = True
            flux_needed = True
        else:
            chunk_layout = None
            chunks_needed = False
    
    if chunks_needed:

        sim = mp.Simulation(resolution=resolution,
                            cell_size=cell_size,
                            boundary_layers=pml_layers,
                            sources=sources,
                            k_point=mp.Vector3(),
                            Courant=courant,
                            default_material=mp.Medium(index=submerged_index),
                            output_single_precision=True,
                            split_chunks_evenly=split_chunks_evenly,
                            # symmetries=symmetries,
                            geometry=final_geometry)
    
        sim.init_sim()
        
        chunks_path = vm.save_chunks(sim, params, path)         
        chunk_layout = os.path.join(chunks_path, "Layout.h5")
        
        del sim
    
    if flux_needed:
        
        #% FIRST RUN: SET UP
        
        measure_ram()
        
        sim = mp.Simulation(resolution=resolution,
                            cell_size=cell_size,
                            boundary_layers=pml_layers,
                            sources=sources,
                            k_point=mp.Vector3(),
                            Courant=courant,
                            default_material=mp.Medium(index=submerged_index),
                            split_chunks_evenly=split_chunks_evenly,
                            chunk_layout=chunk_layout,
                            output_single_precision=True,
                            geometry=initial_geometry)#,
                            # symmetries=symmetries)
        # >> k_point zero specifies boundary conditions needed
        # for the source to be infinitely extended
        
        measure_ram()

        # Scattered power --> Computed by surrounding it with closed DFT flux box 
        # (its size and orientation are irrelevant because of Poynting's theorem) 
        box_x1 = sim.add_flux(freq_center, freq_width, nfreq, 
                              mp.FluxRegion(center=mp.Vector3(x=-flux_box_size/2),
                                            size=mp.Vector3(0,flux_box_size,flux_box_size)))
        box_x2 = sim.add_flux(freq_center, freq_width, nfreq, 
                              mp.FluxRegion(center=mp.Vector3(x=+flux_box_size/2),
                                            size=mp.Vector3(0,flux_box_size,flux_box_size)))
        box_y1 = sim.add_flux(freq_center, freq_width, nfreq,
                              mp.FluxRegion(center=mp.Vector3(y=-flux_box_size/2),
                                            size=mp.Vector3(flux_box_size,0,flux_box_size)))
        box_y2 = sim.add_flux(freq_center, freq_width, nfreq,
                              mp.FluxRegion(center=mp.Vector3(y=+flux_box_size/2),
                                            size=mp.Vector3(flux_box_size,0,flux_box_size)))
        box_z1 = sim.add_flux(freq_center, freq_width, nfreq,
                              mp.FluxRegion(center=mp.Vector3(z=-flux_box_size/2),
                                            size=mp.Vector3(flux_box_size,flux_box_size,0)))
        box_z2 = sim.add_flux(freq_center, freq_width, nfreq,
                              mp.FluxRegion(center=mp.Vector3(z=+flux_box_size/2),
                                            size=mp.Vector3(flux_box_size,flux_box_size,0)))
        # Funny you can encase the sphere (r radius) so closely (2r-sided box)
        
        measure_ram()
        
        if near2far:
            near2far_box = sim.add_near2far(freq_center, freq_width, nfreq, 
                mp.Near2FarRegion(center=mp.Vector3(x=-flux_box_size/2),
                                  size=mp.Vector3(0,flux_box_size,flux_box_size),
                                   weight=-1),
                mp.Near2FarRegion(center=mp.Vector3(x=+flux_box_size/2),
                                  size=mp.Vector3(0,flux_box_size,flux_box_size),
                                   weight=+1),
                mp.Near2FarRegion(center=mp.Vector3(y=-flux_box_size/2),
                                  size=mp.Vector3(flux_box_size,0,flux_box_size),
                                   weight=-1),
                mp.Near2FarRegion(center=mp.Vector3(y=+flux_box_size/2),
                                  size=mp.Vector3(flux_box_size,0,flux_box_size),
                                   weight=+1),
                mp.Near2FarRegion(center=mp.Vector3(z=-flux_box_size/2),
                                  size=mp.Vector3(flux_box_size,flux_box_size,0),
                                   weight=-1),
                mp.Near2FarRegion(center=mp.Vector3(z=+flux_box_size/2),
                                  size=mp.Vector3(flux_box_size,flux_box_size,0),
                                   weight=+1))
            measure_ram()
        else:
            near2far_box = None
            # used_ram.append(used_ram[-1])
        
        #% FIRST RUN: INITIALIZE
        
        temp = time()
        sim.init_sim()
        enlapsed.append( time() - temp )
        measure_ram()
        
        step_ram_function = lambda sim : measure_ram()
        
        #% FIRST RUN: SIMULATION NEEDED TO NORMALIZE
        
        temp = time()
        sim.run(mp.at_beginning(step_ram_function), 
                mp.at_time(int(until_after_sources / 2), 
                           step_ram_function),
                mp.at_end(step_ram_function),
                until_after_sources=until_after_sources )
            #     mp.stop_when_fields_decayed(
            # np.mean(wlen_range), # dT = mean period of source
            # mp.Ez, # Component of field to check
            # mp.Vector3(0.5*cell_width - pml_width, 0, 0), # Where to check
            # 1e-3)) # Factor to decay
        enlapsed.append( time() - temp )
        
        #% SAVE MID DATA

        for p in params_list: params[p] = eval(p)

        flux_path =  vm.save_midflux(sim, box_x1, box_x2, box_y1, 
                                     box_y2, box_z1, box_z2, 
                                     near2far_box, params, path)
                
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

        if not split_chunks_evenly:
            vm.save_chunks(sim, params, path)
            
        if parallel:
            f = h5.File(file("MidRAM.h5"), "w", driver='mpio', comm=MPI.COMM_WORLD)
            current_process = mp.my_rank()
            f.create_dataset("RAM", (len(used_ram), np_process), dtype="float")
            f["RAM"][:, current_process] = used_ram
            for a in params: f["RAM"].attrs[a] = params[a]
            f.create_dataset("SWAP", (len(used_ram), np_process), dtype="int")
            f["SWAP"][:, current_process] = swapped_ram
            for a in params: f["SWAP"].attrs[a] = params[a]
        else:
            f = h5.File(file("MidRAM.h5"), "w")
            f.create_dataset("RAM", data=used_ram)
            for a in params: f["RAM"].attrs[a] = params[a]
            f.create_dataset("SWAP", data=swapped_ram)
            for a in params: f["SWAP"].attrs[a] = params[a]
        f.close()
        del f

        #% PLOT FLUX FOURIER MID DATA
        
        if vm.parallel_assign(1, np_process, parallel) and make_plots:
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
            del fig, ax, ylims, a, h
            
        sim.reset_meep()
        del data_mid, box_x2_flux0, box_y1_flux0, box_y2_flux0
        del box_z1_flux0, box_z2_flux0, header_mid
    
    #%% SECOND RUN: SETUP
    
    measure_ram()
    
    sim = mp.Simulation(resolution=resolution,
                        cell_size=cell_size,
                        boundary_layers=pml_layers,
                        sources=sources,
                        k_point=mp.Vector3(),
                        Courant=courant,
                        default_material=mp.Medium(index=submerged_index),
                        output_single_precision=True,
                        split_chunks_evenly=split_chunks_evenly,
                        chunk_layout=chunk_layout,
                        # symmetries=symmetries,
                        geometry=final_geometry)
    
    measure_ram()
    
    box_x1 = sim.add_flux(freq_center, freq_width, nfreq, 
                          mp.FluxRegion(center=mp.Vector3(x=-flux_box_size/2),
                                        size=mp.Vector3(0,flux_box_size,flux_box_size)))
    box_x2 = sim.add_flux(freq_center, freq_width, nfreq, 
                          mp.FluxRegion(center=mp.Vector3(x=+flux_box_size/2),
                                        size=mp.Vector3(0,flux_box_size,flux_box_size)))
    box_y1 = sim.add_flux(freq_center, freq_width, nfreq, 
                          mp.FluxRegion(center=mp.Vector3(y=-flux_box_size/2),
                                        size=mp.Vector3(flux_box_size,0,flux_box_size)))
    box_y2 = sim.add_flux(freq_center, freq_width, nfreq, 
                          mp.FluxRegion(center=mp.Vector3(y=+flux_box_size/2),
                                        size=mp.Vector3(flux_box_size,0,flux_box_size)))
    box_z1 = sim.add_flux(freq_center, freq_width, nfreq, 
                          mp.FluxRegion(center=mp.Vector3(z=-flux_box_size/2),
                                        size=mp.Vector3(flux_box_size,flux_box_size,0)))
    box_z2 = sim.add_flux(freq_center, freq_width, nfreq, 
                          mp.FluxRegion(center=mp.Vector3(z=+flux_box_size/2),
                                        size=mp.Vector3(flux_box_size,flux_box_size,0)))
    
    measure_ram()

    if near2far and surface_index != submerged_index:
        near2far_box = sim.add_near2far(freq_center, freq_width, nfreq, 
            mp.Near2FarRegion(center=mp.Vector3(x=-flux_box_size/2),
                              size=mp.Vector3(0,flux_box_size,flux_box_size),
                               weight=-1),
            mp.Near2FarRegion(center=mp.Vector3(x=+flux_box_size/2),
                              size=mp.Vector3(0,flux_box_size,flux_box_size),
                               weight=+1),
            mp.Near2FarRegion(center=mp.Vector3(y=-flux_box_size/2),
                              size=mp.Vector3(flux_box_size,0,flux_box_size),
                               weight=-1),
            mp.Near2FarRegion(center=mp.Vector3(y=+flux_box_size/2),
                              size=mp.Vector3(flux_box_size,0,flux_box_size),
                               weight=+1),
            mp.Near2FarRegion(center=mp.Vector3(z=-flux_box_size/2),
                              size=mp.Vector3(flux_box_size,flux_box_size,0),
                               weight=-1),
            mp.Near2FarRegion(center=mp.Vector3(z=+flux_box_size/2),
                              size=mp.Vector3(flux_box_size,flux_box_size,0),
                               weight=+1))
        separate_simulations_needed = False
        measure_ram()
    else:
        near2far_box = None
        separate_simulations_needed = True
        
        # used_ram.append(used_ram[-1])

    #%% SECOND RUN: INITIALIZE
    
    temp = time()
    sim.init_sim()
    enlapsed.append( time() - temp )
    measure_ram()
    
    #%% LOAD FLUX FROM FILE
    
    vm.load_midflux(sim, box_x1, box_x2, box_y1, box_y2, box_z1, box_z2, 
                    near2far_box, flux_path)
    
    measure_ram()

    freqs = np.asarray(mp.get_flux_freqs(box_x1))
    box_x1_flux0 = np.asarray(mp.get_fluxes(box_x1))
    box_x1_data = sim.get_flux_data(box_x1)
    box_x2_data = sim.get_flux_data(box_x2)
    box_y1_data = sim.get_flux_data(box_y1)
    box_y2_data = sim.get_flux_data(box_y2)
    box_z1_data = sim.get_flux_data(box_z1)
    box_z2_data = sim.get_flux_data(box_z2)
    if near2far and not separate_simulations_needed: 
        near2far_data = sim.get_near2far_data(near2far_box)
     
    temp = time()
    sim.load_minus_flux_data(box_x1, box_x1_data)
    sim.load_minus_flux_data(box_x2, box_x2_data)
    sim.load_minus_flux_data(box_y1, box_y1_data)
    sim.load_minus_flux_data(box_y2, box_y2_data)
    sim.load_minus_flux_data(box_z1, box_z1_data)
    sim.load_minus_flux_data(box_z2, box_z2_data)
    if near2far and not separate_simulations_needed: 
        sim.load_minus_near2far_data(near2far_box, near2far_data)
    enlapsed.append( time() - temp )
    del box_x1_data, box_x2_data, box_y1_data, box_y2_data
    del box_z1_data, box_z2_data
    if near2far and not separate_simulations_needed:  
        del near2far_data
    
    measure_ram()

    #%% SECOND RUN: SIMULATION :D
    
    step_ram_function = lambda sim : measure_ram()
    
    temp = time()
    sim.run(mp.at_beginning(step_ram_function), 
            mp.at_time(int(second_time_factor * until_after_sources / 2), 
                       step_ram_function),
            mp.at_end(step_ram_function),
            until_after_sources=second_time_factor * until_after_sources )
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
    
    #%% SCATTERING ANALYSIS
    
    scatt_flux = box_x1_flux - box_x2_flux
    scatt_flux = scatt_flux + box_y1_flux - box_y2_flux
    scatt_flux = scatt_flux + box_z1_flux - box_z2_flux
    
    intensity = box_x1_flux0/(flux_box_size)**2
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
    
    #%% ANGULAR PATTERN ANALYSIS
    
    if near2far and not separate_simulations_needed: 
        
        fraunhofer_distance = 8 * (r**2) / min(wlen_range)        
        radial_distance = max( 10 * fraunhofer_distance , 1.5 * cell_width/2 ) 
        # radius of far-field circle must be at least Fraunhofer distance
        
        azimuthal_angle = np.arange(0, 2 + 2/nazimuthal, 2/nazimuthal) # in multiples of pi
        polar_angle = np.arange(0, 1 + 1/npolar, 1/npolar)
        
        poynting_x = []
        poynting_y = []
        poynting_z = []
        poynting_r = []
        
        for phi in azimuthal_angle:
            
            poynting_x.append([])
            poynting_y.append([])
            poynting_z.append([])
            poynting_r.append([])
            
            for theta in polar_angle:
                
                farfield_dict = sim.get_farfields(near2far_box, 1,
                                                  where=mp.Volume(
                                                      center=mp.Vector3(radial_distance * np.cos(np.pi * phi) * np.sin(np.pi * theta),
                                                                        radial_distance * np.sin(np.pi * phi) * np.sin(np.pi * theta),
                                                                        radial_distance * np.cos(np.pi * theta))))
                
                Px = farfield_dict["Ey"] * np.conjugate(farfield_dict["Hz"])
                Px -= farfield_dict["Ez"] * np.conjugate(farfield_dict["Hy"])
                Py = farfield_dict["Ez"] * np.conjugate(farfield_dict["Hx"])
                Py -= farfield_dict["Ex"] * np.conjugate(farfield_dict["Hz"])
                Pz = farfield_dict["Ex"] * np.conjugate(farfield_dict["Hy"])
                Pz -= farfield_dict["Ey"] * np.conjugate(farfield_dict["Hx"])
                
                Px = np.real(Px)
                Py = np.real(Py)
                Pz = np.real(Pz)
                
                poynting_x[-1].append( Px )
                poynting_y[-1].append( Py )
                poynting_z[-1].append( Pz )
                poynting_r[-1].append( np.sqrt( np.square(Px) + np.square(Py) + np.square(Pz) ) )
        
        del Px, Py, Pz, farfield_dict
        
        poynting_x = np.array(poynting_x)
        poynting_y = np.array(poynting_y)
        poynting_z = np.array(poynting_z)
        poynting_r = np.array(poynting_r)
    
    elif separate_simulations_needed: 
        print("Beware! Near2far is not supported yet for nanoparticle placed on a surface.")

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
    del data
        
    if near2far and not separate_simulations_needed: 
        
        header_near2far = ["Poynting medio Px [u.a.]",
                           "Poynting medio Py [u.a.]",
                           "Poynting medio Pz [u.a.]",
                           "Poynting medio Pr [u.a.]"]
        
        data_near2far = [poynting_x.reshape(poynting_x.size),
                         poynting_y.reshape(poynting_y.size),
                         poynting_z.reshape(poynting_z.size),
                         poynting_r.reshape(poynting_r.size)]
        
        if vm.parallel_assign(1, np_process, parallel):
            vs.savetxt(file("Near2FarResults.txt"), data_near2far, 
                       header=header_near2far, footer=params)
        del header_near2far, data_near2far
        
    if not split_chunks_evenly:
        vm.save_chunks(sim, params, path)        
    
    if parallel:
        f = h5.File(file("RAM.h5"), "w", driver='mpio', comm=MPI.COMM_WORLD)
        current_process = mp.my_rank()
        f.create_dataset("RAM", (len(used_ram), np_process), dtype="float")
        f["RAM"][:, current_process] = used_ram
        for a in params: f["RAM"].attrs[a] = params[a]
        f.create_dataset("SWAP", (len(used_ram), np_process), dtype="int")
        f["SWAP"][:, current_process] = swapped_ram
        for a in params: f["SWAP"].attrs[a] = params[a]
    else:
        f = h5.File(file("RAM.h5"), "w")
        f.create_dataset("RAM", data=used_ram)
        for a in params: f["RAM"].attrs[a] = params[a]
        f.create_dataset("SWAP", data=swapped_ram)
        for a in params: f["SWAP"].attrs[a] = params[a]
    f.close()
    del f
    
    if flux_needed and vm.parallel_assign(0, np_process, parallel):
        os.remove(file("MidRAM.h5"))
    
    #%% PLOT ALL TOGETHER
    
    if surface_index==submerged_index and make_plots:
        if vm.parallel_assign(0, np_process, parallel):
            plt.figure()
            plt.plot(1e3*from_um_factor/freqs, scatt_eff_meep,'bo-',label='Meep')
            plt.plot(1e3*from_um_factor/freqs, scatt_eff_theory,'ro-',label='Theory')
            plt.xlabel('Wavelength [nm]')
            plt.ylabel('Scattering efficiency [σ/πr$^{2}$]')
            plt.legend()
            plt.title('Scattering of Au Sphere With {:.1f} nm Radius'.format(r*from_um_factor*1e3))
            plt.tight_layout()
            plt.savefig(file("Comparison.png"))
    
    #%% PLOT SEPARATE
    
    if vm.parallel_assign(1, np_process, parallel) and make_plots:
        plt.figure()
        plt.plot(1e3*from_um_factor/freqs, scatt_eff_meep,'bo-',label='Meep')
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Scattering efficiency [σ/πr$^{2}$]')
        plt.legend()
        plt.title('Scattering of Au Sphere With {:.1f} nm Radius'.format(r*from_um_factor*1e3))
        plt.tight_layout()
        plt.savefig(file("Meep.png"))
        
    if surface_index==submerged_index and make_plots:
        if vm.parallel_assign(0, np_process, parallel):
            plt.figure()
            plt.plot(1e3*from_um_factor/freqs, scatt_eff_theory,'ro-',label='Theory')
            plt.xlabel('Wavelength [nm]')
            plt.ylabel('Scattering efficiency [σ/πr$^{2}$]')
            plt.legend()
            plt.title('Scattering of Au Sphere With {:.1f} nm Radius'.format(r*10))
            plt.tight_layout()
            plt.savefig(file("Theory.png"))
    
    #%% PLOT ONE ABOVE THE OTHER
    
    if surface_index==submerged_index and make_plots:
        if vm.parallel_assign(1, np_process, parallel):
            fig, axes = plt.subplots(nrows=2, sharex=True)
            fig.subplots_adjust(hspace=0)
            plt.suptitle('Scattering of Au Sphere With {:.1f} nm Radius'.format(r*from_um_factor*1e3))
            
            axes[0].plot(1e3*from_um_factor/freqs, scatt_eff_meep,'bo-',label='Meep')
            axes[0].yaxis.tick_right()
            axes[0].set_ylabel('Scattering efficiency [σ/πr$^{2}$]')
            axes[0].legend()
            
            axes[1].plot(1e3*from_um_factor/freqs, scatt_eff_theory,'ro-',label='Theory')
            axes[1].set_xlabel('Wavelength [nm]')
            axes[1].set_ylabel('Scattering efficiency [σ/πr$^{2}$]')
            axes[1].legend()
            
            plt.savefig(file("SeparatedComparison.png"))
            
    #%% PLOT FLUX FOURIER FINAL DATA
    
    if vm.parallel_assign(0, np_process, parallel) and make_plots:
        
        ylims = (np.min(data_base[:,2:8]), np.max(data_base[:,2:8]))
        ylims = (ylims[0]-.1*(ylims[1]-ylims[0]),
                  ylims[1]+.1*(ylims[1]-ylims[0]))
        
        fig, ax = plt.subplots(3, 2, sharex=True)
        fig.subplots_adjust(hspace=0, wspace=.05)
        plt.suptitle('Final flux of Au Sphere With {:.1f} nm Radius'.format(r*from_um_factor*1e3))
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
    
    #%% PLOT ANGULAR PATTERN IN 3D
    
    if near2far and vm.parallel_assign(1, np_process, parallel) and separate_simulations_needed and make_plots:
    
        freq_index = np.argmin(np.abs(freqs - freq_center))
    
        fig = plt.figure()
        plt.suptitle('Angular Pattern of Au Sphere With {:.1f} nm Radius at {:.1f} nm'.format(r*from_um_factor*1e3,
                                                                                              from_um_factor*1e3/freqs[freq_index]))
        ax = fig.add_subplot(1,1,1, projection='3d')
        ax.plot_surface(
            poynting_x[:,:,freq_index], 
            poynting_y[:,:,freq_index], 
            poynting_z[:,:,freq_index], cmap=plt.get_cmap('jet'), 
            linewidth=1, antialiased=False, alpha=0.5)
        ax.set_xlabel(r"$P_x$")
        ax.set_ylabel(r"$P_y$")
        ax.set_zlabel(r"$P_z$")
        
        plt.savefig(file("AngularPattern.png"))
        
    #%% PLOT ANGULAR PATTERN PROFILE FOR DIFFERENT POLAR ANGLES
        
    if near2far and vm.parallel_assign(0, np_process, parallel) and separate_simulations_needed and make_plots:
        
        freq_index = np.argmin(np.abs(freqs - freq_center))
        index = [list(polar_angle).index(alpha) for alpha in [0, .25, .5, .75, 1]]
        
        plt.figure()
        plt.suptitle('Angular Pattern of Au Sphere With {:.1f} nm Radius at {:.1f} nm'.format(r*from_um_factor*1e3,
                                                                                              from_um_factor*1e3/freqs[freq_index]))
        ax_plain = plt.axes()
        for i in index:
            ax_plain.plot(poynting_x[:,i,freq_index], 
                          poynting_y[:,i,freq_index], 
                          ".-", label=rf"$\theta$ = {polar_angle[i]:.2f} $\pi$")
        plt.legend()
        ax_plain.set_xlabel(r"$P_x$")
        ax_plain.set_ylabel(r"$P_y$")
        ax_plain.set_aspect("equal")
        
        plt.savefig(file("AngularPolar.png"))
        
    #%% PLOT ANGULAR PATTERN PROFILE FOR DIFFERENT AZIMUTHAL ANGLES
        
    if near2far and vm.parallel_assign(1, np_process, parallel) and separate_simulations_needed and make_plots:
        
        freq_index = np.argmin(np.abs(freqs - freq_center))
        index = [list(azimuthal_angle).index(alpha) for alpha in [0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2]]
        
        plt.figure()
        plt.suptitle('Angular Pattern of Au Sphere With {:.1f} nm Radius at {:.1f} nm'.format(r*from_um_factor*1e3,
                                                                                      from_um_factor*1e3/freqs[freq_index]))
        ax_plain = plt.axes()
        for i in index:
            ax_plain.plot(poynting_x[i,:,freq_index], 
                          poynting_z[i,:,freq_index], 
                          ".-", label=rf"$\phi$ = {azimuthal_angle[i]:.2f} $\pi$")
        plt.legend()
        ax_plain.set_xlabel(r"$P_x$")
        ax_plain.set_ylabel(r"$P_z$")
        
        plt.savefig(file("AngularAzimuthal.png"))
        
    if near2far and vm.parallel_assign(0, np_process, parallel) and separate_simulations_needed and make_plots:
        
        freq_index = np.argmin(np.abs(freqs - freq_center))
        index = [list(azimuthal_angle).index(alpha) for alpha in [0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2]]
        
        plt.figure()
        plt.suptitle('Angular Pattern of Au Sphere With {:.1f} nm Radius at {:.1f} nm'.format(r*from_um_factor*1e3,
                                                                                      from_um_factor*1e3/freqs[freq_index]))
        ax_plain = plt.axes()
        for i in index:
            ax_plain.plot(np.sqrt(np.square(poynting_x[i,:,freq_index]) + np.square(poynting_y[i,:,freq_index])), 
                                  poynting_z[i,:,freq_index], ".-", label=rf"$\phi$ = {azimuthal_angle[i]:.2f} $\pi$")
        plt.legend()
        ax_plain.set_xlabel(r"$P_\rho$")
        ax_plain.set_ylabel(r"$P_z$")
        
        plt.savefig(file("AngularAzimuthalAbs.png"))
        
#%%

if __name__ == '__main__':
    main()
