#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Scattering efficiency in visible spectrum of 120nm-diameter Au sphere.

Adapted from Meep Tutorial: Mie Scattering of a Lossless Dielectric Sphere
"""

script = "au_mie_scattering"

from socket import gethostname
if "Nano" in gethostname():
    syshome = "/home/nanofisica/Documents/Vale/NanoMeepPlasmonics"
elif "vall" in gethostname():
    syshome = "/home/vall/Documents/Thesis/NanoMeepPlasmonics"
else:
    syshome = "/nfs/home/vpais/NanoMeepPlasmonics"
    # raise ValueError("Your PC must be registered at the top of this code")

import sys
sys.path.append(syshome)
sys.path.append(syshome+"/PlotRoutines")

import click as cli
import meep as mp
import numpy as np
import os
import vmp_materials as vml
import vmp_utilities as vmu
import v_save as vs
import v_utilities as vu

from np_planewave_cell_plot import plot_np_planewave_cell
from np_pulse_scattering_plot import mid_plots_np_scattering
from np_pulse_scattering_plot import plots_np_scattering

rm = vmu.ResourcesMonitor()
rm.measure_ram()

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
@cli.option("--surface-index", "-si", "surface_index", 
            type=float, default=None,
            help="Reflective index of surface medium")
@cli.option("--overlap", "-over", "overlap", default=0, type=float,
            help="Overlap of sphere and surface in nm")
@cli.option("--wlen-range", "-wr", "wlen_range", 
            type=cli.Tuple([float, float]), default=(450,600),
            help="Wavelength range expressed in nm")
@cli.option("--nfreq", "-nfreq", "nfreq", 
            type=int, default=100,
            help="Quantity of frequencies to sample in wavelength range")
@cli.option("--empty-r-factor", "-empty", "empty_r_factor", 
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
@cli.option("--split-chunks-evenly", "-chev", "split_chunks_evenly", 
            type=bool, default=True,
            help="Whether to split chunks evenly or not during parallel run")
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
@cli.option("--load-resources", "-loadr", "load_resources", 
            type=bool, default=False,
            help="Whether to necessarily load monitored resources, if possible, or not")
@cli.option("--near2far", "-n2f", "near2far", 
            type=bool, default=False,
            help="Whether to calculate angular pattern or not.")
@cli.option("--make-plots", "-plt", "make_plots", 
            type=bool, default=True,
            help="Whether to make plots while running or not.")
def main(from_um_factor, resolution, courant, 
         r, material, paper, reference, submerged_index, 
         overlap, surface_index, wlen_range, nfreq,
         empty_r_factor, pml_wlen_factor, flux_r_factor,
         time_factor_cell, second_time_factor,
         series, folder, 
         n_cores, n_nodes, split_chunks_evenly,
         load_flux, load_chunks, load_resources, near2far, make_plots):

    #%% CLASSIC INPUT PARAMETERS    
    
    if any('SPYDER' in name for name in os.environ):
        
        rm.reset()
        
        # Simulation size
        from_um_factor = 10e-3 # Conversion of 1 μm to my length unit (=10nm/1μm)
        resolution = 2 # >=8 pixels per smallest wavelength, i.e. np.floor(8/wvl_min)
        courant = 0.5
        
        # Nanoparticle specifications: Sphere in Vacuum :)
        material = "Au"
        r = 51.5  # Radius of sphere in nm
        paper = "R"
        reference = "Meep"
        overlap = 0 # Upwards displacement of the surface from the bottom of the sphere in nm
        submerged_index = 1.33 # 1.33 for water
        surface_index = None # 1.54 for glass
        
        # Frequency and wavelength
        wlen_range = np.array([450,600]) # Wavelength range in nm
        nfreq = 100
        
        # Box dimensions
        pml_wlen_factor = 0.38
        empty_r_factor = 0.5
        flux_r_factor = 0 #0.1
        
        # Simulation time
        time_factor_cell = 1.2
        second_time_factor = 10
        
        # Saving directories
        series = "Test/TestNPScatt"
        folder = "Test"
        
        # Configuration
        n_cores = 1
        n_nodes = 1
        split_chunks_evenly = True
        load_flux = False
        load_chunks = True
        load_resources = False
        near2far = False
        make_plots = True
        
        print("Loaded Spyder parameters")

    #%% MORE INPUT PARAMETERS
       
    # Simulation size
    resolution_wlen = 8
    resolution_bandwidth = 10
    resolution_nanoparticle = 5
    
    # Frequency and wavelength
    cutoff = 3.2 # Gaussian planewave source's parameter of shape
    nazimuthal = 16
    npolar = 20
    
    # Run configuration
    english = False
    
    ### TREATED INPUT PARAMETERS
    
    # Computation
    pm = vmu.ParallelManager(n_cores, n_nodes)
    n_processes, n_cores, n_nodes = pm.specs
    parallel = pm.parallel
    
    # Simulation size
    resolution_wlen = (from_um_factor * 1e3) * resolution_wlen / min(wlen_range)
    resolution_wlen = int( np.ceil(resolution_wlen) )
    resolution_nanoparticle = (from_um_factor * 1e3) * resolution_nanoparticle / (2 * r)
    resolution_nanoparticle = int( np.ceil(resolution_nanoparticle) )
    resolution_bandwidth = (from_um_factor * 1e3) * resolution_bandwidth / (max(wlen_range) - min(wlen_range))
    resolution_bandwidth = int( np.ceil(resolution_bandwidth) )
    min_resolution = max(resolution_wlen, 
                         resolution_bandwidth,
                         resolution_nanoparticle)
    
    if min_resolution > resolution:
        pm.log(f"Resolution will be raised from {resolution} to {min_resolution}")
        resolution = min_resolution
    del min_resolution
    
    # Nanoparticle specifications: Sphere in Vacuum :)
    r = r / ( from_um_factor * 1e3 ) # Now in Meep units
    if reference=="Meep": 
        medium = vml.import_medium(material, from_um_factor=from_um_factor, paper=paper)
        # Importing material constants dependant on frequency from Meep Library
    elif reference=="RIinfo":
        medium = vml.MediumFromFile(material, paper=paper, reference=reference, from_um_factor=from_um_factor)
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
    empty_width = empty_r_factor * r # 0.5 * max(wlen_range)
    flux_box_size = 2 * ( 1 + flux_r_factor ) * r
    cell_width = 2 * (pml_width + empty_width + r)
    
    # Time configuration
    until_after_sources = time_factor_cell * cell_width
    # Enough time for the pulse to pass through all the cell
    # Originally: Aprox 3 periods of lowest frequency, using T=λ/c=λ in Meep units 
    
    # Saving directories
    sa = vmu.SavingAssistant(series, folder)
    home = sa.home
    sysname = sa.sysname
    path = sa.path
    
    trs = vu.BilingualManager(english=english)
    
    params_list = ["from_um_factor", "resolution", "courant",
                   "material", "r", "paper", "reference", "submerged_index",
                   "overlap", "surface_index",
                   "wlen_range", "nfreq", "nazimuthal", "npolar", "cutoff", "flux_box_size",
                   "cell_width", "pml_width", "empty_width", "source_center",
                   "until_after_sources", "time_factor_cell", "second_time_factor",
                   "parallel", "n_processes", "n_cores", "n_nodes",
                   "split_chunks_evenly", "near2far",
                   "script", "sysname", "path"]
    
    #%% GENERAL GEOMETRY SETUP
    
    ### ROUND UP ACCORDING TO GRID DISCRETIZATION
    
    pml_width = vu.round_to_multiple(pml_width, 1/resolution)
    cell_width = vu.round_to_multiple(cell_width/2, 1/resolution)*2
    empty_width = cell_width/2 - r - pml_width
    overlap = vu.round_to_multiple(overlap - r, 1/resolution) + r
    source_center = -0.5*cell_width + pml_width
    flux_box_size = vu.round_to_multiple(flux_box_size/2, 1/resolution, round_up=True)*2
    
    until_after_sources = vu.round_to_multiple(until_after_sources, courant/resolution, round_up=True)
    
    ### DEFINE OBJETS
    
    pml_layers = [mp.PML(thickness=pml_width)]
    cell_size = mp.Vector3(cell_width, cell_width, cell_width)
        
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
    
    #%% PLOT CELL

    params = {}
    for p in params_list: params[p] = eval(p)

    if pm.assign(0):
        
        plot_np_planewave_cell(params, series, folder,
                               with_flux_box=True, with_nanoparticle=False, 
                               english=trs.english)
        
    #%% FIRST RUN
    
    rm.measure_ram()
    
    stable, max_courant = vmu.check_stability(params)
    if stable:
        pm.log("As a whole, the simulation should be stable")
    else:
        pm.log("As a whole, the simulation might be unstable")
        pm.log(f"Recommended maximum Courant factor is {max_courant}")
    del stable, max_courant
        
    if load_flux:
        try:
            flux_path = vmu.check_midflux(params)[-1]
            if load_resources and sysname != "TC":
                if os.path.isfile( os.path.join(flux_path, "Resources.h5") ):
                    flux_needed = False
                    pm.log("Found resources")
                else:
                    flux_needed = True
                    pm.log("Didn't find resources")
            else:
                flux_needed  = False
        except:
            flux_needed = True
    else:
        flux_needed = True
        
    if load_chunks and not split_chunks_evenly:
        try:
            chunks_path = vmu.check_chunks(params)[-1]
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
                            Courant=courant,
                            geometry=final_geometry,
                            sources=sources,
                            k_point=mp.Vector3(),
                            default_material=mp.Medium(index=submerged_index),
                            cell_size=cell_size,
                            boundary_layers=pml_layers,
                            output_single_precision=True,
                            split_chunks_evenly=split_chunks_evenly)
                            # symmetries=symmetries,
    
        sim.init_sim()
        
        chunks_path = vmu.save_chunks(sim, params, path)         
        chunk_layout = os.path.join(chunks_path, "Layout.h5")
        
        del sim
        
    if not flux_needed:
        if sysname != "TC":
            rm.load( os.path.join(flux_path, "Resources.h5") )
    else:
        
        #% FIRST RUN: SET UP
        
        rm.measure_ram()
        
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
        
        rm.measure_ram()

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
        
        rm.measure_ram()
        
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
            rm.measure_ram()
        else:
            near2far_box = None
            # rm.used_ram.append(rm.used_ram[-1])
        
        #% FIRST RUN: INITIALIZE
        
        rm.start_measure_time()
        sim.init_sim()
        rm.end_measure_time()
        rm.measure_ram()
        
        step_ram_function = lambda sim : rm.measure_ram()
        
        #% FIRST RUN: SIMULATION NEEDED TO NORMALIZE
        
        rm.start_measure_time()
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
        rm.end_measure_time()
        
        #% SAVE MID DATA

        for p in params_list: params[p] = eval(p)

        flux_path =  vmu.save_midflux(sim, box_x1, box_x2, box_y1, 
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
        
        header_mid = [r"Longitud de onda $\lambda$ [nm]", 
                      "Flujo X10 [u.a.]", "Flujo X20 [u.a]",
                      "Flujo Y10 [u.a]", "Flujo Y20 [u.a]",
                      "Flujo Z10 [u.a]", "Flujo Z20 [u.a]"]
        
        if pm.assign(0):
            vs.savetxt(sa.file("MidFlux.txt"), data_mid, 
                       header=header_mid, footer=params)

        if not split_chunks_evenly:
            vmu.save_chunks(sim, params, path)
            
        if sysname != "TC":
            rm.save(os.path.join(flux_path, "Resources.h5"), params)
            rm.save(sa.file("Resources.h5"), params)

        #% PLOT FLUX FOURIER MID DATA
        
        if make_plots: mid_plots_np_scattering(series, folder, 
                                               english=trs.english)
            
        sim.reset_meep()
        del data_mid, box_x2_flux0, box_y1_flux0, box_y2_flux0
        del box_z1_flux0, box_z2_flux0, header_mid
    
    #%% SECOND RUN: SETUP
    
    rm.measure_ram()
    
    sim = mp.Simulation(resolution=resolution,
                        Courant=courant,
                        geometry=final_geometry,
                        sources=sources,
                        k_point=mp.Vector3(),
                        default_material=mp.Medium(index=submerged_index),
                        cell_size=cell_size,
                        boundary_layers=pml_layers,
                        output_single_precision=True,
                        split_chunks_evenly=split_chunks_evenly,
                        chunk_layout=chunk_layout)
                        # symmetries=symmetries,
    
    rm.measure_ram()
    
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
    
    rm.measure_ram()

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
        rm.measure_ram()
    else:
        near2far_box = None
        separate_simulations_needed = True
        
        # rm.used_ram.append(rm.used_ram[-1])
        
    #%% SECOND RUN: PLOT CELL
    
    for p in params_list: params[p] = eval(p)
    params["flux_path"] = flux_path
    
    if pm.assign(0):
        plot_np_planewave_cell(params, series, folder,
                               with_flux_box=True, with_nanoparticle=True, 
                               english=trs.english)

    #%% SECOND RUN: INITIALIZE
    
    rm.start_measure_time()
    sim.init_sim()
    rm.end_measure_time()
    rm.measure_ram()
    
    #%% LOAD FLUX FROM FILE
    
    vmu.load_midflux(sim, box_x1, box_x2, box_y1, box_y2, box_z1, box_z2, 
                    near2far_box, flux_path)
        
    rm.measure_ram()

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
     
    rm.start_measure_time()
    sim.load_minus_flux_data(box_x1, box_x1_data)
    sim.load_minus_flux_data(box_x2, box_x2_data)
    sim.load_minus_flux_data(box_y1, box_y1_data)
    sim.load_minus_flux_data(box_y2, box_y2_data)
    sim.load_minus_flux_data(box_z1, box_z1_data)
    sim.load_minus_flux_data(box_z2, box_z2_data)
    if near2far and not separate_simulations_needed: 
        sim.load_minus_near2far_data(near2far_box, near2far_data)
    rm.end_measure_time()
    del box_x1_data, box_x2_data, box_y1_data, box_y2_data
    del box_z1_data, box_z2_data
    if near2far and not separate_simulations_needed:  
        del near2far_data
    
    rm.measure_ram()

    #%% SECOND RUN: SIMULATION :D
    
    step_ram_function = lambda sim : rm.measure_ram()
    
    rm.start_measure_time()
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
    rm.end_measure_time()
    # Aprox 30 periods of lowest frequency, using T=λ/c=λ in Meep units 
    
    box_x1_flux = np.asarray(mp.get_fluxes(box_x1))
    box_x2_flux = np.asarray(mp.get_fluxes(box_x2))
    box_y1_flux = np.asarray(mp.get_fluxes(box_y1))
    box_y2_flux = np.asarray(mp.get_fluxes(box_y2))
    box_z1_flux = np.asarray(mp.get_fluxes(box_z1))
    box_z2_flux = np.asarray(mp.get_fluxes(box_z2))
    
    #%% SCATTERING ANALYSIS
    
    scatt_flux = box_x2_flux - box_x1_flux
    scatt_flux = scatt_flux + box_y2_flux - box_y1_flux
    scatt_flux = scatt_flux + box_z2_flux - box_z1_flux
    
    intensity = box_x1_flux0/(flux_box_size)**2
    # Flux of one of the six monitor planes / Área
    # (the closest one, facing the planewave source) 
    # This is why the six sides of the flux box are separated
    # (Otherwise, the box could've been one flux object with weights ±1 per side)
    
    scatt_cross_section = np.divide(scatt_flux, intensity)
    # Scattering cross section σ = 
    # = scattered power in all directions / incident intensity.
    
    scatt_eff_meep = scatt_cross_section / (np.pi*r**2)
    # Scattering efficiency =
    # = scattering cross section / cross sectional area of the sphere
    
    freqs = np.array(freqs)
    wlens = 1e3 * from_um_factor / freqs 
    scatt_eff_theory = vml.sigma_scatt_meep(r * from_um_factor * 1e3, # Radius [nm]
                                            material, paper, 
                                            wlens, # Wavelength [nm]
                                            surrounding_index=submerged_index,
                                            asEfficiency=True)
    # Results are validated by comparing with analytic Mie theory
    
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
    
    elif separate_simulations_needed and near2far: 
        pm.log("Beware! Near2far is not supported yet for nanoparticle placed on a surface.")

    #%% SAVE FINAL DATA
    
    for p in params_list: params[p] = eval(p)

    data = np.array([1e3*from_um_factor/freqs, scatt_eff_meep, scatt_eff_theory]).T
    
    header = [r"Longitud de onda $\lambda$ [nm]", 
              "Sección eficaz efectiva (Meep) [u.a.]", 
              "Sección eficaz efectiva (Theory) [u.a.]"]
    
    data_base = np.array([1e3*from_um_factor/freqs, box_x1_flux0, box_x1_flux,
                          box_x2_flux, box_y1_flux, box_y2_flux, 
                          box_z1_flux, box_z2_flux, 
                          intensity, scatt_flux, scatt_cross_section]).T
    
    header_base = [r"Longitud de onda $\lambda$ [nm]", 
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
    
    if pm.assign(0):
        vs.savetxt(sa.file("Results.txt"), data, 
                   header=header, footer=params)
        vs.savetxt(sa.file("BaseResults.txt"), data_base, 
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
        
        if pm.assign(1):
            vs.savetxt(sa.file("Near2FarResults.txt"), np.array(data_near2far).T, 
                       header=header_near2far, footer=params)
        del header_near2far, data_near2far
        
    if not split_chunks_evenly:
        vmu.save_chunks(sim, params, path)
        
    if sysname != "TC":
        rm.save(sa.file("Resources.h5"), params)
    
    #%% MAKE PLOTS, IF NEEDED
    
    if make_plots: plots_np_scattering(series, folder, near2far, 
                                       separate_simulations_needed, 
                                       english=trs.english)
        
#%%

if __name__ == '__main__':
    main()
