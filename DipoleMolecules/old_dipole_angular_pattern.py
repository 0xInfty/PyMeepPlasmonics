#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Scattering and angular pattern in visible spectrum of a point dipole.

Adapted from Meep Tutorial: Mie Scattering of a Lossless Dielectric Sphere 
+ Near to Far Field Spectra Radiation Pattern of an Antenna
"""

script = "dipole_scattering"

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
import v_meep as vm
import v_save as vs

used_ram, swapped_ram, measure_ram = vm.ram_manager()
measure_ram()

#%% COMMAND LINE FORMATTER

@cli.command()
@cli.option("--from-um-factor", "-fum", "from_um_factor", 
            default=100e-3, type=float,
            help="Conversion of 1 μm to my length unit (i.e. 10e-3=10nm/1μm)")
@cli.option("--resolution-wlen", "-res", "resolution_wlen", 
            default=8, type=int,
            help="Spatial resolution. Minimum number of points in maximum wavelength")
# >=8 pixels per smallest wavelength, i.e. np.floor(8/wvl_min)
@cli.option("--courant", "-c", "courant", 
            type=float, default=0.5,
            help="Courant factor: time discretization from space discretization")
@cli.option("--index", "-i", "submerged_index", 
            type=float, default=1,
            help="Reflective index of sourrounding medium")
@cli.option("--displacement", "-dis", "displacement", default=0, type=float,
            help="Overlap of sphere and surface in nm")
@cli.option("--surface-index", "-si", "surface_index", 
            type=float, default=1,
            help="Reflective index of surface medium")
@cli.option("--wlen-center", "-wlen", "wlen_center", 
            type=float, default=650,
            help="Dipole main wavelength expressed in nm")
@cli.option("--wlen-width", "-wd", "wlen_width", 
            type=float, default=200,
            help="Dipole wavelength amplitude expressed in nm")
@cli.option("--standing", "-stand", type=bool, default=True,
            help="Whether the polarization is perpendicular or parallel to the surface")
@cli.option("--nfreq", "-nfreq", "nfreq", 
            type=int, default=100,
            help="Quantity of frequencies to sample in wavelength range")
@cli.option("--air-wlen-factor", "-air", "air_r_factor", 
            type=float, default=0.15,
            help="Empty layer width expressed in multiples of maximum wavelength")
@cli.option("--pml-wlen-factor", "-pml", "pml_wlen_factor", 
            type=float, default=0.38,
            help="PML layer width expressed in multiples of maximum wavelength")
@cli.option("--flux-wlen-factor", "-flux", "flux_r_factor", 
            type=float, default=0.1,
            help="Flux box padding expressed in multiples of maximum wavelength")
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
@cli.option("--parallel", "-par", type=bool, default=False,
            help="Whether the program is being run in parallel or in serial")
@cli.option("--split-chunks-evenly", "-chev", "split_chunks_evenly", 
            type=bool, default=True,
            help="Whether to split chunks evenly or not during parallel run")
@cli.option("--n-processes", "-np", "n_processes", type=int, default=1,
            help="Number of nuclei used to run the program in parallel")
def main(from_um_factor, resolution_wlen, courant, 
         submerged_index, displacement, surface_index, 
         wlen_center, wlen_width, nfreq, standing,
         air_wlen_factor, pml_wlen_factor, flux_wlen_factor,
         time_factor_cell, second_time_factor,
         series, folder, parallel, n_processes, split_chunks_evenly):

    #%% CLASSIC INPUT PARAMETERS    
    """
    # Simulation size
    from_um_factor = 100e-3 # Conversion of 1 μm to my length unit (=100nm/1μm)
    resolution_wlen = 20 # >=8 pixels per smallest wavelength, i.e. np.floor(8/wvl_min)
    courant = 0.5
    
    # Cell specifications
    displacement = 0 # Displacement of the surface from the bottom of the sphere in nm
    submerged_index = 1.33 # 1.33 for water
    surface_index = 1.54 # 1.54 for glass
    
    # Frequency and wavelength
    wlen_center = 650 # Main wavelength range in nm
    wlen_width = 200 # Wavelength band in nm
    nfreq = 100 # Number of discrete frequencies
    standing = True # Perpendicular to surface if true, parallel if false
    
    # Box dimensions
    pml_wlen_factor = 0.38
    air_wlen_factor = 0.15
    flux_wlen_factor = 0.1
    
    # Simulation time
    time_factor_cell = 1.2
    second_time_factor = 10
    
    # Saving directories
    series = "1stTest"
    folder = "Test/TestDipole/DipoleGlass"
    
    # Configuration
    parallel = False
    n_processes = 1
    split_chunks_evenly = True
    """

    #%% MORE INPUT PARAMETERS
    
    # Simulation size
    resolution = (from_um_factor * 1e3) * resolution_wlen / (wlen_center + wlen_width/2)
    resolution = int( np.ceil(resolution/2) ) * 2
    
    # Frequency and wavelength
    cutoff = 3.2 # Gaussian planewave source's parameter of shape
    nazimuthal = 16
    npolar = 20
    
    ### TREATED INPUT PARAMETERS
    
    # Nanoparticle specifications: Sphere in Vacuum :)
    displacement = displacement / ( from_um_factor * 1e3 ) # Now in Meep units
    
    # Frequency and wavelength
    wlen_center = wlen_center / ( from_um_factor * 1e3 ) # Now in Meep units
    wlen_width = wlen_width / ( from_um_factor * 1e3 ) # Now in Meep units
    freq_center = 1/wlen_center # Hz center frequency in Meep units
    freq_width = 1/wlen_width # Hz range in Meep units from highest to lowest
    
    # Space configuration
    pml_width = pml_wlen_factor * (wlen_center + wlen_width/2) # 0.5 * max(wlen_range)
    air_width = air_wlen_factor * (wlen_center + wlen_width/2)
    flux_box_size = flux_wlen_factor * (wlen_center + wlen_width/2)
    
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
    params_list = ["from_um_factor", "resolution_wlen", "resolution", "courant",
                   "submerged_index", "displacement", "surface_index",
                   "wlen_center", "wlen_width", "cutoff", "standing",
                   "nfreq", "nazimuthal", "npolar", "flux_box_size",
                   "cell_width", "pml_width", "air_width",
                   "until_after_sources", "time_factor_cell", "second_time_factor",
                   "enlapsed", "parallel", "n_processes", "split_chunks_evenly", 
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

    cell_width = 2 * (pml_width + air_width)
    cell_width = cell_width - cell_width%(1/resolution)
    cell_size = mp.Vector3(cell_width, cell_width, cell_width)
    
    # surface_center = r/4 - displacement/2 + cell_width/4
    # surface_center = surface_center - surface_center%(1/resolution)
    # displacement = r/2 + cell_width/2 - 2*surface_center
    
    displacement = displacement - displacement%(1/resolution)
    
    flux_box_size = flux_box_size - flux_box_size%(1/resolution)

    if standing:
        polarization = mp.Ez
    else:
        polarization = mp.Ex
        
    sources = [mp.Source(mp.GaussianSource(freq_center,
                                           fwidth=freq_width,
                                           is_integrated=True,
                                           cutoff=cutoff),
                          center=mp.Vector3(),
                          component=polarization)]
    # Point pulse, polarized parallel or perpendicular to surface
    # (its size parameter is null and it is centered at zero)
    # >> The planewave source extends into the PML 
    # ==> is_integrated=True must be specified
    
    until_after_sources = time_factor_cell * cell_width * submerged_index
    # Enough time for the pulse to pass through all the cell
    # Originally: Aprox 3 periods of lowest frequency, using T=λ/c=λ in Meep units 
    # Now: Aprox 3 periods of highest frequency, using T=λ/c=λ in Meep units 
        
    home = vs.get_home()
    sysname = vs.get_sys_name()
    path = os.path.join(home, folder, series)
    if not os.path.isdir(path) and vm.parallel_assign(0, n_processes, parallel):
        os.makedirs(path)
    file = lambda f : os.path.join(path, f)
        
    #%% BASE SIMULATION: SETUP
    
    measure_ram()
    
    params = {}
    for p in params_list: params[p] = eval(p)
    
    sim = mp.Simulation(resolution=resolution,
                        cell_size=cell_size,
                        boundary_layers=pml_layers,
                        sources=sources,
                        k_point=mp.Vector3(),
                        Courant=courant,
                        default_material=mp.Medium(index=submerged_index),
                        output_single_precision=True,
                        split_chunks_evenly=split_chunks_evenly)
    
    measure_ram()
    
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
    # used_ram.append(used_ram[-1])

    #%% BASE SIMULATION: INITIALIZE
    
    temp = time()
    sim.init_sim()
    enlapsed.append( time() - temp )
    measure_ram()

    #%% BASE SIMULATION: SIMULATION :D
    
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
        
    #%% BASE SIMULATION: ANGULAR PATTERN ANALYSIS

    freqs = np.linspace(freq_center - freq_center/2, freq_center + freq_width/2, nfreq)
    wlens = 1/freqs

    # fraunhofer_distance = 8 * (r**2) / min(wlen_range)        
    # radial_distance = max( 10 * fraunhofer_distance , 1.5 * cell_width/2 ) 
    # radius of far-field circle must be at least Fraunhofer distance
    
    radial_distance = cell_width
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
        
    del phi, theta, farfield_dict, Px, Py, Pz
    
    poynting_x = np.array(poynting_x)
    poynting_y = np.array(poynting_y)
    poynting_z = np.array(poynting_z)
    poynting_r = np.array(poynting_r)

    #%% BASE SIMULATION: SAVE DATA
    
    for p in params_list: params[p] = eval(p)
    
    os.chdir(path)
    sim.save_near2far("BaseNear2Far", near2far_box)
    if vm.parallel_assign(0, np_process, parallel):
            f = h5.File("BaseNear2Far.h5", "r+")
            for key, par in params.items():
                f[ list(f.keys())[0] ].attrs[key] = par
            f.close()
            del f
    os.chdir(syshome)

    if vm.parallel_assign(1, np_process, parallel):
            f = h5.File(file("BaseResults.h5"), "w")
            f["Px"] = poynting_x
            f["Py"] = poynting_y
            f["Pz"] = poynting_z
            f["Pr"] = poynting_r
            for dset in f.values():
                for key, par in params.items():
                    dset.attrs[key] = par
            f.close()
            del f
        
    # if not split_chunks_evenly:
    #     vm.save_chunks(sim, params, path)
    
    if parallel:
        f = h5.File(file("BaseRAM.h5"), "w", driver='mpio', comm=MPI.COMM_WORLD)
        current_process = mp.my_rank()
        f.create_dataset("RAM", (len(used_ram), np_process), dtype="float")
        f["RAM"][:, current_process] = used_ram
        for a in params: f["RAM"].attrs[a] = params[a]
        f.create_dataset("SWAP", (len(used_ram), np_process), dtype="int")
        f["SWAP"][:, current_process] = swapped_ram
        for a in params: f["SWAP"].attrs[a] = params[a]
    else:
        f = h5.File(file("BaseRAM.h5"), "w")
        f.create_dataset("RAM", data=used_ram)
        for a in params: f["RAM"].attrs[a] = params[a]
        f.create_dataset("SWAP", data=swapped_ram)
        for a in params: f["SWAP"].attrs[a] = params[a]
    f.close()
    del f
    
    sim.reset_meep()
    
    #%% FURTHER SIMULATION
        
    if surface_index!=1:
    
        #% FURTHER SIMULATION: SETUP
        
        measure_ram()
        
        params = {}
        for p in params_list: params[p] = eval(p)
        
        sim = mp.Simulation(resolution=resolution,
                            cell_size=cell_size,
                            boundary_layers=pml_layers,
                            sources=sources,
                            k_point=mp.Vector3(),
                            Courant=courant,
                            default_material=mp.Medium(index=surface_index),
                            output_single_precision=True,
                            split_chunks_evenly=split_chunks_evenly)
        
        measure_ram()
        
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
        # used_ram.append(used_ram[-1])
    
        #% FURTHER SIMULATION: INITIALIZE
        
        temp = time()
        sim.init_sim()
        enlapsed.append( time() - temp )
        measure_ram()
    
        #% FURTHER SIMULATION: SIMULATION :D
        
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
            
        #% FURTHER SIMULATION: ANGULAR PATTERN ANALYSIS
            
        poynting_x2 = []
        poynting_y2 = []
        poynting_z2 = []
        poynting_r2 = []
        
        for phi in azimuthal_angle:
            
            poynting_x2.append([])
            poynting_y2.append([])
            poynting_z2.append([])
            poynting_r2.append([])
            
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
                
                poynting_x2[-1].append( Px )
                poynting_y2[-1].append( Py )
                poynting_z2[-1].append( Pz )
                poynting_r2[-1].append( np.sqrt( np.square(Px) + np.square(Py) + np.square(Pz) ) )
        
        del phi, theta, farfield_dict, Px, Py, Pz
        
        poynting_x2 = np.array(poynting_x2)
        poynting_y2 = np.array(poynting_y2)
        poynting_z2 = np.array(poynting_z2)
        poynting_r2 = np.array(poynting_r2)
    
        #% FURTHER SIMULATION: SAVE DATA
        
        for p in params_list: params[p] = eval(p)
        
        os.chdir(path)
        sim.save_near2far("FurtherNear2Far", near2far_box)
        if vm.parallel_assign(0, np_process, parallel):
                f = h5.File("FurtherNear2Far.h5", "r+")
                for key, par in params.items():
                    f[ list(f.keys())[0] ].attrs[key] = par
                f.close()
                del f
        os.chdir(syshome)
    
        if vm.parallel_assign(1, np_process, parallel):
                f = h5.File(file("FurtherResults.h5"), "w")
                f["Px"] = poynting_x2
                f["Py"] = poynting_y2
                f["Pz"] = poynting_z2
                f["Pr"] = poynting_r2
                for dset in f.values():
                    for key, par in params.items():
                        dset.attrs[key] = par
                f.close()
                del f
            
        # if not split_chunks_evenly:
        #     vm.save_chunks(sim, params, path)
        
        if parallel:
            f = h5.File(file("FurtherRAM.h5"), "w", driver='mpio', comm=MPI.COMM_WORLD)
            current_process = mp.my_rank()
            f.create_dataset("RAM", (len(used_ram), np_process), dtype="float")
            f["RAM"][:, current_process] = used_ram
            for a in params: f["RAM"].attrs[a] = params[a]
            f.create_dataset("SWAP", (len(used_ram), np_process), dtype="int")
            f["SWAP"][:, current_process] = swapped_ram
            for a in params: f["SWAP"].attrs[a] = params[a]
        else:
            f = h5.File(file("FurtherRAM.h5"), "w")
            f.create_dataset("RAM", data=used_ram)
            for a in params: f["RAM"].attrs[a] = params[a]
            f.create_dataset("SWAP", data=swapped_ram)
            for a in params: f["SWAP"].attrs[a] = params[a]
        f.close()
        del f
    
    #%% NORMALIZE AND REARANGE DATA
    
    if surface_index!=1:
        max_poynting_r = np.max([np.max(np.abs(poynting_r)), 
                                 np.max(np.abs(poynting_r2))])
    else:
        max_poynting_r = np.max(np.abs(poynting_r))
    
    poynting_x = np.array(poynting_x) / max_poynting_r
    poynting_y = np.array(poynting_y) / max_poynting_r
    poynting_z = np.array(poynting_z) / max_poynting_r
    poynting_r = np.array(poynting_r) / max_poynting_r
    
    if surface_index!=1:
        
        poynting_x0 = np.array(poynting_x)
        poynting_y0 = np.array(poynting_y)
        poynting_z0 = np.array(poynting_z)
        poynting_r0 = np.array(poynting_r)
        
        poynting_x2 = np.array(poynting_x2) / max_poynting_r
        poynting_y2 = np.array(poynting_y2) / max_poynting_r
        poynting_z2 = np.array(poynting_z2) / max_poynting_r
        poynting_r2 = np.array(poynting_r2) / max_poynting_r
        
    #%%    
    
    """
    poynting_x = np.array(poynting_x0)
    poynting_y = np.array(poynting_y0)
    poynting_z = np.array(poynting_z0)
    poynting_r = np.array(poynting_r0)
    """
    
    if surface_index!=1:
        
        polar_limit = np.arcsin(displacement/radial_distance) + .5
        
        for i in range(len(polar_angle)):
            if polar_angle[i] > polar_limit:
                index_limit = i
                break
        
        poynting_x[:, index_limit:, :] = poynting_x2[:, index_limit:, :]
        poynting_y[:, index_limit:, :] = poynting_y2[:, index_limit:, :]
        poynting_z[:, index_limit:, :] = poynting_z2[:, index_limit:, :]
        poynting_r[:, index_limit:, :] = poynting_r2[:, index_limit:, :]
        
        poynting_x[:, index_limit-1, :] = np.array([[np.mean([p2, p0]) for p2, p0 in zip(poy2, poy0)] for poy2, poy0 in zip(poynting_x2[:, index_limit-1, :], poynting_x0[:, index_limit-1, :])])
        poynting_y[:, index_limit-1, :] = np.array([[np.mean([p2, p0]) for p2, p0 in zip(poy2, poy0)] for poy2, poy0 in zip(poynting_y2[:, index_limit-1, :], poynting_y0[:, index_limit-1, :])])
        poynting_z[:, index_limit-1, :] = np.array([[np.mean([p2, p0]) for p2, p0 in zip(poy2, poy0)] for poy2, poy0 in zip(poynting_z2[:, index_limit-1, :], poynting_z0[:, index_limit-1, :])])
        poynting_r[:, index_limit-1, :] = np.sqrt( np.squared(poynting_x[:, index_limit-1, :]) + np.squared(poynting_y[:, index_limit-1, :]) + np.squared(poynting_y[:, index_limit-1, :]))

    #%% SAVE FINAL DATA
    
    if surface_index!=1 and vm.parallel_assign(0, np_process, parallel):
        
        os.remove(file("BaseRAM.h5"))
        os.rename(file("FurtherRAM.h5"), file("RAM.h5"))
    
    elif vm.parallel_assign(0, np_process, parallel):
        
        os.rename(file("BaseRAM.h5"), file("RAM.h5"))
    
    
    #%% PLOT ANGULAR PATTERN IN 3D
    
    if vm.parallel_assign(1, np_process, parallel):
    
        freq_index = np.argmin(np.abs(freqs - freq_center))
    
        fig = plt.figure()
        plt.suptitle(f'Angular Pattern of point dipole molecule at {from_um_factor * 1e3 * wlen_center:.0f} nm')
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
        
    if vm.parallel_assign(0, np_process, parallel):
        
        freq_index = np.argmin(np.abs(freqs - freq_center))
        index = [list(polar_angle).index(alpha) for alpha in [0, .25, .5, .75, 1]]
        
        plt.figure()
        plt.suptitle(f'Angular Pattern of point dipole molecule at {from_um_factor * 1e3 * wlen_center:.0f} nm')
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
        
    if vm.parallel_assign(1, np_process, parallel):
        
        freq_index = np.argmin(np.abs(freqs - freq_center))
        index = [list(azimuthal_angle).index(alpha) for alpha in [0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2]]
        
        plt.figure()
        plt.suptitle(f'Angular Pattern of point dipole molecule at {from_um_factor * 1e3 * wlen_center:.0f} nm')
        ax_plain = plt.axes()
        for i in index:
            ax_plain.plot(poynting_x[i,:,freq_index], 
                          poynting_z[i,:,freq_index], 
                          ".-", label=rf"$\phi$ = {azimuthal_angle[i]:.2f} $\pi$")
        plt.legend()
        ax_plain.set_xlabel(r"$P_x$")
        ax_plain.set_ylabel(r"$P_z$")
        
        plt.savefig(file("AngularAzimuthal.png"))
        
    if vm.parallel_assign(0, np_process, parallel):
        
        freq_index = np.argmin(np.abs(freqs - freq_center))
        index = [list(azimuthal_angle).index(alpha) for alpha in [0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2]]
        
        plt.figure()
        plt.suptitle(f'Angular Pattern of point dipole molecule at {from_um_factor * 1e3 * wlen_center:.0f} nm')
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
