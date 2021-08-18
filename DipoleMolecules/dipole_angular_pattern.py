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
import v_utilities as vu

used_ram, swapped_ram, measure_ram = vm.ram_manager()
measure_ram()

#%% COMMAND LINE FORMATTER

@cli.command()
@cli.option("--from-um-factor", "-fum", "from_um_factor", 
            default=100e-3, type=float,
            help="Conversion of 1 μm to my length unit (i.e. 10e-3=10nm/1μm)")
@cli.option("--resolution-wlen", "-res", "resolution_wlen", 
            default=8, type=int,
            help="Spatial resolution. Minimum number of points in minimum wavelength")
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
            type=float, default=None,
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
@cli.option("--empty-wlen-factor", "-empty", "empty_wlen_factor", 
            type=float, default=0.22,
            help="Empty layer width expressed in multiples of maximum wavelength")
@cli.option("--pml-wlen-factor", "-pml", "pml_wlen_factor", 
            type=float, default=0.38,
            help="PML layer width expressed in multiples of maximum wavelength")
@cli.option("--flux-wlen-factor", "-flux", "flux_wlen_factor", 
            type=float, default=0.15,
            help="Flux box half size expressed in multiples of maximum wavelength")
@cli.option("--flux-padd-factor", "-padd", "flux_padd_factor", 
            type=float, default=.05,
            help="Flux box extra padding expressed in grid points")
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
def main(from_um_factor, resolution_wlen, courant, 
         submerged_index, displacement, surface_index, 
         wlen_center, wlen_width, nfreq, standing,
         empty_wlen_factor, pml_wlen_factor, #flux_padd_factor,
         flux_wlen_factor, flux_padd_factor,
         time_factor_cell, second_time_factor,
         series, folder, n_cores, n_nodes, split_chunks_evenly):

    #%% CLASSIC INPUT PARAMETERS    
    """
    # Simulation size
    from_um_factor = 100e-3 # Conversion of 1 μm to my length unit (=100nm/1μm)
    resolution_wlen = 16 # >=8 pixels per smallest wavelength, i.e. np.floor(8/wvl_min)
    courant = 0.5
    
    # Cell specifications
    displacement = 50 # Displacement of the surface relative to the dipole
    submerged_index = 1 # 1.33 for water
    surface_index = 1 # 1.54 for glass
    
    # Frequency and wavelength
    wlen_center = 650 # Main wavelength range in nm
    wlen_width = 200 # Wavelength band in nm
    nfreq = 100 # Number of discrete frequencies
    standing = True # Perpendicular to surface if true, parallel if false
    
    # Box dimensions
    pml_wlen_factor = 0.38
    empty_wlen_factor = 0.22
    flux_wlen_factor = 0.10
    flux_padd_factor = .05
    
    # Simulation time
    time_factor_cell = 1.2
    second_time_factor = 10
    
    # Saving directories
    series = "TestSpyderMonoch" # "2nd"
    folder = "Test/TestDipole/DipoleStanding" # DataAnalysis/DipoleStandingGlassTest"
    
    # Run configuration
    n_cores = 1
    n_nodes = 1
    split_chunks_evenly = True
    """

    #%% MORE INPUT PARAMETERS
    
    # Simulation size
    resolution = (from_um_factor * 1e3) * resolution_wlen / (wlen_center - wlen_width/2)
    resolution = int( np.ceil(resolution/2) ) * 2
    # resolution = 20
    padd_points = 2
    
    # Frequency and wavelength
    cutoff = 3.2 # Gaussian planewave source's parameter of shape
    nazimuthal = 32 # Suggestion: multiple of 8, to make plots correctly.
    npolar = 40 # Suggestion: multiple of 4, to make plots correctly.
    
    # Run configuration
    english = False
    
    ### TREATED INPUT PARAMETERS
    
    # Cell general specifications: Surface below dipole
    displacement = displacement / ( from_um_factor * 1e3 ) # Now in Meep units
    if surface_index is None:
        surface_index = submerged_index
    
    # Frequency and wavelength
    wlen_center = wlen_center / ( from_um_factor * 1e3 ) # Now in Meep units
    wlen_width = wlen_width / ( from_um_factor * 1e3 ) # Now in Meep units
    freq_center = 1/wlen_center # Hz center frequency in Meep units
    freq_width = 1/wlen_width # Hz range in Meep units from highest to lowest
    
    # Space configuration
    pml_width = pml_wlen_factor * (wlen_center + wlen_width/2) # 0.5 * max(wlen_range)
    empty_width = max(empty_wlen_factor * (wlen_center + wlen_width/2),
                      displacement + padd_points / resolution)
    if submerged_index != surface_index:
        flux_box_size = 2 * empty_width * (1 - flux_padd_factor)
    else:
        flux_box_size = 2 * flux_wlen_factor * (wlen_center + wlen_width/2)
    if flux_box_size == 2 * empty_width:
       flux_box_size = flux_box_size - padd_points/resolution 
    surface_box_size = flux_box_size * (1 - flux_padd_factor)
    if surface_box_size == flux_box_size:
        surface_box_size = surface_box_size - padd_points/resolution
    
    if 2*empty_width <= flux_box_size:
        raise ValueError("Flux box is too big compared to empty space!")
    if displacement >= 0.8 * empty_width:
        raise ValueError("Displacement is too big compared to empty space!")
    
    # Computation
    elapsed = []
    
    n_processes = mp.count_processors()
    parallel_specs = np.array([n_processes, n_cores, n_nodes], dtype=int)
    max_index = np.argmax(parallel_specs)
    for index, item in enumerate(parallel_specs): 
        if item == 0: parallel_specs[index] = 1
    parallel_specs[0:max_index] = np.full(parallel_specs[0:max_index].shape, 
                                          max(parallel_specs))
    n_processes, n_cores, n_nodes = parallel_specs
    parallel = max(parallel_specs) > 1
    del parallel_specs, max_index, index, item
                    
    parallel_assign = vm.parallel_manager(n_processes, parallel)
    
    # Saving directories
    if series is None:
        series = "Test"
    if folder is None:
        folder = "Test"
    trs = vu.BilingualManager(english=english)
    
    params_list = ["from_um_factor", "resolution_wlen", "resolution", "courant",
                   "submerged_index", "displacement", "surface_index",
                   "wlen_center", "wlen_width", "cutoff", "standing",
                   "nfreq", "nazimuthal", "npolar", 
                   "flux_box_size", "surface_box_size", 
                   "empty_wlen_factor", "flux_wlen_factor", "flux_padd_factor",
                   "cell_width", "pml_width", "empty_width",
                   "until_after_sources", "time_factor_cell", "second_time_factor",
                   "elapsed", "parallel", "n_processes", "split_chunks_evenly", 
                   "script", "sysname", "path"]
    
    #%% GENERAL GEOMETRY SETUP
    
    ### ROUND UP ACCORDING TO GRID DISCRETIZATION
    
    empty_width = vu.round_to_multiple(empty_width, 1/resolution, round_up=True)
    pml_width = vu.round_to_multiple(pml_width, 1/resolution, round_up=True)
    
    # symmetries = [mp.Mirror(mp.Y), 
    #               mp.Mirror(mp.Z, phase=-1)]
    # Two mirror planes reduce cell size to 1/4
    # Issue related that lead me to comment this lines:
    # https://github.com/NanoComp/meep/issues/1484
        
    displacement = vu.round_to_multiple(displacement, 1/resolution, round_up=False)
    flux_box_size = vu.round_to_multiple(flux_box_size, 1/resolution, round_up=False)
    surface_box_size = vu.round_to_multiple(surface_box_size, 1/resolution, round_up=False)
    
    ### DEFINE OBJETS
    
    pml_layers = [mp.PML(thickness=pml_width)]
    cell_width = 2 * (pml_width + empty_width)
    cell_size = mp.Vector3(cell_width, cell_width, cell_width)
   
    if surface_index != submerged_index:
        # Cell filled with submerged_index, with a glass below.
        initial_geometry = [mp.Block(material=mp.Medium(index=surface_index),
                                     center=mp.Vector3(
                                         0, 0,
                                         - displacement/2 - surface_box_size/4),
                                     size=mp.Vector3(
                                         surface_box_size, surface_box_size,
                                         surface_box_size/2 - displacement))]
        initial_background = mp.Medium(index=submerged_index)
        # Celled filled with surface_index, with air above
        final_geometry = [mp.Block(material=mp.Medium(index=submerged_index),
                                     center=mp.Vector3(
                                         0, 0,
                                         - displacement/2 + surface_box_size/4),
                                     size=mp.Vector3(
                                         surface_box_size, surface_box_size,
                                         surface_box_size/2 + displacement))]
        final_background = mp.Medium(index=surface_index)
    else:
        initial_geometry = []
        initial_background = mp.Medium(index=submerged_index)
        
    # If required, there will be a certain material surface underneath the dipole

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
    
    # sources = [mp.Source(mp.ContinuousSource(wavelength=wlen_center,
    #                                          is_integrated=True,
    #                                          cutoff=cutoff,
    #                                          width=0),
    #                       center=mp.Vector3(),
    #                       component=polarization)]
    
    until_after_sources = time_factor_cell * cell_width * submerged_index
    # Enough time for the pulse to pass through all the cell
    # Originally: Aprox 3 periods of lowest frequency, using T=λ/c=λ in Meep units 
    # Now: Aprox 3 periods of highest frequency, using T=λ/c=λ in Meep units 
        
    home = vs.get_home()
    sysname = vs.get_sys_name()
    path = os.path.join(home, folder, series)
    if not os.path.isdir(path) and parallel_assign(0):
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
                        default_material=initial_background,
                        output_single_precision=True,
                        split_chunks_evenly=split_chunks_evenly,
                        geometry=initial_geometry)#,
                        #force_complex_fields=True)
    
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

    #%% PLOT CELL

    if parallel_assign(0):
    
        fig, ax = plt.subplots()
        
        # PML borders
        pml_out_square = plt.Rectangle((-cell_width/2, -cell_width/2), 
                                       cell_width, cell_width,
                                       fill=False, edgecolor="m", linestyle="dashed",
                                       hatch='/', 
                                       zorder=-20,
                                       label=trs.choose("PML borders", "Bordes PML"))
        pml_inn_square = plt.Rectangle((-cell_width/2+pml_width,
                                        -cell_width/2+pml_width), 
                                       cell_width - 2*pml_width, 
                                       cell_width - 2*pml_width,
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
            surface_square = plt.Rectangle((-surface_box_size/2, -surface_box_size/2),
                                           surface_box_size,
                                           surface_box_size/2 - displacement,
                                           edgecolor="navy", hatch=r"\\", 
                                           fill=False, zorder=-3,
                                           label=trs.choose(fr"Surface $n$={surface_index}",
                                                            fr"Superficie $n$={surface_index}"))
        
        # Source
        ax.plot(0, 0, "o", color="r", zorder=5, 
                label=trs.choose("Point Dipole ", "Dipolo puntual ") + 
                                 f"{wlen_center * from_um_factor * 1e3:.0f} nm")
        
        # Flux box
        flux_square = plt.Rectangle((-flux_box_size/2,-flux_box_size/2), 
                                    flux_box_size, flux_box_size,
                                    linewidth=1, edgecolor="limegreen", 
                                    linestyle="dashed", fill=False, zorder=10, 
                                    label=trs.choose("Flux box", "Caja de flujo"))
        
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
        plt.legend(bbox_to_anchor=trs.choose( (1.47, 0.5), (1.54, 0.5) ), 
                   loc="center right", frameon=False)
        
        fig.set_size_inches(7.5, 4.8)
        ax.set_aspect("equal")
        plt.xlim(-cell_width/2, cell_width/2)
        plt.ylim(-cell_width/2, cell_width/2)
        plt.xlabel(trs.choose("Position X [Mp.u.]", "Posición X [u.Mp.]"))
        plt.ylabel(trs.choose("Position Z [Mp.u.]", "Posición Z [u.Mp.]"))
        
        plt.annotate(trs.choose(f"1 Meep Unit = {from_um_factor * 1e3:.0f} nm",
                                f"1 Unidad de Meep = {from_um_factor * 1e3:.0f} nm"),
                     (5, 5), xycoords='figure points')
        
        plt.savefig(file("SimBox.png"))
        
        del pml_inn_square, pml_out_square, flux_square
        if submerged_index!=1: del surrounding_square
        if surface_index!=submerged_index: del surface_square
        del fig, box, ax

    #%% BASE SIMULATION: INITIALIZE
    
    temp = time()
    sim.init_sim()
    elapsed.append( time() - temp )
    measure_ram()

    #%% BASE SIMULATION: SIMULATION :D
    
    step_ram_function = lambda sim : measure_ram()
    
    temp = time()
    sim.run(mp.at_beginning(step_ram_function), 
            mp.at_time(int(second_time_factor * until_after_sources / 2), 
                       step_ram_function),
            mp.at_end(step_ram_function),
            # until=second_time_factor * until_after_sources )
            until_after_sources=second_time_factor * until_after_sources )
            # mp.stop_when_fields_decayed(
            #     np.mean(wlen_range), # dT = mean period of source
            #     mp.Ez, # Component of field to check
            #     mp.Vector3(0.5*cell_width - pml_width, 0, 0), # Where to check
            #     1e-3)) # Factor to decay
    elapsed.append( time() - temp )
    del temp
    # Aprox 30 periods of lowest frequency, using T=λ/c=λ in Meep units 
    
    if parallel_assign(0): print("Ended simulation block")
        
    #%% BASE SIMULATION: ANGULAR PATTERN ANALYSIS

    if parallel_assign(0):
        
        freqs = np.linspace(freq_center - freq_center/2, freq_center + freq_width/2, nfreq)
        wlens = 1/freqs
        
        radial_distance = cell_width
        azimuthal_angle = np.arange(0, 2 + 2/nazimuthal, 2/nazimuthal) # in multiples of pi
        polar_angle = np.arange(0, 1 + 1/npolar, 1/npolar)
        
        poynting_x = np.zeros((nazimuthal+1, npolar+1, nfreq))
        poynting_y = np.zeros((nazimuthal+1, npolar+1, nfreq))
        poynting_z = np.zeros((nazimuthal+1, npolar+1, nfreq))
        poynting_r = np.zeros((nazimuthal+1, npolar+1, nfreq))
        
        if surface_index != submerged_index:
            polar_limit = np.arcsin(displacement/radial_distance) + .5
            sim_polar_angle = polar_angle[polar_angle <= polar_limit]
        else:
            sim_polar_angle = polar_angle
        
        for i, phi in enumerate(azimuthal_angle):
            
            for j, theta in enumerate(sim_polar_angle):
                
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
                
                poynting_x[i, j, :] = Px
                poynting_y[i, j, :] = Py
                poynting_z[i, j, :] = Pz
                poynting_r[i, j, :] = np.sqrt( np.square(Px) + np.square(Py) + np.square(Pz) )
            
        del phi, theta, farfield_dict, Px, Py, Pz
        
    # mp.all_wait()
    if parallel_assign(0): print("Ended calculation of far field")
    
    if submerged_index == surface_index and parallel_assign(0):
    
        max_poynting_r = np.max(np.abs(poynting_r))
        
        poynting_x = np.array(poynting_x) / max_poynting_r
        poynting_y = np.array(poynting_y) / max_poynting_r
        poynting_z = np.array(poynting_z) / max_poynting_r
        poynting_r = np.array(poynting_r) / max_poynting_r
        
        if parallel_assign(0): print("Normalized")
    
    #%% BASE SIMULATION: SAVE DATA
    
    for p in params_list: params[p] = eval(p)
    
    os.chdir(path)
    sim_filename = sim.filename_prefix
    sim.filename_prefix = "Base"
    sim.save_near2far("Near2Far", near2far_box)
    mp.all_wait()
    
    if parallel_assign(0): print("Saved near2far flux data")
    
    if parallel_assign(1):
            f = h5.File(file("Base-Near2Far.h5"), "r+")
            for key, par in params.items():
                f[ list(f.keys())[0] ].attrs[key] = par
            f.close()
            del f
    
    if parallel_assign(0): print("Added params to near2far file.")
        
    if parallel_assign(0):
            f = h5.File(file("Base-Results.h5"), "w")
            f["Px"] = poynting_x
            f["Py"] = poynting_y
            f["Pz"] = poynting_z
            f["Pr"] = poynting_r
            for dset in f.values():
                for key, par in params.items():
                    dset.attrs[key] = par
            f.close()
            del f
            # print("Saved far fields")
    mp.all_wait()
    
    if parallel_assign(0): print("Saved far fields")
        
    # if not split_chunks_evenly:
    #     vm.save_chunks(sim, params, path)
    
    f = vs.parallel_hdf_file(file("Base-RAM.h5"), "w", parallel)
    if parallel:
        current_process = mp.my_rank()
        f.create_dataset("RAM", (len(used_ram), n_processes), dtype="float")
        f["RAM"][:, current_process] = used_ram
        f.create_dataset("SWAP", (len(used_ram), n_processes), dtype="int")
        f["SWAP"][:, current_process] = swapped_ram
    else:
        f.create_dataset("RAM", data=used_ram)
        f.create_dataset("SWAP", data=swapped_ram)
    for a in params: f["RAM"].attrs[a] = params[a]
    for a in params: f["SWAP"].attrs[a] = params[a]
    f.close()
    del f
    
    if parallel_assign(0): print("Saved RAM")
    
    mp.all_wait()
    sim.filename_prefix = sim_filename
    sim.reset_meep()
    
    os.chdir(path)
    
    if parallel_assign(0): print("Ready to go with further simulation")
    
    #%% FURTHER SIMULATION
        
    if surface_index!=submerged_index:
    
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
                            default_material=final_background,
                            output_single_precision=True,
                            split_chunks_evenly=split_chunks_evenly,
                            geometry=final_geometry)
        
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
    
    #%%
    
        # #%% PLOT CELL
    
        if parallel_assign(0):
    
            fig, ax = plt.subplots()
            
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
           
            # Surface medium
            surface_square = plt.Rectangle((-cell_width/2, -cell_width/2),
                                            cell_width,
                                            cell_width,
                                            edgecolor="navy", hatch=r"\\", 
                                            fill=False, zorder=-6,
                                            label=trs.choose(fr"Surface $n$={surface_index}",
                                                            fr"Superficie $n$={surface_index}"))
            
            # Surrounding medium
            if submerged_index != 1:
                submerged_color = "blue"
            else:
                submerged_color = "white"            
            surrounding_square_0 = plt.Rectangle((-surface_box_size/2, 
                                                  -displacement),
                                                  surface_box_size,
                                                  surface_box_size/2 + displacement,
                                                  color="white", zorder=-4) 
    
            surrounding_square = plt.Rectangle((-surface_box_size/2, 
                                                -displacement),
                                                surface_box_size,
                                                surface_box_size/2 + displacement,
                                                color=submerged_color, alpha=.1, zorder=-3,
                                                label=trs.choose(fr"Medium $n$={submerged_index}",
                                                                fr"Medio $n$={submerged_index}"))
            
            surrounding_square_2 = plt.Rectangle((-surface_box_size/2, 
                                                  -displacement),
                                                surface_box_size, 
                                                surface_box_size/2 + displacement,
                                                zorder=-2,
                                                edgecolor="navy", fill=False) 
            
            # Source
            ax.plot(0, 0, "o", color="r", zorder=5, 
                    label=trs.choose("Point Dipole", "Dipolo puntual") + 
                                      f"{wlen_center * from_um_factor * 1e3:.0f} nm")
            
            # Flux box
            flux_square = plt.Rectangle((-flux_box_size/2,-flux_box_size/2), 
                                        flux_box_size, flux_box_size,
                                        linewidth=1, edgecolor="limegreen", linestyle="dashed",
                                        fill=False, zorder=10, 
                                        label=trs.choose("Flux box", "Caja de flujo"))
            
            ax.add_patch(surrounding_square_0)
            ax.add_patch(surrounding_square)
            ax.add_patch(surrounding_square_2)
            ax.add_patch(surface_square)
            ax.add_patch(flux_square)
            ax.add_patch(pml_out_square)
            ax.add_patch(pml_inn_square)
            
            # General configuration
            
            box = ax.get_position()
            box.x0 = box.x0 - .15 * (box.x1 - box.x0)
            # box.x1 = box.x1 - .05 * (box.x1 - box.x0)
            box.y1 = box.y1 + .10 * (box.y1 - box.y0)
            ax.set_position(box)
            plt.legend(bbox_to_anchor=trs.choose( (1.47, 0.5), (1.54, 0.5) ), 
                        loc="center right", frameon=False)
            
            fig.set_size_inches(7.5, 4.8)
            ax.set_aspect("equal")
            plt.xlim(-cell_width/2, cell_width/2)
            plt.ylim(-cell_width/2, cell_width/2)
            plt.xlabel(trs.choose("Position X [Mp.u.]", "Posición X [u.Mp.]"))
            plt.ylabel(trs.choose("Position Z [Mp.u.]", "Posición Z [u.Mp.]"))
            
            plt.annotate(trs.choose(f"1 Meep Unit = {from_um_factor * 1e3:.0f} nm",
                                    f"1 Unidad de Meep = {from_um_factor * 1e3:.0f} nm"),
                          (5, 5), xycoords='figure points')
            
            plt.savefig(file("FurtherSimBox.png"))
            
            del pml_inn_square, pml_out_square, flux_square
            del surrounding_square, surrounding_square_0, surrounding_square_2
            del surface_square, submerged_color
            del fig, box, ax
    
    #%%
    
        #% FURTHER SIMULATION: INITIALIZE
        
        temp = time()
        sim.init_sim()
        elapsed.append( time() - temp )
        measure_ram()
    
        #% FURTHER SIMULATION: SIMULATION :D
        
        step_ram_function = lambda sim : measure_ram()
        
        temp = time()
        sim.run(mp.at_beginning(step_ram_function), 
                mp.at_time(int(second_time_factor * until_after_sources / 2), 
                            step_ram_function),
                mp.at_end(step_ram_function),
                until_after_sources=second_time_factor * until_after_sources )
        elapsed.append( time() - temp )
        del temp
        # Aprox 30 periods of lowest frequency, using T=λ/c=λ in Meep units 
        
        if parallel_assign(0): print("Ended simulation")
        
#         #% FURTHER SIMULATION: ANGULAR PATTERN ANALYSIS
    
        if parallel_assign(0):
    
            sim_polar_angle = polar_angle[polar_angle > polar_limit]
            index_limit = list(polar_angle > polar_limit).index(True)
        
            for i, phi in enumerate(azimuthal_angle):
                
                for j, theta in enumerate(sim_polar_angle):
                    
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
                                    
                    poynting_x[i, j+index_limit, :] = Px
                    poynting_y[i, j+index_limit, :] = Py
                    poynting_z[i, j+index_limit, :] = Pz
                    poynting_r[i, j+index_limit, :] = np.sqrt( np.square(Px) + np.square(Py) + np.square(Pz) )
                
            del phi, theta, farfield_dict, Px, Py, Pz
        
        if parallel_assign(0): print("Ended calculation of far field")
                
        if parallel_assign(0):
            poynting_x = np.array(poynting_x) / np.max(poynting_r)
            poynting_y = np.array(poynting_y) / np.max(poynting_r)
            poynting_z = np.array(poynting_z) / np.max(poynting_r)
            poynting_r = np.array(poynting_r) / np.max(poynting_r)
        
        if parallel_assign(0): print("Normalized")

        mp.all_wait()
        
        if parallel_assign(0): print("Ended mp.all_wait()")
    
        #% FURTHER SIMULATION: SAVE DATA
        
        for p in params_list: params[p] = eval(p)
        
        os.chdir(path)
        sim_filename = sim.filename_prefix
        sim.filename_prefix = "Further"
        sim.save_near2far("Near2Far", near2far_box)
        # mp.all_wait()
        
        if parallel_assign(0): print("Saved near2far flux data")
        
        if parallel_assign(1):
                f = h5.File(file("Further-Near2Far.h5"), "r+")
                for key, par in params.items():
                    f[ list(f.keys())[0] ].attrs[key] = par
                f.close()
                del f
                print("Added params to near2far data")
        os.chdir(syshome)
    
        if parallel_assign(0):
                f = h5.File(file("Further-Results.h5"), "w")
                f["Px"] = poynting_x
                f["Py"] = poynting_y
                f["Pz"] = poynting_z
                f["Pr"] = poynting_r
                for dset in f.values():
                    for key, par in params.items():
                        dset.attrs[key] = par
                f.close()
                del f
                print("Saved poynting calculation")
        mp.all_wait()
        
        # if not split_chunks_evenly:
        #     vm.save_chunks(sim, params, path)
        
        f = vs.parallel_hdf_file(file("Further-RAM.h5"), "w", parallel)
        if parallel:
            current_process = mp.my_rank()
            f.create_dataset("RAM", (len(used_ram), n_processes), dtype="float")
            f["RAM"][:, current_process] = used_ram
            f.create_dataset("SWAP", (len(used_ram), n_processes), dtype="int")
            f["SWAP"][:, current_process] = swapped_ram
        else:
            f.create_dataset("RAM", data=used_ram)
            f.create_dataset("SWAP", data=swapped_ram)
        for a in params: f["RAM"].attrs[a] = params[a]
        for a in params: f["SWAP"].attrs[a] = params[a]
        f.close()
        del f
        
        if parallel_assign(0): print("Saved RAM")
        
        sim.filename_prefix = sim_filename
        os.chdir(syshome)
    
    #%% CHOOSE FINAL DATA TO SAVE AND DUMP ALL REST

    if parallel_assign(1):
        if submerged_index != surface_index:
            os.rename(file("Further-RAM.h5"), file("RAM.h5"))
            os.remove(file("Base-RAM.h5"))
            
            os.rename(file("Further-Results.h5"), file("Results.h5"))
            os.remove(file("Base-Results.h5"))
        else:
            os.rename(file("Base-RAM.h5"), file("RAM.h5"))
            os.rename(file("Base-Results.h5"), file("Results.h5"))
    
    #%% PLOT ANGULAR PATTERN IN 3D
    
    if parallel_assign(0):
    
        wlen_chosen = wlen_center
        freq_index = np.argmin(np.abs(freqs - 1/wlen_chosen))
    
        fig = plt.figure()
        plt.suptitle(trs.choose(
            f'Angular Pattern of Point Dipole at {from_um_factor * 1e3 * wlen_center:.0f} nm for ',
            f'Patrón angular de dipolo puntual en {from_um_factor * 1e3 * wlen_center:.0f} nm para ') + 
            f"{wlen_chosen * 1e3 * from_um_factor:.0f} nm")
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
        
    if parallel_assign(1):
        
        wlen_chosen = wlen_center
        freq_index = np.argmin(np.abs(freqs - 1/wlen_chosen))
        index = [list(polar_angle).index(alpha) for alpha in [0, .25, .5, .75, 1]]
        
        plt.figure()
        plt.suptitle(trs.choose(
            f'Angular Pattern of Point Dipole at {from_um_factor * 1e3 * wlen_center:.0f} nm for ',
            f'Patrón angular de dipolo puntual en {from_um_factor * 1e3 * wlen_center:.0f} nm para ') + 
            f"{wlen_chosen * 1e3 * from_um_factor:.0f} nm")
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
        
    if parallel_assign(0):
        
        wlen_chosen = wlen_center
        freq_index = np.argmin(np.abs(freqs - 1/wlen_chosen))
        index = [list(azimuthal_angle).index(alpha) for alpha in [0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2]]
        
        plt.figure()
        plt.suptitle(trs.choose(
            f'Angular Pattern of Point Dipole at {from_um_factor * 1e3 * wlen_center:.0f} nm for ',
            f'Patrón angular de dipolo puntual en {from_um_factor * 1e3 * wlen_center:.0f} nm para ') + 
            f"{wlen_chosen * 1e3 * from_um_factor:.0f} nm")
        ax_plain = plt.axes()
        for i in index:
            ax_plain.plot(poynting_x[i,:,freq_index], 
                          poynting_z[i,:,freq_index], 
                          ".-", label=rf"$\phi$ = {azimuthal_angle[i]:.2f} $\pi$")
        plt.legend()
        ax_plain.set_xlabel(r"$P_x$")
        ax_plain.set_ylabel(r"$P_z$")
        
        plt.savefig(file("AngularAzimuthal.png"))
        
    if parallel_assign(0):
        
        wlen_chosen = wlen_center
        freq_index = np.argmin(np.abs(freqs - 1/wlen_chosen))
        index = [list(azimuthal_angle).index(alpha) for alpha in [0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2]]
        
        plt.figure()
        plt.suptitle(trs.choose(
            f'Angular Pattern of Point Dipole at {from_um_factor * 1e3 * wlen_center:.0f} nm for ',
            f'Patrón angular de dipolo puntual en {from_um_factor * 1e3 * wlen_center:.0f} nm para ') + 
            f"{wlen_chosen * 1e3 * from_um_factor:.0f} nm")
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
