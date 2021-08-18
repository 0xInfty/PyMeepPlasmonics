#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Field of Au sphere given a visible monochromatic incident wave.
"""

script = "np_monoch_field"

from socket import gethostname
if "Nano" in gethostname():
    syshome = "/home/nanofisica/Documents/Vale/ThesisPython"
elif "vall" in gethostname():
    syshome = "/home/vall/Documents/Thesis/ThesisPython"
else:
    raise ValueError("Your PC must be registered at the top of this code")

import sys
sys.path.append(syshome)
sys.path.append(syshome+"/PlotRoutines")

import click as cli
import imageio as mim
import h5py as h5
import matplotlib.pyplot as plt
import meep as mp
import numpy as np
import os
from time import time
from v_materials import import_medium
import v_meep as vm
import v_save as vs
import v_utilities as vu
from np_monoch_field import plots_monoch_field

used_ram, swapped_ram, measure_ram = vm.ram_manager()
measure_ram()

#%% COMMAND LINE FORMATTER

@cli.command()
@cli.option("--resolution", "-res", required=True, type=int,
            help="Spatial resolution. Number of divisions of each Meep unit")
@cli.option("--from-um-factor", "-fum", "from_um_factor", 
            default=10e-3, type=float,
            help="Conversion of 1 μm to my length unit (i.e. 10e-3=10nm/1μm)")
# >=8 pixels per smallest wavelength, i.e. np.floor(8/wvl_min)
@cli.option("--courant", "-c", "courant", 
            type=float, default=0.5,
            help="Courant factor: time discretization from space discretization")
@cli.option("--radius", "-r", "r", default=60, type=int,
            help="Radius of sphere expressed in nm")
@cli.option("--material", "-mat", default="Au", type=str,
            help="Material of spherical nanoparticle")
@cli.option("--paper", "-pp", "paper", type=str, default="R",
            help="Source of inner material experimental data. Options: 'JC'/'R'/'P'")
@cli.option("--index", "-i", "submerged_index", 
            type=float, default=1,
            help="Reflective index of sourrounding medium")
@cli.option("--overlap", "-over", "overlap", default=0, type=float,
            help="Overlap of sphere and surface in nm")
@cli.option("--surface-index", "-si", "surface_index", 
            type=float, default=None,
            help="Reflective index of surface medium")
@cli.option("--wlen", "-wl", "wlen", default=532, type=float,
            help="Wavelength expressed in nm")
@cli.option("--empty-r-factor", "-empty", "empty_r_factor", 
            type=float, default=0.5,
            help="Empty layer width expressed in multiples of radius")
@cli.option("--pml-wlen-factor", "-pml", "pml_wlen_factor", 
            type=float, default=0.38,
            help="PML layer width expressed in multiples of maximum wavelength")
@cli.option("--time-period-factor", "-tpfc", "time_period_factor", 
            type=float, default=10,
            help="Simulation total time expressed as multiples of periods")
@cli.option("--series", "-s", type=str, 
            help="Series name used to create a folder and save files")
@cli.option("--folder", "-f", type=str, 
            help="Series folder used to save files")
@cli.option("--split-chunks-evenly", "-chev", "split_chunks_evenly", 
            type=bool, default=True,
            help="Whether to split chunks evenly or not during parallel run")
@cli.option("--n-cores", "-nc", "n_cores", type=int, default=0,
            help="Number of cores used to run the program in parallel")
@cli.option("--n-nodes", "-nn", "n_nodes", type=int, default=0,
            help="Number of nodes used to run the program in parallel")
@cli.option("--hfield", "-hf", "hfield", type=bool, default=False,
            help="Whether to also save and analyse magnetic field or not")
@cli.option("--make-plots", "-plt", "make_plots", 
            type=bool, default=True,
            help="Whether to make plots while running or not.")
@cli.option("--make-gifs", "-gifs", "make_gifs", 
            type=bool, default=False,
            help="Whether to make gifs while running or not.")
def main(from_um_factor, resolution, courant,
         r, material, paper, wlen, submerged_index, overlap, surface_index,
         empty_r_factor, pml_wlen_factor, 
         time_period_factor,
         series, folder,
         n_cores, n_nodes, split_chunks_evenly,
         hfield, make_plots, make_gifs):
    
    #%% DEFAULT PARAMETERS
    
    """
    # Sim configuration
    resolution = 2
    from_um_factor = 10e-3
    courant = 0.5
    
    # Cell configuration
    r = 51.5 # nm
    material = "Au"
    paper = "R"
    overlap = 0 # Upwards displacement of the surface from the bottom of the sphere in nm
    submerged_index = 1 # 1.33 for water
    surface_index = 1 # 1.54 for glass
    
    # Source configuration
    wlen = 532
    
    # Box spatial dimensions
    pml_wlen_factor = 0.38
    empty_r_factor = 0.5
    
    # Sim temporal dimension
    time_period_factor = 10    
    
    # Files configuration
    series = None
    folder = None
    
    # Run configuration
    parallel = False
    n_processes = 1
    n_cores = 1
    n_nodes = 1
    split_chunks_evenly = True
    
    # Routine configuration
    hfield = True
    make_plots = True
    make_gifs = True
    """
    
    #%% ADDITIONAL PARAMETERS
    
    # Field Measurements
    n_period_line = 100
    n_period_plane = 100
    
    # Routine configuration
    english = False
    
    #%% TREATED PARAMETERS
    
    # Au sphere
    r = r  / ( from_um_factor * 1e3 )  # Radius of sphere now in Meep units
    medium = import_medium("Au", from_um_factor, paper=paper) # Medium of sphere: gold (Au)
    
    # Frequency and wavelength
    wlen = wlen / ( from_um_factor * 1e3 ) # Wavelength now in Meep units
    period = submerged_index * wlen
    
    # Space configuration
    pml_width = pml_wlen_factor * wlen # 0.5 * wlen
    empty_width = empty_r_factor * r # 0.5 * max(wlen_range)
    
    # Time configuration
    until_time = time_period_factor * period
    period_line = period / n_period_line
    period_plane = period / n_period_line    
    
    # Computation time
    elapsed = []
    
    # Saving directories
    if series is None:
        series = f"TestAuSphereField{2*r*from_um_factor*1e3:.0f}WLen{wlen*from_um_factor*1e3:.0f}"
    if folder is None:
        folder = "AuMieSphere/AuSphereField"
    
    params_list = ["from_um_factor", "resolution", "courant",
                   "material", "r", "paper", 
                   "submerged_index", "wlen", "surface_index", "overlap",
                   "cell_width", "pml_width", "empty_width", "source_center",
                   "until_time", "time_period_factor", 
                   "n_period_line", "n_period_plane", "period_line", "period_plane",
                   "elapsed", "parallel", "n_processes", "n_cores", "n_nodes",
                   "split_chunks_evenly", "hfield",
                   "script", "sysname", "path"]
    
    #%% GENERAL GEOMETRY SETUP
    
    empty_width = empty_width - empty_width%(1/resolution)
    
    pml_width = pml_width - pml_width%(1/resolution)
    pml_layers = [mp.PML(thickness=pml_width)]
    
    # symmetries = [mp.Mirror(mp.Y), 
    #               mp.Mirror(mp.Z, phase=-1)]
    # Two mirror planes reduce cell size to 1/4
    
    cell_width = 2 * (pml_width + empty_width + r)
    cell_width = cell_width - cell_width%(1/resolution)
    cell_size = mp.Vector3(cell_width, cell_width, cell_width)
    
    source_center = -0.5*cell_width + pml_width
    sources = [mp.Source(mp.ContinuousSource(wavelength=wlen, 
                                             is_integrated=True),
                         center=mp.Vector3(source_center),
                         size=mp.Vector3(0, cell_width, cell_width),
                         component=mp.Ez)]
    # Ez-polarized monochromatic planewave 
    # (its size parameter fills the entire cell in 2d)
    # The planewave source extends into the PML 
    # ==> is_integrated=True must be specified
    
    until_time = until_time - until_time%(courant/resolution)
    period_line = period_line - period_line%(courant/resolution)
    period_plane = period_plane - period_plane%(courant/resolution)
    
    nanoparticle = mp.Sphere(material=medium,
                             center=mp.Vector3(),
                             radius=r)
    # Au sphere with frequency-dependant characteristics imported from Meep.
    
    if surface_index != submerged_index:
        surface = mp.Block(material=mp.Medium(index=surface_index),
                           center=mp.Vector3(
                               r/2 - overlap/2 + cell_width/4,
                               0, 0),
                           size=mp.Vector3(
                               cell_width/2 - r + overlap,
                               cell_width, cell_width))
        geometry = [surface, nanoparticle]
    else:
        geometry = [nanoparticle]
    # If required, a certain material surface underneath it

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
    
    home = vs.get_home()
    sysname = vs.get_sys_name()
    path = os.path.join(home, folder, series)
    if not os.path.isdir(path) and parallel_assign(0):
        os.makedirs(path)
    file = lambda f : os.path.join(path, f)
    
    trs = vu.BilingualManager(english=english)
    
    #%% PLOT CELL

    if parallel_assign(0) and make_plots:
        
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
       
        # Surrounding medium
        if submerged_index != 1:
            surrounding_square = plt.Rectangle((-cell_width/2, -cell_width/2),
                                               cell_width, cell_width,
                                               color="blue", alpha=.1, zorder=-6,
                                               label=trs.choose(fr"Medium $n$={submerged_index}",
                                                                fr"Medio $n$={submerged_index}"))
    
        # Nanoparticle
        if material=="Au":
            circle_color = "gold"
        elif material=="Ag":
            circle_color="silver"
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
        ax.hlines(0, -cell_width/2, cell_width/2,
                  color="limegreen", linestyle=":", zorder=7, 
                  label=trs.choose("Sampling Line", "Línea de muestreo"))
        
        
        
        # Sampling plane
        ax.vlines(0, -cell_width/2, cell_width/2,
                  color="limegreen", linestyle="dashed", zorder=7, 
                  label=trs.choose("Sampling Plane", "Plano de muestreo"))
        
        ax.add_patch(circle)
        if submerged_index!=1: ax.add_patch(surrounding_square)
        ax.add_patch(pml_out_square)
        ax.add_patch(pml_inn_square)
        
        # General configuration
        box = ax.get_position()
        box.x0 = box.x0 - .26 * (box.x1 - box.x0)
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
        
        del pml_inn_square, pml_out_square, circle, circle_color
        if submerged_index!=1: del surrounding_square
        del fig, box, ax
    
    #%% INITIALIZE
    
    params = {}
    for p in params_list: params[p] = eval(p)
    
    # stable, max_courant = vm.check_stability(params)
    # if stable:
    #     print("As a whole, the simulation should be stable")
    # else:
    #     print("As a whole, the simulation could not be stable")
    #     print(f"Recommended maximum courant factor is {max_courant}")
    # del stable, max_courant
    
    measure_ram()
    
    sim = mp.Simulation(resolution=resolution,
                        Courant=courant,
                        geometry=geometry,
                        sources=sources,
                        k_point=mp.Vector3(),
                        default_material=mp.Medium(index=submerged_index),
                        cell_size=cell_size,
                        boundary_layers=pml_layers,
                        output_single_precision=True,
                        split_chunks_evenly=split_chunks_evenly)
    
    measure_ram()
    
    temp = time()
    sim.init_sim()
    elapsed.append(time()-temp)
    
    measure_ram()
    
    #%% DEFINE SAVE STEP FUNCTIONS
    
    sampling_line = mp.Volume(center=mp.Vector3(),
                             size=mp.Vector3(cell_width),
                             dims=1)
    sampling_plane = mp.Volume(center=mp.Vector3(),
                              size=mp.Vector3(0, cell_width, cell_width),
                              dims=2)
    
    save_line = mp.in_volume(sampling_line, 
                             mp.to_appended("Lines",
                                            mp.output_efield_z))
    save_plane = mp.in_volume(sampling_plane, 
                              mp.to_appended("Planes",
                                             mp.output_efield_z))
    
    to_do_while_running = [mp.at_every(period_line, save_line),
                           mp.at_every(period_plane, save_plane)]
    
    if hfield:
        save_hline = mp.in_volume(sampling_line, 
                                  mp.to_appended("HLines",
                                                 mp.output_hfield_y))
        save_hplane = mp.in_volume(sampling_plane, 
                                   mp.to_appended("HPlanes",
                                                  mp.output_hfield_y))
        to_do_while_running = [*to_do_while_running, 
                               mp.at_every(period_line, save_hline),
                               mp.at_every(period_plane, save_hplane)]
    
    step_ram_function = lambda sim : measure_ram()
    
    to_do_while_running = [*to_do_while_running,
                           mp.at_beginning(step_ram_function), 
                           mp.at_time(int(until_time / 2), 
                                      step_ram_function),
                           mp.at_end(step_ram_function)]
    
    #%% RUN!
    
    measure_ram()
    
    filename_prefix = sim.filename_prefix
    sim.filename_prefix = "Field"    
    os.chdir(path)
    temp = time()
    sim.run(*to_do_while_running, until=until_time)
    elapsed.append(time() - temp)
    os.chdir(syshome)
    sim.filename_prefix = filename_prefix
    
    #%% SAVE METADATA
    
    params = {}
    for p in params_list: params[p] = eval(p)
    
    volumes = [sampling_line, sampling_plane]
    periods = [period_line, period_plane]
    dimensions = []
    for vol, per in zip(volumes, periods):
        t = np.arange(0, until_time, per)
        x, y, z, more = sim.get_array_metadata(vol=vol)
        dimensions.append([t, x, y, z])
    del t, x, y, z, more
    del volumes, periods, vol, per
    
    if parallel_assign(0):
        
        files = ["Field-Lines", "Field-Planes"]
        for fil, dims in zip(files, dimensions):
            
            f = h5.File(file(fil + ".h5"), "r+")
            keys = [vu.camel(k) for k in f.keys()]
            for oldk, newk in zip(list(f.keys()), keys):
                f[newk] = f[oldk]
                del f[oldk]
        
            for k, array in zip(["T", "X","Y","Z"], dims): 
                f[k] = array
            
            for k in f.keys(): 
                for kpar in params.keys(): 
                    f[k].attrs[kpar] = params[kpar] 
            
            f.close()
            
        del f, keys, oldk, newk, k, kpar, array
    
        if hfield:
            
            hfiles = ["Field-HLines", "Field-HPlanes"]
            for hfil, fil in zip(hfiles, files):
            
                fh = h5.File(file(hfil + ".h5"), "r+")
                keys = [vu.camel(k) for k in fh.keys()]
                for oldk, newk in zip(list(fh.keys()), keys):
                    fh[newk] = fh[oldk]
                    del fh[oldk]
                    
                f = h5.File(file(fil + ".h5"), "r+")
                for k in fh.keys():
                    array = np.array(fh[k])
                    f[k] = array
                del array
                
                f.close()
                fh.close()
                os.remove(file(hfil + ".h5"))
            
            del hfiles, f, fh, keys, oldk, newk, k
          
        del files
        
    if parallel:
        f = vs.parallel_hdf_file(file("RAM.h5"), "w")
        current_process = mp.my_rank()
        f.create_dataset("RAM", (len(used_ram), n_processes), dtype="float")
        f["RAM"][:, current_process] = used_ram
        for a in params: f["RAM"].attrs[a] = params[a]
        f.create_dataset("SWAP", (len(used_ram), n_processes), dtype="int")
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
    
    mp.all_wait()
    
    #%% GET READY TO LOAD DATA
    
    if make_plots or make_gifs:
        
        plots_monoch_field(series, folder, hfield, 
                           make_plots, make_gifs, trs.english)
    
    sim.reset_meep()

#%%

if __name__ == '__main__':
    main()
