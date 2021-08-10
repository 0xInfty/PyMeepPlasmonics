#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Field of Au sphere given a visible monochromatic incident wave.
"""

script = "sphere_monoch_field"

from socket import gethostname
if "Nano" in gethostname():
    syshome = "/home/nanofisica/Documents/Vale/ThesisPython"
elif "vall" in gethostname():
    syshome = "/home/vall/Documents/Thesis/ThesisPython"
else:
    raise ValueError("Your PC must be registered at the top of this code")

import sys
sys.path.append(syshome)

import click as cli
import h5py as h5
import matplotlib.pyplot as plt
import meep as mp
import os
from time import time
from v_materials import import_medium
import v_meep as vm
import v_save as vs

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
@cli.option("--wlen", "-wl", "wlen", default=532, type=float,
            help="Wavelength expressed in nm")
@cli.option("--empty-r-factor", "-empty", "empty_r_factor", 
            type=float, default=0.5,
            help="Empty layer width expressed in multiples of radius")
@cli.option("--pml-wlen-factor", "-pml", "pml_wlen_factor", 
            type=float, default=0.38,
            help="PML layer width expressed in multiples of maximum wavelength")
@cli.option("--time-factor-cell", "-tfc", "time_factor_cell", 
            type=float, default=10,
            help="Simulation total time expressed as multiples of time \
                required to go through the cell")
@cli.option("--series", "-s", type=str, 
            help="Series name used to create a folder and save files")
@cli.option("--folder", "-f", type=str, 
            help="Series folder used to save files")
@cli.option("--parallel", "-par", type=bool, default=False,
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
def main(from_um_factor, resolution, courant,
         r, material, paper, wlen, submerged_index, 
         empty_r_factor, pml_wlen_factor, 
         time_factor_cell,
         series, folder,
         parallel, n_processes, n_cores, n_nodes, split_chunks_evenly):
    
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
    submerged_index = 1 # 1.33 for water
    
    # Source configuration
    wlen = 532
    
    # Box dimensions
    pml_wlen_factor = 0.38
    empty_r_factor = 0.5
    
    # Files configuration
    series = None
    folder = None
    
    # Run configuration
    parallel = False
    n_processes = 1
    n_cores = 1
    n_nodes = 1
    split_chunks_evenly = True
    """
    
    #%% PARAMETERS
    
    # Au sphere
    r = r  / ( from_um_factor * 1e3 )  # Radius of sphere now in Meep units
    medium = import_medium("Au", from_um_factor, paper=paper) # Medium of sphere: gold (Au)
    
    # Frequency and wavelength
    wlen = wlen / ( from_um_factor * 1e3 ) # Wavelength now in Meep units
    
    # Space configuration
    pml_width = pml_wlen_factor * wlen # 0.5 * wlen
    empty_width = empty_r_factor * r # 0.5 * max(wlen_range)
    
    # Field Measurements
    period_line = 1
    period_plane = 1
    after_cell_run_time = 10 * wlen
    
    # Computation time
    enlapsed = []
    
    # Saving directories
    if series is None:
        series = f"AuSphereField{2*r*from_um_factor*1e3:.0f}WLen{wlen*from_um_factor*1e3:.0f}"
    if folder is None:
        folder = "AuMieSphere/AuSphereField"
    
    params_list = ["from_um_factor", "resolution", "courant",
                   "material", "r", "paper", "submerged_index", "wlen", 
                   "cell_width", "pml_width", "empty_width", "source_center",
                   "until_time", "time_factor_cell", 
                   "enlapsed", "parallel", "n_processes", "n_cores", "n_nodes",
                   "split_chunks_evenly", 
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
    print("Resto Source Center: {}".format(source_center%(1/resolution)))
    sources = [mp.Source(mp.ContinuousSource(wavelength=wlen, 
                                             is_integrated=True),
                         center=mp.Vector3(source_center),
                         size=mp.Vector3(0, cell_width, cell_width),
                         component=mp.Ez)]
    # Ez-polarized monochromatic planewave 
    # (its size parameter fills the entire cell in 2d)
    # The planewave source extends into the PML 
    # ==> is_integrated=True must be specified
    
    until_time = time_factor_cell * cell_width * submerged_index
    
    geometry = [mp.Sphere(material=medium,
                          center=mp.Vector3(),
                          radius=r)]
    # Au sphere with frequency-dependant characteristics imported from Meep.
    
    home = vs.get_home()
    sysname = vs.get_sys_name()
    path = os.path.join(home, folder, series)
    if not os.path.isdir(path) and vm.parallel_assign(0, n_processes, parallel):
        os.makedirs(path)
    file = lambda f : os.path.join(path, f)
    
    #%% SAVE GET FUNCTIONS
    
    def get_line(sim):
        return sim.get_array(
            center=mp.Vector3(), 
            size=mp.Vector3(cell_width), 
            component=mp.Ez)
    
    def get_plane(sim):
        return sim.get_array(
            center=mp.Vector3(), 
            size=mp.Vector3(0, cell_width, cell_width), 
            component=mp.Ez)
    
    #%% PLOT CELL

    if vm.parallel_assign(0, n_processes, parallel):
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
        
        # Sensing line
        ax.hlines(0, -cell_width/2, cell_width/2,
                  color="limegreen", linestyle=":", zorder=7, label="Sensing Line")
        
        
        
        # Sensing plane
        ax.vlines(0, -cell_width/2, cell_width/2,
                  color="limegreen", linestyle="dashed", zorder=7, 
                  label="Sensing Plane")
        
        ax.add_patch(circle)
        if submerged_index!=1: ax.add_patch(surrounding_square)
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
        plt.ylabel("Position Z [Meep Units]")
        
        plt.annotate(f"1 Meep Unit = {from_um_factor * 1e3:.0f} nm",
                (5, 5),
                xycoords='figure points')
        
        plt.savefig(file("SimBox.png"))
        
        del pml_inn_square, pml_out_square, flux_square, circle, circle_color
        if submerged_index!=1: del surrounding_square
        del fig, box, ax
    
    #%% INITIALIZE
    
    params = {}
    for p in params_list: params[p] = eval(p)
    
    stable, max_courant = vm.check_stability(params)
    if stable:
        print("As a whole, the simulation should be stable")
    else:
        print("As a whole, the simulation could not be stable")
        print(f"Recommended maximum courant factor is {max_courant}")
    del stable, max_courant
    
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
    
    temp = time()
    sim.init_sim()
    enlapsed.append(time()-temp)
    
    #%% DEFINE SAVE STEP FUNCTIONS
    
    f, save_line = vs.save_slice_generator(sim, file("Lines.h5"), "Ez", get_line)
    g, save_plane = vs.save_slice_generator(sim, file("Planes.h5"), "Ez", get_plane)
    
    to_do_while_running = [mp.at_every(period_line, save_line),
                           mp.at_every(period_plane, save_plane)]
    
    #%% RUN!
    
    temp = time()
    sim.run(*to_do_while_running, until=until_time)
    del f, g
    enlapsed.append(time() - temp)
    
    #%% SAVE METADATA
    
    params = {}
    for p in params_list: params[p] = eval(p)
    
    f = h5.File(file("Lines.h5"), "r+")
    for a in params: f["Ez"].attrs[a] = params[a]
    f.close()
    del f
    
    g = h5.File(file("Planes.h5"), "r+")
    for a in params: g["Ez"].attrs[a] = params[a]
    g.close()
    del g
    
    sim.reset_meep()

#%%

if __name__ == '__main__':
    main()
