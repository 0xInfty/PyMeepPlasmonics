#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 15:29:32 2020

@author: vall
"""

# Field of Au sphere given submerged in a medium with a visible monochromatic incident wave.

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
import meep as mp
import os
from time import time
# from v_meep import import_medium
import v_save as vs

#%% COMMAND LINE FORMATTER

@cli.command()
@cli.option("--series", "-s", type=str, 
            help="Series name used to create a folder and save files")
@cli.option("--folder", "-f", type=str, 
            help="Series folder used to save files")
@cli.option("--resolution", "-res", required=True, type=int,
            help="Spatial resolution. Number of divisions of each Meep unit")
@cli.option("--from-um-factor", "-fum", "from_um_factor", 
            default=10e-3, type=float,
            help="Conversion of 1 μm to my length unit (i.e. 10e-3=10nm/1μm)")
# >=8 pixels per smallest wavelength, i.e. np.floor(8/wvl_min)
@cli.option("--radius", "-r", "r", default=60, type=int,
            help="Radius of sphere expressed in nm")
# @cli.option("--paper", "-pp", "paper", type=str, default="R",
#             help="Source of inner material experimental data. Options: 'JC'/'R'/'P'")
@cli.option("--index", "-i", "submerged_index", 
            type=float, default=1.33,
            help="Reflective index of sourrounding medium")
@cli.option("--wlen", "-wl", "wlen", default=532, type=float,
            help="Wavelength expressed in nm")
def main(series, folder, resolution, from_um_factor, #paper,
         r, submerged_index, wlen):
    #%% PARAMETERS
    
    # Au sphere
    r = r  / ( from_um_factor * 1e3 )  # Radius of sphere now in Meep units
    # medium = import_medium("Au", from_um_factor, paper=paper) # Medium of sphere: gold (Au)
    
    # Frequency and wavelength
    wlen = wlen / ( from_um_factor * 1e3 ) # Wavelength now in Meep units
    
    # Space configuration
    pml_width = 0.5 * wlen # 0.5 * wlen
    air_width = r/2 # 2 * r
    
    # Field Measurements
    period_line = 1
    period_plane = 1
    after_cell_run_time = 10*wlen
    
    # Computation time
    enlapsed = []
    
    # Saving directories
    if series is None:
        series = f"AuSphereField{2*r*from_um_factor*1e3:.0f}WLen{wlen*from_um_factor*1e3:.0f}"
    if folder is None:
        folder = "AuMieSphere/AuSphereField"
    home = vs.get_home()
    
    #%% GENERAL GEOMETRY SETUP
    
    air_width = air_width - air_width%(1/resolution)
    
    pml_width = pml_width - pml_width%(1/resolution)
    pml_layers = [mp.PML(thickness=pml_width)]
    
    symmetries = [mp.Mirror(mp.Y), 
                  mp.Mirror(mp.Z, phase=-1)]
    # Cause of symmetry, two mirror planes reduce cell size to 1/4
    
    cell_width = 2 * (pml_width + air_width + r)
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
    # >> The planewave source extends into the PML 
    # ==> is_integrated=True must be specified
    
    # geometry = [mp.Sphere(material=medium,
    #                       center=mp.Vector3(),
    #                       radius=r)]
    # Au sphere with frequency-dependant characteristics imported from Meep.
    
    path = os.path.join(home, folder, series)
    if not os.path.isdir(path): vs.new_dir(path)
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
    
    #%% INITIALIZE
    
    sim = mp.Simulation(resolution=resolution,
                        cell_size=cell_size,
                        boundary_layers=pml_layers,
                        sources=sources,
                        symmetries=symmetries,
                        # geometry=geometry,
                        default_material=mp.Medium(index=submerged_index))
    
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
    sim.run(*to_do_while_running, until=cell_width+after_cell_run_time)
    del f, g
    enlapsed.append(time() - temp)
    
    #%% SAVE METADATA
    
    params = dict(
        from_um_factor=from_um_factor,
        resolution=resolution,
        r=r,
        # paper=paper,
        submerged_index=submerged_index,
        pml_width=pml_width,
        air_width=air_width,
        cell_width=cell_width,
        source_center=source_center,
        wlen=wlen,
        period_line=period_line,
        period_plane=period_plane,
        after_cell_run_time=after_cell_run_time,
        series=series,
        folder=folder, 
        home=home,
        enlapsed=enlapsed
        )
    
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
