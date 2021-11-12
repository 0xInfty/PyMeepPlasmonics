#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Field of Au sphere given a visible monochromatic incident wave.
"""

script = "np_monoch_field"

from socket import gethostname
if "Nano" in gethostname():
    syshome = "/home/nanofisica/Documents/Vale/PyMeepPlasmonics"
elif "vall" in gethostname():
    syshome = "/home/vall/Documents/Thesis/PyMeepPlasmonics"
else:
    raise ValueError("Your PC must be registered at the top of this code")

import sys
sys.path.append(syshome)
sys.path.append(syshome+"/PlotRoutines")

import click as cli
import h5py as h5
import meep as mp
import numpy as np
import os
from vmp_materials import import_medium
import vmp_utilities as vmu
import v_utilities as vu

from np_planewave_cell_plot import plot_np_planewave_cell
from np_monoch_field_plot import plots_np_monoch_field

rm = vmu.ResourcesMonitor()
rm.measure_ram()

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
            type=float, default=20,
            help="Simulation total time expressed as multiples of periods")
@cli.option("--norm-period-factor", "-ntpfc", "norm_period_factor",
            type=float, default=8)
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
@cli.option("--load-field", "-loadf", "load_field", 
            type=bool, default=True,
            help="Whether to search the normfield database and load, if possible, or not.")
# @cli.option("--load-chunks", "-loadc", "load_chunks", 
#             type=bool, default=True,
#             help="Whether to search the chunks layout database and load, if possible, or not.")
@cli.option("--load-resources", "-loadr", "load_resources", 
            type=bool, default=False,
            help="Whether to necessarily load monitored resources, if possible, or not")
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
         time_period_factor, norm_period_factor,
         series, folder,
         n_cores, n_nodes, split_chunks_evenly,
         hfield, load_field, load_resources, make_plots, make_gifs):
    
    #%% DEFAULT PARAMETERS
    
    if any('SPYDER' in name for name in os.environ):
    
        rm.reset()    
    
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
        surface_index = None # 1.54 for glass
        
        # Source configuration
        wlen = 532
        
        # Box spatial dimensions
        pml_wlen_factor = 0.38
        empty_r_factor = 0.5
        
        # Sim temporal dimension
        time_period_factor = 20
        norm_period_factor = 15
        
        # Files configuration
        series = "TestFieldNorm"
        folder = "Test"
        
        # Run configuration
        parallel = False
        n_processes = 1
        n_cores = 1
        n_nodes = 1
        split_chunks_evenly = True
        
        # Routine configuration
        hfield = False
        load_field = True
        load_resources = True
        make_plots = True
        make_gifs = True
        
        print("Loaded Spyder parameters")
    
    #%% ADDITIONAL PARAMETERS
    
    # Simulation size
    resolution_wlen = 8
    resolution_nanoparticle = 5
    
    # Field Measurements
    n_period_line = 100
    n_period_plane = 100
    
    # Routine configuration
    english = False
    
    #%% TREATED PARAMETERS
    
    # Computation
    pm = vmu.ParallelManager(n_cores, n_nodes)
    n_processes, n_cores, n_nodes = pm.specs
    parallel = pm.parallel
    
    # Simulation size
    resolution_wlen = (from_um_factor * 1e3) * resolution_wlen / wlen
    resolution_wlen = int( np.ceil(resolution_wlen) )
    resolution_nanoparticle = (from_um_factor * 1e3) * resolution_nanoparticle / (2 * r)
    resolution_nanoparticle = int( np.ceil(resolution_nanoparticle) )
    min_resolution = max(resolution_wlen, 
                         resolution_nanoparticle)
    
    if min_resolution > resolution:
        pm.log(f"Resolution will be raised from {resolution} to {min_resolution}")
        resolution = min_resolution
    del min_resolution
    
    # Au sphere
    r = r  / ( from_um_factor * 1e3 )  # Radius of sphere now in Meep units
    medium = import_medium("Au", paper=paper, from_um_factor=from_um_factor) # Medium of sphere: gold (Au)
    if surface_index is None:
        surface_index = submerged_index
    
    # Frequency and wavelength
    wlen = wlen / ( from_um_factor * 1e3 ) # Wavelength now in Meep units
    period = submerged_index * wlen
    
    # Space configuration
    pml_width = pml_wlen_factor * wlen # 0.5 * wlen
    empty_width = empty_r_factor * r # 0.5 * max(wlen_range)
    cell_width = 2 * (pml_width + empty_width + r)
    
    # Time configuration
    until_time = time_period_factor * wlen # Should multiply and divide by submerged_index
    norm_until_time = norm_period_factor * wlen
    period_line = wlen / n_period_line # If I use period instead of wlen, the discretization will be different
    period_plane = wlen / n_period_plane
    
    # Saving directories
    sa = vmu.SavingAssistant(series, folder)
    home = sa.home
    sysname = sa.sysname
    path = sa.path
    
    trs = vu.BilingualManager(english=english)
    
    params_list = ["from_um_factor", "resolution", "courant",
                   "material", "r", "paper", 
                   "submerged_index", "wlen", "surface_index", "overlap",
                   "cell_width", "pml_width", "empty_width", "source_center",
                   "until_time", "norm_until_time", 
                   "time_period_factor", "norm_period_factor",
                   "n_period_line", "n_period_plane", "period_line", "period_plane",
                   "parallel", "n_processes", "n_cores", "n_nodes",
                   "split_chunks_evenly", "hfield",
                   "script", "sysname", "path"]
    
    #%% GENERAL GEOMETRY SETUP
    
    ### ROUND UP ACCORDING TO GRID DISCRETIZATION
        
    pml_width = vu.round_to_multiple(pml_width, 1/resolution)
    cell_width = vu.round_to_multiple(cell_width/2, 1/resolution)*2
    empty_width = cell_width/2 - r - pml_width
    overlap = vu.round_to_multiple(overlap - r, 1/resolution) + r
    source_center = -0.5*cell_width + pml_width
    
    until_time = vu.round_to_multiple(until_time, courant/resolution, round_up=True)
    norm_until_time = vu.round_to_multiple(norm_until_time, courant/resolution, round_up=True)
    period_line = vu.round_to_multiple(period_line, courant/resolution, round_down=True)
    period_plane = vu.round_to_multiple(period_plane, courant/resolution, round_down=True)
    
    ### DEFINE OBJETS
    
    pml_layers = [mp.PML(thickness=pml_width)]
    cell_size = mp.Vector3(cell_width, cell_width, cell_width)
    
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
        initial_geometry = [surface]
        final_geometry = [surface, nanoparticle]
    else:
        initial_geometry = []
        final_geometry = [nanoparticle]
    # If required, a certain material surface underneath it
    
    sources = [mp.Source(mp.ContinuousSource(wavelength=wlen, 
                                             is_integrated=True),
                         center=mp.Vector3(source_center),
                         size=mp.Vector3(0, cell_width, cell_width),
                         component=mp.Ez)]
    # Ez-polarized monochromatic planewave 
    # (its size parameter fills the entire cell in 2d)
    # The planewave source extends into the PML 
    # ==> is_integrated=True must be specified
    
    #%% FIRST RUN
    
    rm.measure_ram()
    
    params = {}
    for p in params_list: params[p] = eval(p)
    
    # stable, max_courant = vmu.check_stability(params)
    # if stable:
    #     pm.log("As a whole, the simulation should be stable")
    # else:
    #     pm.log("As a whole, the simulation might be unstable")
    #     pm.log(f"Recommended maximum Courant factor is {max_courant}")
    # del stable, max_courant
        
    if load_field:
        try:
            norm_path = vmu.check_normfield(params)[-1]
            if load_resources:
                if os.path.isfile( os.path.join(norm_path, "Resources.h5") ):
                    norm_needed = False
                    pm.log("Found resources")
                else:
                    norm_needed = True
                    pm.log("Didn't find resources")
            else:
                norm_needed  = False
        except:
            norm_needed = True
    else:
        norm_needed = True
        
    # if load_chunks and not split_chunks_evenly:
    #     try:
    #         chunks_path = vmu.check_chunks(params)[-1]
    #         chunk_layout = os.path.join(chunks_path, "Layout.h5")
    #         norm_needed = False
    #     except:
    #         chunks_needed = True
    #         norm_needed = True
    # else:
    #     if not split_chunks_evenly:
    #         chunks_needed = True
    #         norm_needed = True
    #     else:
    #         chunk_layout = None
    #         chunks_needed = False
    
    # if chunks_needed:

    #     sim = mp.Simulation(resolution=resolution,
    #                         Courant=courant,
    #                         geometry=final_geometry,
    #                         sources=sources,
    #                         k_point=mp.Vector3(),
    #                         default_material=mp.Medium(index=submerged_index),
    #                         cell_size=cell_size,
    #                         boundary_layers=pml_layers,
    #                         output_single_precision=True,
    #                         split_chunks_evenly=split_chunks_evenly)
    #                         # symmetries=symmetries,
    
    #     sim.init_sim()
        
    #     chunks_path = vmu.save_chunks(sim, params, path)         
    #     chunk_layout = os.path.join(chunks_path, "Layout.h5")
        
    #     del sim
        
    if not norm_needed:
        rm.load( os.path.join(norm_path, "Resources.h5") )
    else:
    
        #% % FIRST RUN: PLOT CELL
        
        params = {}
        for p in params_list: params[p] = eval(p)
    
        if pm.assign(0):
            plot_np_planewave_cell(params, series, folder, 
                                   with_line=True, with_plane=False, 
                                   with_nanoparticle=False, english=trs.english)
        
        #% % FIRST RUN: INITIALIZE
        
        # stable, max_courant = vmu.check_stability(params)
        # if stable:
        #     pm.log("As a whole, the simulation should be stable")
        # else:
        #     pm.log("As a whole, the simulation could not be stable")
        #     pm.log(f"Recommended maximum courant factor is {max_courant}")
        # del stable, max_courant
        
        rm.measure_ram()
        
        sim = mp.Simulation(resolution=resolution,
                            Courant=courant,
                            geometry=initial_geometry,
                            sources=sources,
                            k_point=mp.Vector3(),
                            default_material=mp.Medium(index=submerged_index),
                            cell_size=cell_size,
                            boundary_layers=pml_layers,
                            output_single_precision=True,
                            split_chunks_evenly=split_chunks_evenly)
        
        rm.measure_ram()
        
        rm.start_measure_time()
        sim.init_sim()
        rm.end_measure_time()
        
        rm.measure_ram()
        
        #% % FIRST RUN: DEFINE SAVE STEP FUNCTIONS
        
        sampling_line = mp.Volume(center=mp.Vector3(),
                                  size=mp.Vector3(cell_width),
                                  dims=1)
        
        save_line = mp.in_volume(sampling_line, 
                                 mp.to_appended("Lines-Norm",
                                                mp.output_efield_z))
        
        to_do_while_running = [mp.at_every(period_line, save_line)]
        
        if hfield:
            save_hline = mp.in_volume(sampling_line, 
                                      mp.to_appended("HLines-Norm",
                                                     mp.output_hfield_y))
            to_do_while_running = [*to_do_while_running, 
                                   mp.at_every(period_line, save_hline)]
        
        step_ram_function = lambda sim : rm.measure_ram()
        
        to_do_while_running = [*to_do_while_running,
                               mp.at_beginning(step_ram_function), 
                               mp.at_time(int(until_time / 2), 
                                          step_ram_function),
                               mp.at_end(step_ram_function)]
        
        #% % FIRST RUN: RUN!
        
        rm.measure_ram()
        
        filename_prefix = sim.filename_prefix
        sim.filename_prefix = "Field"    
        os.chdir(path)
        rm.start_measure_time()
        sim.run(*to_do_while_running, until=norm_until_time)
        rm.end_measure_time()
        os.chdir(syshome)
        sim.filename_prefix = filename_prefix
        
        #% % FIRST RUN: SAVE METADATA
        
        for p in params_list: params[p] = eval(p)
        
        x, y, z, more = sim.get_array_metadata(vol=sampling_line)
        del more
        
        if pm.assign(0):
            
            f = h5.File(sa.file("Field-Lines-Norm.h5"), "r+")
            keys = [vu.camel(k) for k in f.keys()]
            for oldk, newk in zip(list(f.keys()), keys):
                try:
                    f[newk] = f[oldk]
                    del f[oldk]
                except:
                    pass
            
            f["T"] = np.arange(0, f["Ez"].shape[-1]) * period_line
            for k, array in zip(["X","Y","Z"], [x, y, z]):  
                f[k] = array
            
            for k in f.keys(): 
                for kpar in params.keys(): 
                    f[k].attrs[kpar] = params[kpar]
            
            f.close()
            del f, keys, oldk, newk, k, kpar, array
        
            if hfield:
                
                fh = h5.File(sa.file("Field-HLines-Norm.h5"), "r+")
                keys = [vu.camel(k) for k in fh.keys()]
                for oldk, newk in zip(list(fh.keys()), keys):
                    fh[newk] = fh[oldk]
                    del fh[oldk]
                    
                f = h5.File(sa.file("Field-Lines-Norm.h5"), "r+")
                for k in fh.keys():
                    array = np.array(fh[k])
                    f[k] = array
                del array
                
                f.close()
                fh.close()
                os.remove(sa.file("Field-HLines-Norm.h5"))
                
                del f, fh, keys, oldk, newk, k
        
        norm_path = vmu.save_normfield(params, path)
        
        rm.save(os.path.join(norm_path, "Resources.h5"), params)
        rm.save(sa.file("Resources.h5"), params)
        
        mp.all_wait()
        
        sim.reset_meep()
    
    #%% SECOND RUN: PLOT CELL
    
    params = {}
    for p in params_list: params[p] = eval(p)
    params["norm_path"] = norm_path
    
    if pm.assign(0):
        plot_np_planewave_cell(params, series, folder, 
                               with_line=True, with_plane=True, 
                               with_nanoparticle=True, english=trs.english)
    
    #%% SECOND RUN: INITIALIZE
    
    # stable, max_courant = vmu.check_stability(params)
    # if stable:
    #     pm.log("As a whole, the simulation should be stable")
    # else:
    #     pm.log("As a whole, the simulation could not be stable")
    #     pm.log(f"Recommended maximum courant factor is {max_courant}")
    # del stable, max_courant
    
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
                        split_chunks_evenly=split_chunks_evenly)
    
    rm.measure_ram()
    
    rm.start_measure_time()
    sim.init_sim()
    rm.end_measure_time()
    
    rm.measure_ram()
    
    #%% LOAD FIELD FROM FILE
    
    norm_amplitude, norm_period = vmu.load_normfield(norm_path)

    rm.measure_ram()
    
    #%% SECOND RUN: DEFINE SAVE STEP FUNCTIONS
    
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
    
    step_ram_function = lambda sim : rm.measure_ram()
    
    to_do_while_running = [*to_do_while_running,
                           mp.at_beginning(step_ram_function), 
                           mp.at_time(int(until_time / 2), 
                                      step_ram_function),
                           mp.at_end(step_ram_function)]
    
    #%% SECOND RUN: RUN!
    
    rm.measure_ram()
    
    filename_prefix = sim.filename_prefix
    sim.filename_prefix = "Field"    
    os.chdir(path)
    rm.start_measure_time()
    sim.run(*to_do_while_running, until=until_time)
    rm.end_measure_time()
    os.chdir(syshome)
    sim.filename_prefix = filename_prefix
    
    #%% SECOND RUN: SAVE METADATA
    
    for p in params_list: params[p] = eval(p)
    params["norm_amplitude"] = norm_amplitude
    params["norm_period"] = norm_period
    params["norm_path"] = norm_path
    
    volumes = [sampling_line, sampling_plane]
    dimensions = []
    for vol in volumes:
        x, y, z, more = sim.get_array_metadata(vol=vol)
        dimensions.append([x, y, z])
    del x, y, z, more
    del volumes, vol
    
    if pm.assign(0):
        
        files = ["Field-Lines", "Field-Planes"]
        periods = [period_line, period_plane]
        for fil, per, dims in zip(files, periods, dimensions):
            
            f = h5.File(sa.file(fil + ".h5"), "r+")
            keys = [vu.camel(k) for k in f.keys()]
            for oldk, newk in zip(list(f.keys()), keys):
                try:
                    f[newk] = np.asarray(f[oldk] / norm_amplitude)
                    del f[oldk]
                except:
                    pass
            
            f["T"] = np.arange(0, f["Ez"].shape[-1] * per, per)
            for k, array in zip(["X","Y","Z"], dims): 
                f[k] = array
            
            for k in f.keys(): 
                for kpar in params.keys(): 
                    f[k].attrs[kpar] = params[kpar]
            
            f.close()
            
        del f, keys, oldk, newk, k, kpar, array
        del periods, per
    
        if hfield:
            
            hfiles = ["Field-HLines", "Field-HPlanes"]
            for hfil, fil in zip(hfiles, files):
            
                fh = h5.File(sa.file(hfil + ".h5"), "r+")
                keys = [vu.camel(k) for k in fh.keys()]
                for oldk, newk in zip(list(fh.keys()), keys):
                    fh[newk] = fh[oldk]
                    del fh[oldk]
                    
                f = h5.File(sa.file(fil + ".h5"), "r+")
                for k in fh.keys():
                    array = np.array(fh[k])
                    f[k] = array
                del array
                
                f.close()
                fh.close()
                os.remove(sa.file(hfil + ".h5"))
            
            del hfiles, f, fh, keys, oldk, newk, k
          
        del files
            
    rm.save(sa.file("Resources.h5"), params)
    
    mp.all_wait()
    
    #%% SECOND RUN: MAKE PLOTS IF REQUIRED
    
    if make_plots or make_gifs:
        
        plots_np_monoch_field(series, folder, hfield, 
                              make_plots, make_gifs, trs.english)
    
    sim.reset_meep()

#%%

if __name__ == '__main__':
    main()
