#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Field of visible monochromatic wave propagating in free space.
"""

script = "monoch_field"

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
import h5py as h5
import matplotlib.pyplot as plt
import meep as mp
import numpy as np
import os
import v_meep as vm
import v_meep_analysis as vma
import v_utilities as vu
from monoch_field_plot import plots_monoch_field

rm = vm.ResourcesMonitor()
rm.measure_ram()

#%% COMMAND LINE FORMATTER

@cli.command()
@cli.option("--units", "-un", "units", type=bool, default=False,
            help="Whether to express wavelengths in units or not")
@cli.option("--resolution", "-res", type=int, default=2,
            help="Spatial resolution. Number of divisions of each Meep unit")
@cli.option("--resolution-wlen", "-rwlen", type=int, default=10,
            help="Spatial resolution. Number of divisions of a wavelength")
@cli.option("--from-um-factor", "-fum", "from_um_factor", 
            default=10e-3, type=float,
            help="Conversion of 1 μm to my length unit (i.e. 10e-3=10nm/1μm)")
# >=8 pixels per smallest wavelength, i.e. np.floor(8/wvl_min)
@cli.option("--courant", "-c", "courant", 
            type=float, default=0.5,
            help="Courant factor: time discretization from space discretization")
@cli.option("--index", "-i", "submerged_index", 
            type=float, default=1,
            help="Reflective index of sourrounding medium")
@cli.option("--surface-index", "-si", "surface_index", 
            type=float, default=None,
            help="Reflective index of surface medium")
@cli.option("--wlen", "-wl", "wlen", default=532, type=float,
            help="Wavelength expressed in nm")
@cli.option("--empty-wlen-factor", "-empty", "empty_wlen_factor", 
            type=float, default=0.25,
            help="Empty layer width expressed in multiples of wavelength")
@cli.option("--pml-wlen-factor", "-pml", "pml_wlen_factor", 
            type=float, default=0.38,
            help="PML layer width expressed in multiples of wavelength")
@cli.option("--wlen-in-vacuum", "-wlenvac", "wlen_in_vacuum", 
            type=bool, default=True,
            help="Whether to consider wavelength in vacuum or in medium" + 
                 " for PML, resolution and empty width calculation")
@cli.option("--centered-source", "-cs", "centered_source", type=bool, default=False,
            help="Whether to place the source at the center of the cell or not")
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
def main(from_um_factor, resolution, resolution_wlen, courant,
         submerged_index, surface_index,
         empty_wlen_factor, pml_wlen_factor, wlen_in_vacuum, centered_source,
         wlen, time_period_factor,
         series, folder,
         n_cores, n_nodes, split_chunks_evenly,
         hfield, units, make_plots, make_gifs):
    
    #%% DEFAULT PARAMETERS
    
    if any('SPYDER' in name for name in os.environ):
    
        # Sim configuration
        units = True
        resolution_wlen = 10
        resolution = 5
        from_um_factor = 10e-3
        courant = 0.5
        
        # Cell configuration
        submerged_index = 1 # 1.33 for water
        surface_index = None # 1.54 for glass
        
        # Source configuration
        wlen = 405
        
        # Box spatial dimensions
        wlen_in_vacuum = True
        pml_wlen_factor = 0.38
        empty_wlen_factor = 0.25
        
        # Sim temporal dimension
        time_period_factor = 10    
        
        # Files configuration
        series = "TestFieldSerial"
        folder = "Test"
        
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
        
        print("Loaded Spyder parameters")
    
    #%% ADDITIONAL PARAMETERS
        
    # Field Measurements
    n_period_line = 100
    n_period_plane = 100
    
    # Routine configuration
    english = False
    
    #%% TREATED PARAMETERS
    
    # Computation
    pm = vm.ParallelManager(n_cores, n_nodes)
    n_processes, n_cores, n_nodes = pm.specs
    parallel = pm.parallel
    
    # Simulation space configuration
    if surface_index is None: surface_index = submerged_index
    
    # Frequency and wavelength
    if units:
        wlen = wlen / ( from_um_factor * 1e3 ) # Wavelength from nm to Meep units
        if wlen_in_vacuum:
            resolution_wlen = wlen * resolution
        else:
            resolution_wlen = wlen * resolution / submerged_index
        pm.log(f"Running with units: {resolution:.0f} points in a Meep Unit of " + 
               f"{from_um_factor*1e3:.0f} nm with {wlen * from_um_factor * 1e3} nm wavelength")
    else:
        if wlen_in_vacuum:
            wlen = 1
        else:
            wlen = submerged_index * 1 # Wavelength in medium wlen/index is 1 Meep unit
        resolution = resolution_wlen # Divide wlen in vacuum in resolution_wlen pieces
        from_um_factor = 1e3
        if wlen_in_vacuum:
            log_text = "vacuum"
        else:
            log_text = "medium"
        pm.log(f"Running without units: {resolution_wlen:.0f} points in a {log_text} wavelength")
    period = wlen
    
    # Space configuration    
    if units:
        if wlen_in_vacuum:
            pml_width = pml_wlen_factor * wlen # Multiples of vacuum wavelength
        else:
            pml_width = pml_wlen_factor * wlen / submerged_index # Multiples of medium wavelength
        empty_width = empty_wlen_factor * wlen
    else:
        pml_width = pml_wlen_factor # Multiples of reference wavelength, which is 1 Meep unit
        empty_width = empty_wlen_factor # Multiples of reference wavelength, which is 1 Meep unit
    cell_width = 2 * (pml_width + empty_width)
    
    # Time configuration
    until_time = time_period_factor * period # I want a certain number of periods in the source
    period_line = period / n_period_line # Now I want a certain number of instants in the period of the source
    period_plane = period / n_period_plane # If I use period instead of wlen, the discretization will be different
    
    # Saving directories
    sa = vm.SavingAssistant(series, folder)
    home = sa.home
    sysname = sa.sysname
    path = sa.path
    
    trs = vu.BilingualManager(english=english)
    
    params_list = ["from_um_factor", "resolution", "resolution_wlen", "courant",
                   "submerged_index", "wlen", "surface_index",
                   "cell_width", "pml_width", "empty_width", "source_center", "wlen_in_vacuum", 
                   "until_time", "time_period_factor", 
                   "n_period_line", "n_period_plane", "period_line", "period_plane",
                   "parallel", "n_processes", "n_cores", "n_nodes",
                   "split_chunks_evenly", "hfield", "units",
                   "script", "sysname", "path"]
    
    #%% GENERAL GEOMETRY SETUP
    
    ### ROUND UP ACCORDING TO GRID DISCRETIZATION
        
    pml_width = vu.round_to_multiple(pml_width, 1/resolution)
    cell_width = vu.round_to_multiple(cell_width/2, 1/resolution)*2
    empty_width = cell_width/2 - pml_width
    if centered_source: 
        source_center = 0
    else:
        source_center = -0.5*cell_width + pml_width
    
    until_time = vu.round_to_multiple(until_time, courant/resolution, round_up=True)
    period_line = vu.round_to_multiple(period_line, courant/resolution, round_down=True)
    period_plane = vu.round_to_multiple(period_plane, courant/resolution, round_down=True)
    
    ### DEFINE OBJETS
    
    pml_layers = [mp.PML(thickness=pml_width)]
    cell_size = mp.Vector3(cell_width, cell_width, cell_width)
    
    if surface_index != submerged_index:
        surface = mp.Block(material=mp.Medium(index=surface_index),
                           center=mp.Vector3(cell_width/4, 0, 0),
                           size=mp.Vector3(cell_width/2, cell_width, cell_width))
        geometry = [surface]
    else:
        geometry = []
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
    
    #%% PLOT CELL

    if pm.assign(0) and make_plots:
        
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
            
        # Surface medium
        if surface_index != submerged_index:
            surface_square = plt.Rectangle((0, -cell_width/2),
                                           cell_width/2,  cell_width,
                                           edgecolor="navy", hatch=r"\\", 
                                           fill=False, zorder=-3,
                                           label=trs.choose(fr"Surface $n$={surface_index}",
                                                            fr"Superficie $n$={surface_index}"))
            
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
        
        if submerged_index!=1: ax.add_patch(surrounding_square)
        if surface_index!=submerged_index: ax.add_patch(surface_square)
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
        plt.xlabel(trs.choose("Position X [MPu]", "Posición X [uMP]"))
        plt.ylabel(trs.choose("Position Z [MPu]", "Posición Z [uMP]"))
        
        if units:
            plt.annotate(trs.choose(f"1 Meep Unit = {from_um_factor * 1e3:.0f} nm",
                                    f"1 Unidad de Meep = {from_um_factor * 1e3:.0f} nm"),
                         (5, 5), xycoords='figure points')
            plt.annotate(fr"$\lambda$ = {wlen * from_um_factor * 1e3:.0f} nm",
                         (410, 120), xycoords='figure points')
        else:
            plt.annotate(trs.choose(r"1 Meep Unit = $\lambda$",
                                    r"1 Unidad de Meep = $\lambda$"),
                         (5, 5), xycoords='figure points')
        
        plt.savefig(sa.file("SimBox.png"))
        
        del pml_inn_square, pml_out_square
        if submerged_index!=1: del surrounding_square
        if surface_index!=submerged_index: del surface_square
        del fig, box, ax
    
    #%% INITIALIZE
    
    params = {}
    for p in params_list: params[p] = eval(p)
    
    # stable, max_courant = vm.check_stability(params)
    # if stable:
    #     pm.log("As a whole, the simulation should be stable")
    # else:
    #     pm.log("As a whole, the simulation could not be stable")
    #     pm.log(f"Recommended maximum courant factor is {max_courant}")
    # del stable, max_courant
    
    rm.measure_ram()
    
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
    
    rm.measure_ram()
    
    rm.start_measure_time()
    sim.init_sim()
    rm.end_measure_time()
    
    rm.measure_ram()
    
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
    
    step_ram_function = lambda sim : rm.measure_ram()
    
    to_do_while_running = [*to_do_while_running,
                           mp.at_beginning(step_ram_function), 
                           mp.at_time(int(until_time / 2), 
                                      step_ram_function),
                           mp.at_end(step_ram_function)]
    
    #%% RUN!
    
    rm.measure_ram()
    
    filename_prefix = sim.filename_prefix
    sim.filename_prefix = "Field"    
    os.chdir(path)
    rm.start_measure_time()
    sim.run(*to_do_while_running, until=until_time)
    rm.end_measure_time()
    os.chdir(syshome)
    sim.filename_prefix = filename_prefix
    
    #%% NORMALIZE AND SAVE METADATA
    
    for p in params_list: params[p] = eval(p)
    
    volumes = [sampling_line, sampling_plane]
    dimensions = []
    for vol in volumes:
        x, y, z, more = sim.get_array_metadata(vol=vol)
        dimensions.append([x, y, z])
    del x, y, z, more
    del volumes, vol
    
    f = pm.hdf_file(sa.file("Field-Lines.h5"), "r")
    
    results_line = f["ez"]
    t_line = np.arange(0, f["ez"].shape[-1] * period_line, period_line)
    x_line = dimensions[0][0]
    
    x_line_index = vma.def_index_function(x_line)
    
    source_results = vma.get_source_from_line(results_line, x_line_index, source_center)
    norm_period = vma.get_period_from_source(source_results, t_line)
    norm_amplitude = vma.get_amplitude_from_source(source_results)
    
    params["norm_period"] = norm_period
    params["norm_amplitude"] = norm_amplitude
    
    f.close()
    
    if pm.assign(0):
        
        files = ["Field-Lines", "Field-Planes"]
        periods = [period_line, period_plane]
        for fil, per, dims in zip(files, periods, dimensions):
            
            f = h5.File(sa.file(fil + ".h5"), "r+")
            keys = [vu.camel(k) for k in f.keys()]
            for oldk, newk in zip(list(f.keys()), keys):
                try:
                    f[newk] = np.asarray(f[oldk]) / norm_amplitude
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
    
    #%% GET READY TO LOAD DATA
    
    if make_plots or make_gifs:
        
        plots_monoch_field(series, folder, units, hfield, 
                           make_plots, make_gifs, trs.english)
    
    sim.reset_meep()

#%%

if __name__ == '__main__':
    main()
