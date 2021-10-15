#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Field of a pulse planewave of Gaussian frequency spectrum propagating in space.
"""

script = "pulse_field"

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
import v_save as vs
import v_utilities as vu

from np_planewave_cell_plot import plot_np_planewave_cell
from pulse_field_plot import plots_pulse_field

rm = vm.ResourcesMonitor()
rm.measure_ram()

#%% COMMAND LINE FORMATTER

@cli.command()
@cli.option("--units", "-un", "units", type=bool, default=False,
            help="Whether to express wavelengths in units or not")
@cli.option("--resolution", "-res", type=int, default=2,
            help="Spatial resolution. Number of divisions of each Meep unit")
@cli.option("--resolution-wlen", "-rwlen", type=int, default=10,
            help="Spatial resolution. Number of divisions of minimum wavelength")
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
@cli.option("--wlen-range", "-wr", "wlen_range", 
            type=cli.Tuple([float, float]), default=(450,600),
            help="Wavelength range expressed in nm")
@cli.option("--nfreq", "-nfreq", "nfreq", 
            type=int, default=100,
            help="Quantity of frequencies to sample in wavelength range")
@cli.option("--empty-wlen-factor", "-empty", "empty_wlen_factor", 
            type=float, default=0.25,
            help="Empty layer width expressed in multiples of minimum wavelength")
@cli.option("--pml-wlen-factor", "-pml", "pml_wlen_factor", 
            type=float, default=0.38,
            help="PML layer width expressed in multiples of minimum wavelength")
@cli.option("--time-factor-cell", "-tfc", "time_factor_cell", 
            type=float, default=1.2,
            help="First simulation total time expressed as multiples of time \
                required to go through the cell")
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
         empty_wlen_factor, pml_wlen_factor,
         wlen_range, nfreq, time_factor_cell,
         series, folder,
         n_cores, n_nodes, split_chunks_evenly,
         hfield, units, make_plots, make_gifs):
    
    #%% DEFAULT PARAMETERS
    
    if any('SPYDER' in name for name in os.environ):
    
        rm.reset()    
    
        # Sim configuration
        units = False
        resolution_wlen = 10
        resolution = 2
        from_um_factor = 10e-3
        courant = 0.5
        
        # Cell configuration
        submerged_index = 1.33 # 1.33 for water
        surface_index = None # 1.54 for glass
        
        # Source configuration
        if units: wlen_range = (450, 600)
        else: wlen_range = (.9, 1.1)#(.9,1.1) #(.95,1.05)
        nfreq = 100
        
        # Box spatial dimensions
        wlen_in_vacuum = True
        pml_wlen_factor = 0.18
        empty_wlen_factor = 5#0.77 #0.67
        
        # Sim temporal dimension
        time_factor_cell = 1.35
        
        # Files configuration
        series = "JustATest"
        folder = "Test"
        
        # Run configuration
        parallel = False
        n_processes = 1
        n_cores = 1
        n_nodes = 1
        split_chunks_evenly = True
        
        # Routine configuration
        hfield = False
        make_plots = True
        make_gifs = False
        
        print("Loaded Spyder parameters")
    
    #%% ADDITIONAL PARAMETERS
            
    # Source
    cutoff = 3.5
    
    # Field Measurements
    n_period_line = 100
    n_flux_walls = 30
    
    # Routine configuration
    english = False
    wlen_in_vacuum = True
    
    #%% TREATED PARAMETERS
    
    # Computation
    pm = vm.ParallelManager(n_cores, n_nodes)
    n_processes, n_cores, n_nodes = pm.specs
    parallel = pm.parallel
    
    # Simulation space configuration
    if surface_index is None: surface_index = submerged_index
    
    # Frequency and wavelength
    wlen_range = np.array(wlen_range)
    if units:
        wlen_range = wlen_range / ( from_um_factor * 1e3 ) # Wavelength from nm to Meep units
        if wlen_in_vacuum:
            resolution_wlen = np.mean(wlen_range) * resolution
        else:
            resolution_wlen = np.mean(wlen_range) * resolution / submerged_index
        pm.log(f"Running with units: {resolution:.0f} points in a Meep Unit of " + 
               f"{from_um_factor*1e3:.0f} nm with "+
               f"{wlen_range[0] * from_um_factor * 1e3}-{wlen_range[1] * from_um_factor * 1e3} nm wavelength")
    else:
        if wlen_in_vacuum:
            wlen_range = np.array([1 - float(np.diff(wlen_range))/2, 
                                   1 + float(np.diff(wlen_range))/2])
            # Central wavelength in vacuum wlen is 1 Meep unit
        if not wlen_in_vacuum:
            wlen_range = np.array([1 - float(np.diff(wlen_range))/2, 
                                   1 + float(np.diff(wlen_range))/2]) * submerged_index
            # Central wavelength in medium wlen/index is 1 Meep unit
        # resolution = int( resolution_wlen * np.mean(wlen_range)  / min(wlen_range) )
        resolution = resolution_wlen
        # Divide mean wavelength in vacuum in resolution_wlen pieces
        from_um_factor = 1e-3
        if wlen_in_vacuum:
            log_text = "vacuum"
        else:
            log_text = "medium"
        pm.log(f"Running without units: {resolution_wlen:.0f} points in a {log_text} wavelength")
        
    freq_range = 1/wlen_range # Hz range in Meep units from highest to lowest
    freq_center = np.mean(freq_range)
    freq_width = max(freq_range) - min(freq_range)
    period_range = wlen_range
    
    # Space configuration    
    if units:
        if wlen_in_vacuum:
            pml_width = pml_wlen_factor * np.mean(wlen_range) # Multiples of vacuum wavelength
        else:
            pml_width = pml_wlen_factor * np.mean(wlen_range) / submerged_index # Multiples of medium wavelength
        empty_width = empty_wlen_factor * np.mean(wlen_range)
    else:
        pml_width = pml_wlen_factor # Multiples of reference wavelength, which is 1 Meep unit
        if wlen_in_vacuum:
            empty_width = empty_wlen_factor / submerged_index # Multiples of reference wavelength, which is 1 Meep unit
        else:
            empty_width = empty_wlen_factor # Multiples of reference wavelength, which is 1 Meep unit
    cell_width = 2 * (pml_width + empty_width)
    
    # Time configuration
    until_after_sources = time_factor_cell * cell_width
    # Enough time for the pulse to pass through all the cell
    # Originally: Aprox 3 periods of lowest frequency, using T=λ/c=λ in Meep units 
    period_line = np.mean(period_range) / n_period_line # Now I want a certain number of instants in the period of the source
    
    # Saving directories
    sa = vm.SavingAssistant(series, folder)
    home = sa.home
    sysname = sa.sysname
    path = sa.path
    
    trs = vu.BilingualManager(english=english)
    
    params_list = ["from_um_factor", "resolution", "resolution_wlen", "courant",
                   "wlen_range", "cutoff", "nfreq",
                   "submerged_index", "surface_index",
                   "cell_width", "pml_width", "empty_width", "source_center", "wlen_in_vacuum", 
                   "until_after_sources", "time_factor_cell", 
                   "n_period_line", "period_line", 
                   "n_flux_walls", "flux_wall_positions",
                   "parallel", "n_processes", "n_cores", "n_nodes",
                   "split_chunks_evenly", "hfield", "units",
                   "script", "sysname", "path"]
    
    #%% GENERAL GEOMETRY SETUP
    
    ### ROUND UP ACCORDING TO GRID DISCRETIZATION
        
    pml_width = vu.round_to_multiple(pml_width, 1/resolution)
    cell_width = vu.round_to_multiple(cell_width/2, 1/resolution)*2
    empty_width = cell_width/2 - pml_width
    source_center = -0.5*cell_width + pml_width
    
    until_after_sources = vu.round_to_multiple(until_after_sources, courant/resolution, round_up=True)
    period_line = vu.round_to_multiple(period_line, courant/resolution, round_down=True)
    
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
    
    sources = [mp.Source(mp.GaussianSource(freq_center,
                                           fwidth=freq_width,
                                           is_integrated=True,
                                           cutoff=cutoff),
                     center=mp.Vector3(source_center),
                     size=mp.Vector3(0, cell_width, cell_width),
                     component=mp.Ez)]
    # Ez-polarized monochromatic planewave 
    # (its size parameter fills the entire cell in 2d)
    # The planewave source extends into the PML 
    # ==> is_integrated=True must be specified
    
    flux_wall_positions = np.linspace(-cell_width/2+pml_width, 
                                      cell_width/2-pml_width, 
                                      n_flux_walls+2)[1:-1]
    
    #%% PLOT CELL

    params = {}
    for p in params_list: params[p] = eval(p)
    
    if pm.assign(0):
        
        plot_np_planewave_cell(params, series, folder, 
                               with_line=True, with_flux_walls=True, 
                               english=trs.english)
    
    #%% INITIALIZE
    
    # stable, max_courant = vm.check_stability(params)
    # if stable:
    #     pm.log("As a whole, the simulation should be stable")
    # else:
    #     pm.log("As a whole, the simulation could not be stable")
    #     pm.log(f"Recommended maximum courant factor is {max_courant}")
    # del stable, max_courant
    
    rm.measure_ram()
    
    # symmetries = [mp.Mirror(mp.Y), mp.Mirror(mp.Z, phase=-1)]
    
    sim = mp.Simulation(resolution=resolution,
                        Courant=courant,
                        geometry=geometry,
                        sources=sources,
                        k_point=mp.Vector3(),
                        default_material=mp.Medium(index=submerged_index),
                        cell_size=cell_size,
                        # symmetries=symmetries,
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

    save_line = mp.in_volume(sampling_line, 
                             mp.to_appended("Lines",
                                            mp.output_efield_z))
    
    to_do_while_running = [mp.at_every(period_line, save_line)]
    
    if hfield:
        save_hline = mp.in_volume(sampling_line, 
                                  mp.to_appended("HLines",
                                                 mp.output_hfield_y))
        to_do_while_running = [*to_do_while_running, 
                               mp.at_every(period_line, save_hline)]
    
    step_ram_function = lambda sim : rm.measure_ram()
    
    to_do_while_running = [*to_do_while_running,
                           mp.at_beginning(step_ram_function), 
                           mp.at_end(step_ram_function)]
    
    #%% ADD FLUX WALLS
    
    flux_walls = []
    for flux_x in flux_wall_positions:
        flux_walls.append( sim.add_flux(freq_center, freq_width, nfreq, 
                                        mp.FluxRegion(center=mp.Vector3(x=flux_x),
                                                      size=mp.Vector3(
                                                          0,
                                                          cell_width-2*pml_width,
                                                          cell_width-2*pml_width))) )
        
    rm.measure_ram()
    
    #%% RUN!
    
    rm.measure_ram()
    
    filename_prefix = sim.filename_prefix
    sim.filename_prefix = "Field"    
    os.chdir(path)
    rm.start_measure_time()
    sim.run(*to_do_while_running, until_after_sources=until_after_sources)
    rm.end_measure_time()
    os.chdir(syshome)
    sim.filename_prefix = filename_prefix
    
    #%% SAVE DATA AND METADATA
    
    for p in params_list: params[p] = eval(p)
    
    # Get fields metadata
    x, y, z, more = sim.get_array_metadata(vol=sampling_line)
    del more
    
    # Save metadata    
    if pm.assign(0):
        
        f = h5.File(sa.file("Field-Lines.h5"), "r+")
        keys = [vu.camel(k) for k in f.keys()]
        for oldk, newk in zip(list(f.keys()), keys):
            f[newk] = f[oldk]
            del f[oldk]

        f["T"] = np.arange(0, f["Ez"].shape[-1]) * period_line
        for k, array in zip(["X","Y","Z"], [x, y, z]): 
            f[k] = array
        
        for k in f.keys(): 
            for kpar in params.keys(): 
                f[k].attrs[kpar] = params[kpar]
        
        f.close()
        
        del f, keys, oldk, newk, k, kpar, array
    
        if hfield:
                
            fh = h5.File(sa.file("Field-HLines.h5"), "r+")
            keys = [vu.camel(k) for k in fh.keys()]
            for oldk, newk in zip(list(fh.keys()), keys):
                fh[newk] = fh[oldk]
                del fh[oldk]
    
            fh["T"] = np.arange(0, fh["Ez"].shape[-1]) * period_line
            for k, array in zip(["X","Y","Z"], [x, y, z]): 
                fh[k] = array
            
            for k in fh.keys(): 
                for kpar in params.keys(): 
                    fh[k].attrs[kpar] = params[kpar]
            
            fh.close()
            
            del fh, keys, oldk, newk, k, kpar, array

    # Get flux data
    flux_freqs = np.asarray(mp.get_flux_freqs(flux_walls[0]))
    flux_wlens = 1e3 * from_um_factor / flux_freqs
    flux_data = np.array([np.asarray(mp.get_fluxes(fw)) for fw in flux_walls])
    flux_intensity = np.array([fd/(cell_width - 2*pml_width)**2 for fd in flux_data])
    # Flux / Área
    
    # Organize flux data
    data = np.array([flux_wlens, *flux_intensity]).T
    header = [r"Longitud de onda $\lambda$ [nm]", 
              *[f"Intensidad del Flujo X{i} [u.a.]" for i in range(1,n_flux_walls+1)]]

    data_base = np.array([flux_freqs, *flux_data]).T
    header_base = [r"Frecuencia f $\lambda$ [uMP]", 
                   *[f"Flujo X{i} [u.a.]" for i in range(1,n_flux_walls+1)]]
    
    # Save flux data
    if pm.assign(0):
        vs.savetxt(sa.file("Results.txt"), data, 
                   header=header, footer=params, overwrite=True)
        vs.savetxt(sa.file("BaseResults.txt"), data_base, 
                   header=header_base, footer=params, overwrite=True)
    del data, data_base, header, header_base
    
    # Save resources control
    rm.save(sa.file("Resources.h5"), params)
    
    mp.all_wait()
    
    #%% ANALYSE AND PLOT DATA
    
    if make_plots or make_gifs:
        
        plots_pulse_field(series, folder, units, hfield, 
                          make_plots, make_gifs, trs.english)
    
    sim.reset_meep()

#%%

if __name__ == '__main__':
    main()
