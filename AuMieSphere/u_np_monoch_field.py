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

import click as cli
import imageio as mim
import h5py as h5
import matplotlib.pyplot as plt
import meep as mp
try: 
    from mpi4py import MPI
except:
    print("No mpi4py module found!")
import numpy as np
import os
from time import time
from v_materials import import_medium
import v_meep as vm
import v_save as vs
import v_utilities as vu

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
@cli.option("--split-chunks-evenly", "-chev", "split_chunks_evenly", 
            type=bool, default=True,
            help="Whether to split chunks evenly or not during parallel run")
@cli.option("--n-cores", "-nc", "n_cores", type=int, default=0,
            help="Number of cores used to run the program in parallel")
@cli.option("--n-nodes", "-nn", "n_nodes", type=int, default=0,
            help="Number of nodes used to run the program in parallel")
@cli.option("--save-hfield", "-savh", "save_hfield", type=bool, default=False,
            help="Whether to also save magnetic field or not")
def main(from_um_factor, resolution, courant,
         r, material, paper, wlen, submerged_index, 
         empty_r_factor, pml_wlen_factor, 
         time_factor_cell,
         series, folder,
         n_cores, n_nodes, split_chunks_evenly,
         save_hfield):
    
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
    
    # Box spatial dimensions
    pml_wlen_factor = 0.38
    empty_r_factor = 0.5
    
    # Sim temporal dimension
    time_factor_cell = 10    
    
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
    save_hfield = True
    """
    
    #%% ADDITIONAL PARAMETERS
    
    # Field Measurements
    period_line = 1
    period_plane = 1
    
    # Routine configuration
    english = False
    
    #%% TREATED PARAMETERS
    
    # Au sphere
    r = r  / ( from_um_factor * 1e3 )  # Radius of sphere now in Meep units
    medium = import_medium("Au", from_um_factor, paper=paper) # Medium of sphere: gold (Au)
    
    # Frequency and wavelength
    wlen = wlen / ( from_um_factor * 1e3 ) # Wavelength now in Meep units
    
    # Space configuration
    pml_width = pml_wlen_factor * wlen # 0.5 * wlen
    empty_width = empty_r_factor * r # 0.5 * max(wlen_range)
    
    # Computation time
    elapsed = []
    
    # Saving directories
    if series is None:
        series = f"AuSphereField{2*r*from_um_factor*1e3:.0f}WLen{wlen*from_um_factor*1e3:.0f}"
    if folder is None:
        folder = "AuMieSphere/AuSphereField"
    
    params_list = ["from_um_factor", "resolution", "courant",
                   "material", "r", "paper", "submerged_index", "wlen", 
                   "cell_width", "pml_width", "empty_width", "source_center",
                   "until_time", "time_factor_cell", "period_line", "period_plane",
                   "elapsed", "parallel", "n_processes", "n_cores", "n_nodes",
                   "split_chunks_evenly", "save_hfield",
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
    
    until_time = time_factor_cell * cell_width * submerged_index
    
    geometry = [mp.Sphere(material=medium,
                          center=mp.Vector3(),
                          radius=r)]
    # Au sphere with frequency-dependant characteristics imported from Meep.
    
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
    elapsed.append(time()-temp)
    
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
    
    if save_hfield:
        save_hline = mp.in_volume(sampling_line, 
                                  mp.to_appended("HLines",
                                                 mp.output_hfield_y))
        save_hplane = mp.in_volume(sampling_plane, 
                                   mp.to_appended("HPlanes",
                                                  mp.output_hfield_y))
        to_do_while_running = [*to_do_while_running, 
                               mp.at_every(period_line, save_hline),
                               mp.at_every(period_plane, save_hplane)]
    
    #%% RUN!
    
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
    
        if save_hfield:
            
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
    
    mp.all_wait()
    
    # #%% GET READY TO LOAD DATA
    
    # if parallel: 
    #     f = h5.File(file("Field-Lines.h5"), "r+", 
    #                 driver='mpio', comm=MPI.COMM_WORLD)
    # else:
    #     f = h5.File(file(fil + ".h5"), "r+")
    # results_line = f["Ez"]
    
    # if parallel: 
    #     g = h5.File(file("Field-Planes.h5"), "r+", 
    #                 driver='mpio', comm=MPI.COMM_WORLD)
    # else:
    #     g = h5.File(file("Field-Planes.h5"), "r+")
    # results_plane = g["Ez"]
    
    # t_line = np.arange(0, until_time, period_line)
    # t_plane = np.arange(0, until_time, period_plane)
    
    # x, y, z, more = sim.get_array_metadata()
    # del more
    
    # #%% GENERAL USEFUL FUNCTIONS
    
    # t_line_index = lambda t0 : np.argmin(np.abs(t_line - t0))
    # t_plane_index = lambda t0 : np.argmin(np.abs(t_plane - t0))
    
    # x_index = lambda x0 : np.argmin(np.abs(x - x0))
    # y_index = lambda y0 : np.argmin(np.abs(y - y0))
    # z_index = lambda z0 : np.argmin(np.abs(z - z0))
    
    # #%% SHOW SOURCE
    
    # source_index = x_index(source_center)
    # source_field = np.asarray(results_line[source_index, :])
    
    # plt.figure()
    # plt.plot(t_line, source_field)
    # plt.xlabel("Tiempo [u.Mp.]")
    # plt.ylabel("Campo eléctrico Ez [uMp]")
    
    # plt.savefig(file("Source.png"))
    
    # #%% MAKE FOURIER FOR SOURCE
    
    # fourier = np.abs(np.fft.rfft(source_field))
    # fourier_freq = np.fft.rfftfreq(len(source_field), d=period_line)
    # fourier_wlen = 1 / fourier_freq
    
    # plt.figure()
    # plt.plot(fourier_freq, fourier, 'k', linewidth=3)
    # plt.xlabel("Frecuencia [u.Mp.]")
    # plt.ylabel("Transformada del campo eléctrico Ez [u.a.]")
    
    # plt.savefig(file("SourceFFT.png"))
    
    # #%% MAKE PLANE GIF
    
    # # What should be parameters
    # nframes_step = 1
    # nframes = int(results_plane.shape[-1]/nframes_step)
    # call_series = lambda i : results_plane[:,:,i].T
    # label_function = lambda i : 'Tiempo: {:.1f} u.Mp.'.format(i*period_plane)
    
    # # Animation base
    # fig = plt.figure()
    # ax = plt.subplot()
    # lims = (np.min(results_plane), np.max(results_plane))
    
    # def draw_pml_box():
    #     plt.hlines(z_index(-cell_width/2 + pml_width), 
    #                 y_index(-cell_width/2 + pml_width), 
    #                 y_index(cell_width/2 - pml_width),
    #                 linestyle=":", color='k')
    #     plt.hlines(z_index(cell_width/2 - pml_width), 
    #                 y_index(-cell_width/2 + pml_width), 
    #                 y_index(cell_width/2 - pml_width),
    #                 linestyle=":", color='k')
    #     plt.vlines(y_index(-cell_width/2 + pml_width), 
    #                 z_index(-cell_width/2 + pml_width), 
    #                 z_index(cell_width/2 - pml_width),
    #                 linestyle=":", color='k')
    #     plt.vlines(y_index(cell_width/2 - pml_width), 
    #                 z_index(-cell_width/2 + pml_width), 
    #                 z_index(cell_width/2 - pml_width),
    #                 linestyle=":", color='k')
    
    # def make_pic_plane(i):
    #     ax.clear()
    #     plt.imshow(call_series(i), interpolation='spline36', cmap='RdBu', 
    #                 vmin=lims[0], vmax=lims[1])
    #     ax.text(-.1, -.105, label_function(i), transform=ax.transAxes)
    #     draw_pml_box()
    #     plt.show()
    #     plt.xlabel("Distancia Y [u.Mp.]")
    #     plt.ylabel("Distancia Z [u.Mp.]")
    #     return ax
    
    # def make_gif_plane(gif_filename):
    #     pics = []
    #     for i in range(nframes):
    #         ax = make_pic_plane(i*nframes_step)
    #         plt.savefig('temp_pic.png') 
    #         pics.append(mim.imread('temp_pic.png')) 
    #         print(str(i+1)+'/'+str(nframes))
    #     mim.mimsave(gif_filename+'.gif', pics, fps=5)
    #     os.remove('temp_pic.png')
    #     print('Saved gif')
    
    # make_gif_plane(file("PlaneX=0"))
    # plt.close(fig)
    # # del fig, ax, lims, nframes_step, nframes, call_series, label_function
    
    # ### HASTA ACÁ LLEGUÉ
    
    # #%% GET Z LINE ACROSS SPHERE
    
    # index_to_space = lambda i : i/resolution - cell_width/2
    # space_to_index = lambda x : round(resolution * (x + cell_width/2))
    
    # z_profile = np.asarray(results_plane[:, space_to_index(0), :])
    
    # #%% MAKE LINES GIF
    
    # # What should be parameters
    # nframes_step = 1
    # nframes = int(z_profile.shape[0]/nframes_step)
    # call_series = lambda i : z_profile[i, :]
    # label_function = lambda i : 'Tiempo: {:.1f} u.a.'.format(i*period_plane)
    
    # # Animation base
    # fig = plt.figure()
    # ax = plt.subplot()
    # lims = (np.min(z_profile), np.max(z_profile))
    # shape = call_series(0).shape[0]
    
    # def draw_pml_box():
    #     plt.vlines(-cell_width/2 + pml_width, 
    #                 -cell_width/2 + pml_width, 
    #                 cell_width/2 - pml_width,
    #                 linestyle=":", color='k')
    #     plt.vlines(cell_width/2 - pml_width, 
    #                 -cell_width/2 + pml_width, 
    #                 cell_width/2 - pml_width,
    #                 linestyle=":", color='k')
    
    # def make_pic_line(i):
    #     ax.clear()
    #     plt.plot(np.linspace(-cell_width/2, cell_width/2, shape), 
    #               call_series(i))
    #     ax.set_ylim(*lims)
    #     ax.text(-.12, -.1, label_function(i), transform=ax.transAxes)
    #     draw_pml_box()
    #     plt.xlabel("Distancia en z (u.a.)")
    #     plt.ylabel("Campo eléctrico Ez (u.a.)")
    #     plt.show()
    #     return ax
    
    # def make_gif_line(gif_filename):
    #     pics = []
    #     for i in range(nframes):
    #         ax = make_pic_line(i*nframes_step)
    #         plt.savefig('temp_pic.png') 
    #         pics.append(mim.imread('temp_pic.png')) 
    #         print(str(i+1)+'/'+str(nframes))
    #     mim.mimsave(gif_filename+'.gif', pics, fps=5)
    #     os.remove('temp_pic.png')
    #     print('Saved gif')
    
    # make_gif_line(file("AxisZ"))
    # plt.close(fig)
    # # del fig, ax, lims, nframes_step, nframes, call_series, label_function
    
    # #%% MAKE LINES GIF
    
    # # What should be parameters
    # nframes_step = 1
    # nframes = int(results_line.shape[0]/nframes_step)
    # call_series = lambda i : results_line[i]
    # label_function = lambda i : 'Tiempo: {:.1f} u.a.'.format(i*period_line)
    
    # # Animation base
    # fig = plt.figure()
    # ax = plt.subplot()
    # lims = (np.min(results_line), np.max(results_line))
    # shape = call_series(0).shape[0]
    
    # def draw_pml_box():
    #     plt.vlines(-cell_width/2 + pml_width, 
    #                 -cell_width/2 + pml_width, 
    #                 cell_width/2 - pml_width,
    #                 linestyle=":", color='k')
    #     plt.vlines(cell_width/2 - pml_width, 
    #                 -cell_width/2 + pml_width, 
    #                 cell_width/2 - pml_width,
    #                 linestyle=":", color='k')
    
    # def make_pic_line(i):
    #     ax.clear()
    #     plt.plot(np.linspace(-cell_width/2, cell_width/2, shape), 
    #               call_series(i))
    #     ax.set_ylim(*lims)
    #     ax.text(-.12, -.1, label_function(i), transform=ax.transAxes)
    #     draw_pml_box()
    #     plt.xlabel("Distancia en x (u.a.)")
    #     plt.ylabel("Campo eléctrico Ez (u.a.)")
    #     plt.show()
    #     return ax
    
    # def make_gif_line(gif_filename):
    #     pics = []
    #     for i in range(nframes):
    #         ax = make_pic_line(i*nframes_step)
    #         plt.savefig('temp_pic.png') 
    #         pics.append(mim.imread('temp_pic.png')) 
    #         print(str(i+1)+'/'+str(nframes))
    #     mim.mimsave(gif_filename+'.gif', pics, fps=5)
    #     os.remove('temp_pic.png')
    #     print('Saved gif')
    
    # make_gif_line(file("AxisX"))
    # plt.close(fig)
    # # del fig, ax, lims, nframes_step, nframes, call_series, label_function
    
    
    #%%
    
    # f.close()
    # g.close()
    
    sim.reset_meep()

#%%

if __name__ == '__main__':
    main()
