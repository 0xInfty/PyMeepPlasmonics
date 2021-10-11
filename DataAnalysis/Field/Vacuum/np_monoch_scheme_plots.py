#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis of single run of Au sphere field for a visible monochromatic incident wave.

See also
--------
Routines/np_monoch_field
"""

import imageio as mim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import v_meep as vm
import v_meep_analysis as vma
import v_plot as vp
import v_utilities as vu

vp.set_style()

#%% PARAMETERS

"""
series = "DTest532"
folder = "Field/NPMonoch/AuSphere/VacWatTest/DefinitiveTest/Vacuum"

hfield = False

make_plots = True
make_gifs = True

english = False
maxnframes = 300
"""

#%%

def plot_np_monoch_field_cell(params, series, folder, 
                              with_nanoparticle=True, english=False):
    
    #%% SETUP
    
    # Computation
    pm = vm.ParallelManager()
    n_processes, n_cores, n_nodes = pm.specs
    parallel = pm.parallel
    
    # Saving directories
    sa = vm.SavingAssistant(series, folder)
    home = sa.home
    sysname = sa.sysname
    path = sa.path
    
    trs = vu.BilingualManager(english=english)
    
    #%% DATA EXTRACTION
    
    from_um_factor = params["from_um_factor"]
    
    cell_width = params["cell_width"]
    pml_width = params["pml_width"]
    
    r = params["r"]
    material = params["material"]
    
    wlen = params["wlen"]
    source_center = params["source_center"]
    
    submerged_index = params["submerged_index"]
    surface_index = params["surface_index"]
    overlap = params["overlap"]
    
    #%% PLOT
    
    if pm.assign(0): 
    
        fig, ax = plt.subplots()
        ax.grid(False)
                
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
            surface_square = plt.Rectangle((r - overlap, -cell_width/2),
                                           cell_width/2 - r + overlap, 
                                           cell_width,
                                           edgecolor="navy", hatch=r"\\", 
                                           fill=False, zorder=-3,
                                           label=trs.choose(fr"Surface $n$={surface_index}",
                                                            fr"Superficie $n$={surface_index}"))
        # Nanoparticle
        if with_nanoparticle:
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
        if with_nanoparticle:
            ax.vlines(0, -cell_width/2, cell_width/2,
                      color="limegreen", linestyle="dashed", zorder=7, 
                      label=trs.choose("Sampling Plane", "Plano de muestreo"))
        
        if with_nanoparticle: ax.add_patch(circle)
        if submerged_index!=1: ax.add_patch(surrounding_square)
        if surface_index!=submerged_index: ax.add_patch(surface_square)
        ax.add_patch(pml_out_square)
        ax.add_patch(pml_inn_square)
        
        # General configuration
        box = ax.get_position()
        width = box.x1 - box.x0
        height = box.y1 - box.y0
        box.x0 = box.x0 - .26 * width
        box.x1 = box.x0 + width
        box.y0 = box.y0 + .12 * height
        box.y1 = box.y0 + height
        ax.set_position(box)           
        plt.legend(bbox_to_anchor=trs.choose( (1.8, 0.5), (1.87, 0.5) ), 
                   loc="center right", frameon=False)
        
        fig.set_size_inches(7.5, 4.8)
        ax.set_aspect("equal")
        plt.xlim(-cell_width/2, cell_width/2)
        plt.ylim(-cell_width/2, cell_width/2)
        plt.xlabel(trs.choose(r"Position $X$ [MPu]", r"Posición $X$ [uMP]"))
        plt.ylabel(trs.choose(r"Position $Z$ [MPu]", r"Posición $Z$ [uMP]"))
        
        plt.annotate(fr"$\lambda$ = {wlen * from_um_factor * 1e3:.0f} nm",
                     (220, 15), xycoords='figure points', color="r")
        
        plt.annotate(trs.choose(f"1 MPu = {from_um_factor * 1e3:.0f} nm",
                                f"1 uMP = {from_um_factor * 1e3:.0f} nm"),
                     (5, 15), xycoords='figure points')
        
        fig.set_size_inches([6.68, 3.68])
        
        #%%
        
        if with_nanoparticle:
            plt.savefig(sa.file("MiniSimBox.png"))
        else:
            plt.savefig(sa.file("MiniSimBoxNorm.png"))
            
#%% 

def plots_np_monoch_field(series, folder, hfield=False, 
                          make_plots=True, make_gifs=False, 
                          english=False, maxnframes=300):
        
    #%% SETUP
    
    # Computation
    pm = vm.ParallelManager()
    n_processes, n_cores, n_nodes = pm.specs
    parallel = pm.parallel
    
    # Saving directories
    sa = vm.SavingAssistant(series, folder)
    home = sa.home
    sysname = sa.sysname
    path = sa.path
    
    trs = vu.BilingualManager(english=english)
    
    #%% EXIT IF NOTHING TO BE DONE
    
    if not make_plots and not make_gifs:
        return
    
    #%% GET READY TO LOAD DATA
        
    f = pm.hdf_file(sa.file("Field-Lines.h5"), "r+")
    results_line = f["Ez"]
    t_line = np.array(f["T"])
    x_line = np.array(f["X"])
    
    g = pm.hdf_file(sa.file("Field-Planes.h5"), "r+")
    results_plane = g["Ez"]
    t_plane = np.array(g["T"])
    y_plane = np.array(g["Y"])
    z_plane = np.array(g["Z"])
    
    params = dict(f["Ez"].attrs)
    
    from_um_factor = params["from_um_factor"]
    
    r = params["r"]
    material = params["material"]
    
    wlen = params["wlen"]
    
    cell_width = params["cell_width"]
    pml_width = params["pml_width"]
    source_center = params["source_center"]
    
    until_time = params["until_time"]
    period_line = params["period_line"]
    period_plane = params["period_plane"]
    
    try:
        norm_until_time = params["norm_until_time"]
        norm_amplitude = params["norm_amplitude"]
        norm_period = params["norm_period"]
        norm_path = params["norm_path"]
        requires_normalization = False
    except:
        requires_normalization = True
       
    try:
        
        try:
            h = pm.hdf_file(sa.file("Field-Lines-Norm.h5"), "r")
            
        except:
            h = pm.hdf_file(os.path.join(norm_path, "Field-Lines-Norm.h5"), "r")
            
        print("Loaded available normfield")
        
    except:
        
        norm_path = vm.check_normfield(params)
        
        try:
            h = pm.hdf_file(os.path.join(norm_path[0], "Field-Lines-Norm.h5"), "r")
            print("Loaded compatible normfield")
            
        except:
            h = f
            print(f"Using NP data for normalization.",
                  "This is by all means not ideal!!",
                  "If possible, run again these simulations.")
    
    results_line_norm = h["Ez"]
    t_line_norm = np.asarray(h["T"])
    x_line_norm = np.asarray(h["X"])
    
    #%% POSITION RECONSTRUCTION
    
    t_line_index = vma.def_index_function(t_line)
    x_line_index = vma.def_index_function(x_line)
    
    t_plane_index = vma.def_index_function(t_plane)
    y_plane_index = vma.def_index_function(y_plane)
    z_plane_index = vma.def_index_function(z_plane)
    
    t_line_norm_index = vma.def_index_function(t_line_norm)
    x_line_norm_index = vma.def_index_function(x_line_norm)
        
    x_line_cropped = x_line[:x_line_index(cell_width/2 - pml_width)+1]
    x_line_cropped = x_line_cropped[x_line_index(-cell_width/2 + pml_width):]

    y_plane_cropped = y_plane[:y_plane_index(cell_width/2 - pml_width)+1]
    y_plane_cropped = y_plane_cropped[y_plane_index(-cell_width/2 + pml_width):]
    
    z_plane_cropped = z_plane[:z_plane_index(cell_width/2 - pml_width)+1]
    z_plane_cropped = z_plane_cropped[z_plane_index(-cell_width/2 + pml_width):]
    
    #%% DATA EXTRACTION
        
    source_results = vma.get_source_from_line(results_line_norm, 
                                              x_line_norm_index, 
                                              source_center)
        
    if not requires_normalization:
        
        period, amplitude = norm_period, norm_amplitude
        
    else:          
        
        period = vma.get_period_from_source(source_results, t_line)[-1]
        amplitude = vma.get_amplitude_from_source(source_results)[-1]
        
        results_plane = np.asarray(results_plane) / amplitude
        results_line = np.asarray(results_line) / amplitude
    
    sim_source_results = vma.get_source_from_line(results_line, 
                                                  x_line_index, 
                                                  source_center)

    zprofile_results = vma.get_zprofile_from_plane(results_plane, y_plane_index)
    
    zprofile_zintegral = vma.z_integrate_field_zprofile(zprofile_results, z_plane, 
                                                       z_plane_index,
                                                       cell_width, pml_width)
    
    zprofile_max = vma.find_zpeaks_zprofile(zprofile_results, z_plane_index,
                                            cell_width, pml_width)[-1]
    
    field_peaks_all_index = vma.get_all_field_peaks_from_yzplanes(results_plane,
                                                                  y_plane_index, 
                                                                  z_plane_index, 
                                                                  cell_width, 
                                                                  pml_width)[0]
    
    field_peaks_results = vma.get_mean_field_peak_from_yzplanes(results_plane,
                                                                y_plane_index, 
                                                                z_plane_index, 
                                                                cell_width, 
                                                                pml_width)
    field_peaks_single_index = field_peaks_results[0]
    field_peaks_zprofile = field_peaks_results[2]
    field_peaks_plane = field_peaks_results[3]
    del field_peaks_results
    
    zprofile_cropped = vma.crop_field_zprofile(zprofile_results,
                                               z_plane_index,
                                               cell_width,
                                               pml_width)
    results_plane_cropped = vma.crop_field_yzplane(results_plane, 
                                                   y_plane_index, 
                                                   z_plane_index,
                                                   cell_width,
                                                   pml_width)
    
    #%% SHOW SOURCE AND FOURIER USED FOR NORMALIZATION
    
    if make_plots and pm.assign(1): 
    
        fig = plt.figure()
        plt.plot(t_line_norm, source_results/amplitude, linewidth=2)
        plt.xlabel(trs.choose("Time [MPu]", "Tiempo [uMP]"))
        plt.ylabel(trs.choose(r"Electric Field $E_z(y=z=0)$", 
                              r"Campo eléctrico $E_z(y=z=0)$"))
        
        fig.set_size_inches([3.69, 3.15])
        plt.tight_layout()
        
        plt.savefig(sa.file("MiniSource.png"))
        
    #%% SHOW FIELD PEAKS INTENSIFICATION OSCILLATIONS
    
    if make_plots and pm.assign(0):
        
        fig = plt.figure()
        plt.plot(t_plane, zprofile_max, linewidth=2)
        plt.xlabel(trs.choose("Time [MPu]", "Tiempo [uMP]"))
        plt.ylabel(trs.choose("Electric Field Maximum\n"+r"$max[ E_z(z) ]$",
                              "Máximo del campo eléctrico\n"+r"$max[ E_z(z) ]$"))
        
        fig.set_size_inches([4.12, 3.15])
        plt.tight_layout()
        
        plt.savefig(sa.file("MiniMaximum.png"))
        
    #%% PLOT MAXIMUM INTENSIFICATION PROFILE
        
    if make_plots and pm.assign(1): 
        
        fig = plt.figure()
        plt.plot(z_plane_cropped, 
                 field_peaks_zprofile, linewidth=2)
        plt.axhline(linewidth=1, color="k")
        plt.axvline(linewidth=1, color="k")
        plt.axvline(-r, color="k", linestyle="dashed", linewidth=1)
        plt.axvline(r, color="k", linestyle="dashed", linewidth=1)
        
        plt.xlim(min(z_plane_cropped), max(z_plane_cropped))
        plt.xlabel(trs.choose(r"Position $Z$ [MPu]", r"Posición $Z$ [uMP]"))
        plt.ylabel(trs.choose(r"Electric Field $E_z(y=z=0)$",
                              r"Campo eléctrico $E_z(y=z=0)$"))
        
        fig.set_size_inches([3.69, 3.15])
        plt.tight_layout()
        
        plt.savefig(sa.file("MiniIntensification.png"))
        
    #%% PLOT A PIECE OF THE NORMALIZATION LINES FILE
        
    if make_plots and pm.assign(1): 
        
        fig = plt.figure()
        plt.plot(x_line_norm, 
                 results_line_norm[...,200], linewidth=2)
        plt.axhline(linewidth=1, color="k")
        plt.axvline(linewidth=1, color="k")
        plt.axvline(-cell_width/2 + pml_width, color="k", linestyle="dashed")
        plt.axvline(cell_width/2 - pml_width, color="k", linestyle="dashed")
        
        plt.xlim(min(x_line_norm), max(x_line_norm))
        plt.xlabel(trs.choose(r"Position $X$ [MPu]", r"Posición $X$ [uMP]"))
        plt.ylabel(trs.choose(r"Electric Field $E_z(y=z=0)$",
                              r"Campo eléctrico $E_z(y=z=0)$"))
        
        fig.set_size_inches([3.69, 3.15])
        plt.tight_layout()
        
        plt.savefig(sa.file("MiniNormLines.png"))
    
    #%% MAKE ALL GIF
    
    if make_gifs and pm.assign(1):
        
        # What should be parameters
        nframes = min(maxnframes, results_plane.shape[-1])
        nframes = int( vu.round_to_multiple(nframes, params["time_period_factor"] ) )
        nframes_period = int(nframes/np.max(params["time_period_factor"]))
        
        cut_points = []
        for k in range(int(params["time_period_factor"])):
            cut_points.append( np.argmin(np.abs(t_line / period - (k + 1))) )
        cut_points = [0, *cut_points]
        
        frames_index = []
        for k in range(len(cut_points)):
            if k < len(cut_points)-1:
                intermediate_frames = np.linspace(
                    cut_points[k],
                    cut_points[k+1],
                    nframes_period+1)[:-1]
                intermediate_frames = [int(fr) for fr in intermediate_frames]
                frames_index = [*frames_index, *intermediate_frames]
        
        # Animation base
        fig = plt.figure()                
        plot_grid = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
        line_ax = fig.add_subplot(plot_grid[:,0])
        plane_ax = fig.add_subplot(plot_grid[:,-1])
        fig.set_size_inches([8.07 , 3.47])
        
        # line_ax.set_aspect('equal')
        line_ax_lims = (np.min(results_line), np.max(results_line))
        
        plane_ax.set_aspect('equal')
        plane_ax_lims = (np.min(results_plane), np.max(results_plane))
        plane_ax_lims = max([abs(l) for l in plane_ax_lims])
        plane_ax_lims = [-plane_ax_lims, plane_ax_lims]
        move_on_k = 330
        
        def make_pic(k):
            line_ax.clear()
            plane_ax.clear()
            
            line_ax.plot(x_line, results_line[...,k], linewidth=2)
            line_ax.axvline(-cell_width/2 + pml_width, color="k", linestyle="dashed")
            line_ax.axvline(cell_width/2 - pml_width, color="k", linestyle="dashed")
            line_ax.axhline(0, color="k", linewidth=1)
            line_ax.axvline(0, color="k", linewidth=1)
            line_ax.axvline(r, color="k", linestyle="dotted", linewidth=1)
            line_ax.axvline(-r, color="k", linestyle="dotted", linewidth=1)
        
            line_ax.set_xlabel(trs.choose("Position $X$ [MPu]", "Posición $X$ [uMP]"))
            line_ax.set_ylabel(trs.choose(r"Electric Field $E_z(y=z=0)$",
                                          r"Campo eléctrico $E_z(y=z=0)$"))
            line_ax.set_xlim(min(x_line), max(x_line))
            
            ims = plane_ax.imshow(results_plane[...,k].T,
                                  cmap='RdBu', #interpolation='spline36', 
                                  vmin=plane_ax_lims[0], vmax=plane_ax_lims[1],
                                  extent=[min(y_plane), max(y_plane),
                                          min(z_plane), max(z_plane)])
            plane_ax.axhline(-cell_width/2 + pml_width, color="k", linestyle="dashed", linewidth=1)
            plane_ax.axhline(cell_width/2 - pml_width, color="k", linestyle="dashed", linewidth=1)
            plane_ax.axvline(-cell_width/2 + pml_width, color="k", linestyle="dashed", linewidth=1)
            plane_ax.axvline(cell_width/2 - pml_width, color="k", linestyle="dashed", linewidth=1)
            plane_ax.axhline(0, color="k", linewidth=1)
            plane_ax.axvline(0, color="k", linewidth=1)
            plane_ax.axhline(r, color="k", linestyle="dotted", linewidth=1)
            plane_ax.axhline(-r, color="k", linestyle="dotted", linewidth=1)
            plane_ax.axvline(r, color="k", linestyle="dotted", linewidth=1)
            plane_ax.axvline(-r, color="k", linestyle="dotted", linewidth=1)
            
            plane_ax.grid(False)
            plane_ax.set_xlabel(trs.choose("Position $Y$ [MPu]", "Posición $Y$ [uMP]"))
            plane_ax.set_ylabel(trs.choose("Position $Z$ [MPu]", "Posición $Z$ [uMP]"))
            
            if k == move_on_k:
                for ax in [line_ax, plane_ax]:
                    box = ax.get_position()
                    width = box.x1 - box.x0
                    height = box.y1 - box.y0
                    box.x0 = box.x0 - .1 * width
                    box.x1 = box.x0 + 0.9 * width
                    box.y0 = box.y0 + .1 * height
                    box.y1 = box.y0 + 0.9 * height
                    ax.set_position(box)

            line_ax.text(-.15, 1.03, trs.choose(f'Time: {t_line[k]/period:.1f} MPu',
                                                f'Tiempo: {t_line[k]/period:.1f} uMP'), 
                         transform=line_ax.transAxes)
            plt.annotate(trs.choose(f"1 MPu = {from_um_factor * 1e3:.0f} nm",
                                    f"1 uMP = {from_um_factor * 1e3:.0f} nm"),
                         (425, 225), xycoords='figure points')
            line_ax.set_ylim(*line_ax_lims)
            
            cax = plane_ax.inset_axes([1.04, 0, 0.07, 1], #[1.04, 0.2, 0.05, 0.6], 
                                      transform=plane_ax.transAxes)
            cbar = fig.colorbar(ims, ax=plane_ax, cax=cax)
            cbar.set_label(trs.choose("Electric Field $E_z$",
                                      "Campo eléctrico $E_z$"))
            
            plt.show()
            return line_ax, plane_ax
        
        def make_gif(gif_filename):
            pics = []
            for ik, k in enumerate(frames_index):
                line_ax, plane_ax = make_pic(k)
                plt.savefig('temp_pic.png') 
                pics.append(mim.imread('temp_pic.png')) 
                print(str(ik+1)+'/'+str(nframes))
            mim.mimsave(gif_filename+'.gif', pics, fps=5)
            os.remove('temp_pic.png')
            print('Saved gif') 
            
            # 330
            
        #%%
                
        make_gif(sa.file("All"))
        plt.close(fig)
        # del fig, ax, lims, nframes_step, nframes, call_series, label_function
                
    #%%
    
    f.close()
    g.close()
    try:
        h.close()
    except:
        pass