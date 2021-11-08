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
import vmp_utilities as vmu
import vmp_analysis as vma
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

def plots_np_monoch_field(series, folder, hfield=False, 
                          make_plots=True, make_gifs=False, 
                          english=False, maxnframes=300):
        
    #%% SETUP
    
    # Computation
    pm = vmu.ParallelManager()
    n_processes, n_cores, n_nodes = pm.specs
    parallel = pm.parallel
    
    # Saving directories
    sa = vmu.SavingAssistant(series, folder)
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
        
        norm_path = vmu.check_normfield(params)
        
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
    
    if make_plots and pm.assign(0): 

        plt.figure()        
        plt.title(trs.choose('Monochromatic wave {:.0f} nm on {} Sphere With {:.1f} nm Diameter',
                             'Onda monocromática {:.0f} nm sobre esfera de {} con diámetro {:.1f} nm'
                             ).format(wlen*from_um_factor*1e3, material, 2*r*from_um_factor*1e3 ))
        plt.plot(t_line_norm, source_results)
        plt.xlabel(trs.choose("Time [MPu]", "Tiempo [uMP]"))
        plt.ylabel(trs.choose(r"Electric Field $E_z(y=z=0)$ [a.u.]", 
                              r"Campo eléctrico $E_z(y=z=0)$ [u.a.]"))
        
        plt.savefig(sa.file("Source.png"))
        
        fourier = np.abs(np.fft.rfft(source_results))
        fourier_freq = np.fft.rfftfreq(len(source_results), d=period_line)
        fourier_wlen = from_um_factor * 1e3 / fourier_freq
        fourier_max_wlen = fourier_wlen[ np.argmax(fourier) ]
        
        plt.figure()
        plt.title(trs.choose('Monochromatic wave {:.0f} nm on {} Sphere With {:.1f} nm Diameter',
                             'Onda monocromática {:.0f} nm sobre esfera de {} con diámetro {:.1f} nm'
                             ).format(wlen*from_um_factor*1e3, material, 2*r*from_um_factor*1e3 ))
        plt.plot(fourier_wlen, fourier, 'k', linewidth=3)
        plt.xlabel(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
        plt.ylabel(trs.choose(r"Electric Field Fourier $\mathcal{F}\;(E_z)$ [u.a.]",
                              r"Transformada del campo eléctrico $\mathcal{F}\;(E_z)$ [u.a.]"))
        
        plt.annotate(trs.choose(f"Maximum at {fourier_max_wlen:.2f} nm",
                                f"Máximo en {fourier_max_wlen:.2f} nm"),
                     (5, 5), xycoords='figure points')
        
        plt.savefig(sa.file("SourceFFT.png"))
        
        plt.xlim([350, 850])
                
        plt.savefig(sa.file("SourceFFTZoom.png"))
        
    #%% SHOW SOURCE AND FOURIER DURING SIMULATION
    
    if make_plots and pm.assign(1): 
    
        plt.figure()        
        plt.title(trs.choose('Monochromatic wave {:.0f} nm on {} Sphere With {:.1f} nm Diameter',
                             'Onda monocromática {:.0f} nm sobre esfera de {} con diámetro {:.1f} nm'
                             ).format(wlen*from_um_factor*1e3, material, 2*r*from_um_factor*1e3 ))
        plt.plot(t_line, sim_source_results)
        plt.xlabel(trs.choose("Time [MPu]", "Tiempo [uMP]"))
        plt.ylabel(trs.choose(r"Electric Field $E_z(y=z=0)$ [a.u.]", 
                              r"Campo eléctrico $E_z(y=z=0)$ [u.a.]"))
        
        plt.savefig(sa.file("SimSource.png"))
        
        fourier = np.abs(np.fft.rfft(sim_source_results))
        fourier_freq = np.fft.rfftfreq(len(sim_source_results), d=period_line)
        fourier_wlen = from_um_factor * 1e3 / fourier_freq
        fourier_max_wlen = fourier_wlen[ np.argmax(fourier) ]
        
        plt.figure()
        plt.title(trs.choose('Monochromatic wave {:.0f} nm on {} Sphere With {:.1f} nm Diameter',
                             'Onda monocromática {:.0f} nm sobre esfera de {} con diámetro {:.1f} nm'
                             ).format(wlen*from_um_factor*1e3, material, 2*r*from_um_factor*1e3 ))
        plt.plot(fourier_wlen, fourier, 'k', linewidth=3)
        plt.xlabel(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
        plt.ylabel(trs.choose(r"Electric Field Fourier $\mathcal{F}\;(E_z)$ [u.a.]",
                              r"Transformada del campo eléctrico $\mathcal{F}\;(E_z)$ [u.a.]"))
        
        plt.annotate(trs.choose(f"Maximum at {fourier_max_wlen:.2f} nm",
                                f"Máximo en {fourier_max_wlen:.2f} nm"),
                     (5, 5), xycoords='figure points')
        
        plt.savefig(sa.file("SimSourceFFT.png"))
        
        plt.xlim([350, 850])
                
        plt.savefig(sa.file("SimSourceFFTZoom.png"))
        
    #%% SHOW FIELD PEAKS INTENSIFICATION OSCILLATIONS
    
    if make_plots and pm.assign(0):

        plt.figure()        
        plt.title(trs.choose('Monochromatic wave {:.0f} nm on {} Sphere With {:.1f} nm Diameter',
                             'Onda monocromática {:.0f} nm sobre esfera de {} con diámetro {:.1f} nm'
                             ).format(wlen*from_um_factor*1e3, material, 2*r*from_um_factor*1e3 ))
        plt.plot(t_plane, zprofile_zintegral)
        plt.xlabel(trs.choose("Time [MPu]", "Tiempo [uMP]"))
        plt.ylabel(trs.choose(r"Electric Field Integral $\int E_z(z) \; dz$ [a.u.]",
                              r"Integral del campo eléctrico $\int E_z(z) \; dz$ [u.a.]"))
        
        plt.savefig(sa.file("Integral.png"))
        
        plt.figure()
        plt.title(trs.choose('Monochromatic wave {:.0f} nm on {} Sphere With {:.1f} nm Diameter',
                             'Onda monocromática {:.0f} nm sobre esfera de {} con diámetro {:.1f} nm'
                             ).format(wlen*from_um_factor*1e3, material, 2*r*from_um_factor*1e3 ))
        plt.plot(t_plane, zprofile_max)
        plt.xlabel(trs.choose("Time [MPu]", "Tiempo [uMP]"))
        plt.ylabel(trs.choose(r"Electric Field Maximum $max[ E_z(z) ]$",
                              r"Máximo del campo eléctrico $max[ E_z(z) ]$"))
        
        plt.savefig(sa.file("Maximum.png"))
               
    #%% PLOT MAXIMUM INTENSIFICATION PROFILE
        
    if make_plots and pm.assign(1): 
        
        plt.figure()
        plt.title(trs.choose('Monochromatic wave {:.0f} nm on {} Sphere With {:.1f} nm Diameter',
                             'Onda monocromática {:.0f} nm sobre esfera de {} con diámetro {:.1f} nm'
                             ).format(wlen*from_um_factor*1e3, material, 2*r*from_um_factor*1e3 ))
        plt.plot(z_plane, zprofile_results[..., field_peaks_single_index])
        plt.axvline(-cell_width/2 + pml_width, color="k", linestyle="dashed")
        plt.axvline(cell_width/2 - pml_width, color="k", linestyle="dashed")
        
        # plt.xlim(min(z_plane), max(z_plane))
        plt.xlabel(trs.choose("Position Z [MPu]", "Position Z [uMP]"))
        plt.ylabel(trs.choose(r"Electric Field $E_z(y=z=0)$",
                              r"Campo eléctrico $E_z(y=z=0)$"))
        
        plt.savefig(sa.file("ZProfile.png"))
        
        plt.figure()
        plt.title(trs.choose('Monochromatic wave {:.0f} nm on {} Sphere With {:.1f} nm Diameter',
                             'Onda monocromática {:.0f} nm sobre esfera de {} con diámetro {:.1f} nm'
                             ).format(wlen*from_um_factor*1e3, material, 2*r*from_um_factor*1e3 ))
        plt.plot(z_plane_cropped  * from_um_factor * 1e3, 
                 field_peaks_zprofile)
        plt.axvline(-r * from_um_factor * 1e3, color="k", linestyle="dashed")
        plt.axvline(r * from_um_factor * 1e3, color="k", linestyle="dashed")
        
        # plt.xlim(min(z_plane_cropped) * from_um_factor * 1e3, 
        #          max(z_plane_cropped) * from_um_factor * 1e3)
        plt.xlabel(trs.choose("Position Z [nm]", "Position Z [nm]"))
        plt.ylabel(trs.choose(r"Electric Field $E_z(y=z=0)$",
                              r"Campo eléctrico $E_z(y=z=0)$"))
        
        plt.savefig(sa.file("CroppedZProfile.png"))
    
    #%% PLOT MAXIMUM INTENSIFICATION PLANE
        
    if make_plots and pm.assign(0): 
        
        fig = plt.figure()
        ax = plt.subplot()
        ax.set_aspect("equal")
        plt.title(trs.choose('Monochromatic wave {:.0f} nm on {} Sphere With {:.1f} nm Diameter',
                             'Onda monocromática {:.0f} nm sobre esfera de {} con diámetro {:.1f} nm'
                             ).format(wlen*from_um_factor*1e3, material, 2*r*from_um_factor*1e3 ))
        lims = (np.min(results_plane[...,field_peaks_single_index]), 
                np.max(results_plane[...,field_peaks_single_index]))
        lims = max([abs(l) for l in lims])
        lims = [-lims, lims]
                
        ims = ax.imshow(results_plane[...,field_peaks_single_index].T,
                        cmap='RdBu', #interpolation='spline36', 
                        vmin=lims[0], vmax=lims[1],
                        extent=[min(y_plane), max(y_plane),
                                min(z_plane), max(z_plane)])
        plt.axhline(-cell_width/2 + pml_width, color="k", linestyle="dashed", linewidth=1)
        plt.axhline(cell_width/2 - pml_width, color="k", linestyle="dashed", linewidth=1)
        plt.axvline(-cell_width/2 + pml_width, color="k", linestyle="dashed", linewidth=1)
        plt.axvline(cell_width/2 - pml_width, color="k", linestyle="dashed", linewidth=1)

        cax = ax.inset_axes([1.04, 0, 0.07, 1], #[1.04, 0.2, 0.05, 0.6], 
                            transform=ax.transAxes)
        cbar = fig.colorbar(ims, ax=ax, cax=cax)
        cbar.set_label(trs.choose("Electric field $E_z$",
                                  "Campo eléctrico $E_z$"))
            
        plt.xlabel(trs.choose("Distance Y [MPu]", "Distancia Y [uMP]"))
        plt.ylabel(trs.choose("Distance Z [MPu]", "Distancia Z [uMP]"))
        
        plt.savefig(sa.file("YZPlane.png"))
        
        fig = plt.figure()
        ax = plt.subplot()
        ax.set_aspect("equal")
        plt.title(trs.choose('Monochromatic wave {:.0f} nm on {} Sphere With {:.1f} nm Diameter',
                             'Onda monocromática {:.0f} nm sobre esfera de {} con diámetro {:.1f} nm'
                             ).format(wlen*from_um_factor*1e3, material, 2*r*from_um_factor*1e3 ))
        lims = (np.min(field_peaks_plane), np.max(field_peaks_plane))
        lims = max([abs(l) for l in lims])
        lims = [-lims, lims]
                
        ims = ax.imshow(field_peaks_plane,
                        cmap='RdBu', #interpolation='spline36', 
                        vmin=lims[0], vmax=lims[1],
                        extent=[min(y_plane_cropped) * from_um_factor * 1e3,
                                max(y_plane_cropped) * from_um_factor * 1e3,
                                min(z_plane_cropped) * from_um_factor * 1e3,
                                max(z_plane_cropped) * from_um_factor * 1e3])

        cax = ax.inset_axes([1.04, 0, 0.07, 1], #[1.04, 0.2, 0.05, 0.6], 
                            transform=ax.transAxes)
        cbar = fig.colorbar(ims, ax=ax, cax=cax)
        cbar.set_label(trs.choose("Electric field $E_z$",
                                  "Campo eléctrico $E_z$"))
            
        # plt.show()
        plt.xlabel(trs.choose("Distance Y [nm]", "Distancia Y [nm]"))
        plt.ylabel(trs.choose("Distance Z [nm]", "Distancia Z [nm]"))
        
        plt.savefig(sa.file("CroppedYZPlane.png"))
    
    #%% MAKE PLANE GIF
    
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
        ax = plt.subplot()
        ax.set_aspect('equal')
        lims = (np.min(results_plane), np.max(results_plane))
        lims = max([abs(l) for l in lims])
        lims = [-lims, lims]
        
        def make_pic_plane(k):
            ax.clear()
            ims = ax.imshow(results_plane[...,k].T,
                            cmap='RdBu', #interpolation='spline36', 
                            vmin=lims[0], vmax=lims[1],
                            extent=[min(y_plane), max(y_plane),
                                    min(z_plane), max(z_plane)])
            plt.axhline(-cell_width/2 + pml_width, color="k", linestyle="dashed", linewidth=1)
            plt.axhline(cell_width/2 - pml_width, color="k", linestyle="dashed", linewidth=1)
            plt.axvline(-cell_width/2 + pml_width, color="k", linestyle="dashed", linewidth=1)
            plt.axvline(cell_width/2 - pml_width, color="k", linestyle="dashed", linewidth=1)
            
            plt.show()
            plt.grid(False)
            plt.xlabel(trs.choose("Distance $Y$ [MPu]", "Posición $Y$ [uMP]"))
            plt.ylabel(trs.choose("Distance $Z$ [MPu]", "Posición $Z$ [uMP]"))
            
            plt.annotate(trs.choose(f"1 MPu = {from_um_factor * 1e3:.0f} nm",
                                    f"1 uMP = {from_um_factor * 1e3:.0f} nm"),
                         (55, 9), xycoords='figure points') # 50, 300
            ax.text(.98, -.115, trs.choose(f'Time: {t_line[k]/period:.1f} MPu',
                                           f'Tiempo: {t_line[k]/period:.1f} uMP'), 
                    transform=ax.transAxes)
            
            cax = ax.inset_axes([1.04, 0, 0.07, 1], #[1.04, 0.2, 0.05, 0.6], 
                                transform=ax.transAxes)
            cbar = fig.colorbar(ims, ax=ax, cax=cax)
            cbar.set_label(trs.choose("Electric Field $E_z$",
                                      "Campo eléctrico $E_z$"))
            return ax
        
        def make_gif_plane(gif_filename):
            pics = []
            for ik, k in enumerate(frames_index):
                ax = make_pic_plane(k)
                plt.savefig('temp_pic.png') 
                pics.append(mim.imread('temp_pic.png')) 
                print(str(ik+1)+'/'+str(nframes))
            mim.mimsave(gif_filename+'.gif', pics, fps=5)
            os.remove('temp_pic.png')
            print('Saved gif')
        
        make_gif_plane(sa.file("PlaneX=0"))
        plt.close(fig)
        # del fig, ax, lims, nframes_step, nframes, call_series, label_function
        
    #%% MAKE LINES GIF
    
    if make_gifs and pm.assign(0):
        
        # What should be parameters
        nframes = min(maxnframes, results_line.shape[-1])
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
        ax = plt.subplot()
        lims = (np.min(results_line), np.max(results_line))
        
        def make_pic_line(k):
            ax.clear()
            
            ax.plot(x_line, results_line[...,k], linewidth=2)
            plt.axvline(-cell_width/2 + pml_width, color="k", linestyle="dashed")
            plt.axvline(cell_width/2 - pml_width, color="k", linestyle="dashed")
            plt.axhline(0, color="k", linewidth=1)
        
            plt.xlabel(trs.choose("Position $X$ [MPu]", "Position $X$ [uMP]"))
            plt.ylabel(trs.choose(r"Electric Field $E_z(y=z=0)$",
                                  r"Campo eléctrico $E_z(y=z=0)$"))
            plt.xlim(min(x_line), max(x_line))
            plt.annotate(trs.choose(f"1 MPu = {from_um_factor * 1e3:.0f} nm",
                                    f"1 uMP = {from_um_factor * 1e3:.0f} nm"),
                         (355, 9), xycoords='figure points') # 50, 300
                
            ax.text(0, -.11, trs.choose(f'Time: {t_line[k]/period:.1f} MPu',
                                        f'Tiempo: {t_line[k]/period:.1f} uMP'), 
                    transform=ax.transAxes)
            plt.ylim(*lims)
            
            plt.show()
            return ax
        
        def make_gif_line(gif_filename):
            pics = []
            for ik, k in enumerate(frames_index):
                ax = make_pic_line(k)
                plt.savefig('temp_pic.png') 
                pics.append(mim.imread('temp_pic.png')) 
                print(str(ik+1)+'/'+str(nframes))
            mim.mimsave(gif_filename+'.gif', pics, fps=5)
            os.remove('temp_pic.png')
            print('Saved gif')
        
        make_gif_line(sa.file("AxisX=0"))
        plt.close(fig)
        # del fig, ax, lims, nframes_step, nframes, call_series, label_function
    
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
        plot_grid = gridspec.GridSpec(ncols=5, nrows=1, figure=fig, 
                                      wspace=.7)
        line_ax = fig.add_subplot(plot_grid[:,:3])
        plane_ax = fig.add_subplot(plot_grid[:,-2:])
        fig.set_size_inches([13.96,  4.8])
        
        line_ax_lims = (np.min(results_line), np.max(results_line))
        
        plane_ax.set_aspect('equal')
        plane_ax_lims = (np.min(results_plane), np.max(results_plane))
        plane_ax_lims = max([abs(l) for l in plane_ax_lims])
        plane_ax_lims = [-plane_ax_lims, plane_ax_lims]
        
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

            line_ax.text(-.07, -.11, trs.choose(f'Time: {t_line[k]/period:.1f} MPu',
                                                f'Tiempo: {t_line[k]/period:.1f} uMP'), 
                          transform=line_ax.transAxes)
            plt.annotate(trs.choose(f"1 MPu = {from_um_factor * 1e3:.0f} nm",
                                    f"1 uMP = {from_um_factor * 1e3:.0f} nm"),
                         (910, 9), xycoords='figure points')
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
                
        make_gif(sa.file("All"))
        plt.close(fig)
        # del fig, ax, lims, nframes_step, nframes, call_series, label_function
                
    #%% MAKE PROFILE GIF
    
    if make_gifs and pm.assign(0):
        
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
        ax = plt.subplot()
        lims = (np.min(zprofile_results), np.max(zprofile_results))
        
        def make_pic_line(k):
            ax.clear()
            
            ax.plot(z_plane, zprofile_results[...,k])
            plt.axvline(-cell_width/2 + pml_width, color="k", linestyle="dashed")
            plt.axvline(cell_width/2 - pml_width, color="k", linestyle="dashed")
            plt.axhline(0, color="k", linewidth=1)
        
            ax.text(-.1, -.105, trs.choose(f'Time: {t_line[k]/period:.1f} MPu',
                                           f'Tiempo: {t_line[k]/period:.1f} uMP'), 
                    transform=ax.transAxes)
            plt.xlabel(trs.choose("Position Z [MPu]", "Position Z [uMP]"))
            plt.ylabel(trs.choose(r"Electric Field $E_z(x=y=0)$",
                                  r"Campo eléctrico $E_z(x=y=0)$"))
            plt.annotate(trs.choose(f"1 MPu = {from_um_factor * 1e3:.0f} nm",
                                    f"1 uMP = {from_um_factor * 1e3:.0f} nm"),
                         (345, 11), xycoords='figure points')
            plt.ylim(*lims)
            
            plt.show()
            return ax
        
        def make_gif_line(gif_filename):
            pics = []
            for ik, k in enumerate(frames_index):
                ax = make_pic_line(k)
                plt.savefig('temp_pic.png') 
                pics.append(mim.imread('temp_pic.png')) 
                print(str(ik+1)+'/'+str(nframes))
            mim.mimsave(gif_filename+'.gif', pics, fps=5)
            os.remove('temp_pic.png')
            print('Saved gif')
        
        make_gif_line(sa.file("AxisZ"))
        plt.close(fig)
        # del fig, ax, lims, nframes_step, nframes, call_series, label_function
        
    #%% MAKE ABSOLUTE PROFILE GIF
    
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
        ax = plt.subplot()
        lims = (0, np.max(np.abs(zprofile_results)))
        
        def make_pic_line(k):
            ax.clear()
            
            ax.plot(z_plane, np.abs(zprofile_results[...,k]))
            plt.axvline(-cell_width/2 + pml_width, color="k", linestyle="dashed")
            plt.axvline(cell_width/2 - pml_width, color="k", linestyle="dashed")
            plt.axhline(0, color="k", linewidth=1)
        
            ax.text(-.1, -.105, trs.choose(f'Time: {t_line[k]/period:.1f} MPu',
                                           f'Tiempo: {t_line[k]/period:.1f} uMP'), 
                    transform=ax.transAxes)
            plt.xlabel(trs.choose("Position Z [MPu]", "Position Z [uMP]"))
            plt.ylabel(trs.choose(r"Electric Field $|E_z|(x=y=0)$",
                                  r"Campo eléctrico $|E_z|(x=y=0)$"))
            plt.annotate(trs.choose(f"1 MPu = {from_um_factor * 1e3:.0f} nm",
                                    f"1 uMP = {from_um_factor * 1e3:.0f} nm"),
                         (345, 11), xycoords='figure points')
            plt.ylim(*lims)
            
            plt.show()
            return ax
        
        def make_gif_line(gif_filename):
            pics = []
            for ik, k in enumerate(frames_index):
                ax = make_pic_line(k)
                plt.savefig('temp_pic.png') 
                pics.append(mim.imread('temp_pic.png')) 
                print(str(ik+1)+'/'+str(nframes))
            mim.mimsave(gif_filename+'.gif', pics, fps=5)
            os.remove('temp_pic.png')
            print('Saved gif')
        
        make_gif_line(sa.file("AxisZAbs"))
        plt.close(fig)
        
    #%% MAKE MEANINGFUL ALL GIF
    
    if make_gifs and pm.assign(0):
        
        # What should be parameters
        nframes = min(maxnframes, results_plane_cropped.shape[-1])
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
        plot_grid = gridspec.GridSpec(ncols=5, nrows=1, figure=fig, 
                                      wspace=.7)
        line_ax = fig.add_subplot(plot_grid[:,:3])
        plane_ax = fig.add_subplot(plot_grid[:,-2:])
        fig.set_size_inches([13.96,  4.8])
        
        # fig, [line_ax, plane_ax] = plt.subplots(ncols=2)
        # fig.set_size_inches([ 4.65, 10])
        
        line_ax_lims = (np.min(zprofile_cropped), np.max(zprofile_cropped))
        
        plane_ax.set_aspect('equal')
        plane_ax.grid(False)
        plane_ax_lims = (np.min(results_plane_cropped), np.max(results_plane_cropped))
        plane_ax_lims = max([abs(l) for l in plane_ax_lims])
        plane_ax_lims = [-plane_ax_lims, plane_ax_lims]
        
        def make_pic(k):
            line_ax.clear()
            plane_ax.clear()
            
            line_ax.plot(z_plane_cropped, zprofile_cropped[...,k], linewidth=2)
            line_ax.axvline(-r, color="dimgrey", linestyle="dashed", linewidth=1.2)
            line_ax.axvline(r, color="dimgrey", linestyle="dashed", linewidth=1.2)
            line_ax.axhline(0, color="dimgrey", linewidth=1)
            line_ax.axvline(0, color="dimgrey", linewidth=1)
        
            line_ax.set_xlabel(trs.choose("Position $Z$ [MPu]", "Posición $Z$ [uMP]"))
            line_ax.set_ylabel(trs.choose(r"Electric Field $E_z(x=y=0)$",
                                          r"Campo eléctrico $E_z(x=y=0)$"))
            line_ax.set_xlim(min(z_plane_cropped), max(z_plane_cropped))
            
            ims = plane_ax.imshow(results_plane_cropped[...,k].T,
                                  cmap='RdBu', #interpolation='spline36', 
                                  vmin=plane_ax_lims[0], vmax=plane_ax_lims[1],
                                  extent=[min(y_plane_cropped), max(y_plane_cropped),
                                          min(z_plane_cropped), max(z_plane_cropped)])
            plane_ax.axhline(-r, color="dimgrey", linestyle="dashed", linewidth=1.2)
            plane_ax.axhline(r, color="dimgrey", linestyle="dashed", linewidth=1.2)
            plane_ax.axvline(-r, color="dimgrey", linestyle="dashed", linewidth=1.2)
            plane_ax.axvline(r, color="dimgrey", linestyle="dashed", linewidth=1.2)
            plane_ax.axhline(0, color="dimgrey", linewidth=1)
            plane_ax.axvline(0, color="dimgrey", linewidth=1)
            
            plane_ax.set_xlabel(trs.choose("Position $Y$ [MPu]", "Posición $Y$ [uMP]"))
            plane_ax.set_ylabel(trs.choose("Position $Z$ [MPu]", "Posición $Z$ [uMP]"))
            plane_ax.grid(False, axis="both")

            line_ax.text(-.07, -.11, trs.choose(f'Time: {t_line[k]/period:.1f} MPu',
                                                f'Tiempo: {t_line[k]/period:.1f} uMP'), 
                          transform=line_ax.transAxes)
            plt.annotate(trs.choose(f"1 MPu = {from_um_factor * 1e3:.0f} nm",
                                    f"1 uMP = {from_um_factor * 1e3:.0f} nm"),
                         (910, 9), xycoords='figure points')
            line_ax.set_ylim(*line_ax_lims)
            
            cax = plane_ax.inset_axes([1.04, 0, 0.07, 1], #[1.04, 0.2, 0.05, 0.6], 
                                      transform=plane_ax.transAxes)
            cbar = fig.colorbar(ims, ax=plane_ax, cax=cax)
            cbar.set_label(trs.choose("Electric Field $E_z(x=0,y,z)$",
                                      "Campo eléctrico $E_z(x=0,y,z)$"))
            
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
                
        """
        from matplotlib import use as use_backend
        
        use_backend("Agg")
        for k in [490, 515, 550, 575]:
            line_ax, plane_ax = make_pic(535)
            fig.dpi = 200
            plt.savefig(sa.file(f"AdvancingFields{k}.png"))
        use_backend("Qt5Agg")
        
        """ 
           
        make_gif(sa.file("AllSignificant"))
        plt.close(fig)
        # del fig, ax, lims, nframes_step, nframes, call_series, label_function
        
    #%%
    
    f.close()
    g.close()
    try:
        h.close()
    except:
        pass