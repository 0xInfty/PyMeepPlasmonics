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
import numpy as np
import os
import v_meep as vm
import v_meep_analysis as vma
import v_utilities as vu

#%% PARAMETERS

"""
series = "Norm532Res3"
folder = "Field/NPMonoch/AuSphere/VacWatField/Vacuum"

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
    pm = vm.ParallelManager()
    n_processes, n_cores, n_nodes = pm.specs
    parallel = pm.parallel
    
    # Saving directories
    sa = vm.SavingAssistant(series, folder)
    home = sa.home
    sysname = sa.sysname
    path = sa.path
    
    trs = vu.BilingualManager(english=english)
    
    #%%
    
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
    
    #%%
        
    source_results = vma.get_source_from_line(results_line_norm, 
                                              x_line_norm_index, 
                                              source_center)
        
    if not requires_normalization:
        
        period, amplitude = norm_period, norm_amplitude
        
    else:          
        
        period = vma.get_period_from_source(source_results, t_line)
        amplitude = vma.get_amplitude_from_source(source_results)
        
        results_plane = np.asarray(results_plane) / amplitude
        results_line = np.asarray(results_line) / amplitude
    
    sim_source_results = vma.get_source_from_line(results_line, 
                                                  x_line_index, 
                                                  source_center)

    zprofile_results = vma.get_zprofile_from_plane(results_plane, y_plane_index)
    
    zprofile_integral = vma.integrate_field_zprofile(zprofile_results, z_plane_index,
                                                     cell_width, pml_width, 
                                                     period_plane)
    
    zprofile_max = vma.find_zpeaks_zprofile(zprofile_results, z_plane_index,
                                            cell_width, pml_width)[-1]
    
    field_peaks_all_index = vma.get_all_field_peaks_from_yzplanes(results_plane,
                                                                  y_plane_index, 
                                                                  z_plane_index, 
                                                                  cell_width, 
                                                                  pml_width)[0]
    
    field_peaks_results = vma.get_single_field_peak_from_yzplanes(results_plane,
                                                                  y_plane_index, 
                                                                  z_plane_index, 
                                                                  cell_width, 
                                                                  pml_width)
    field_peaks_single_index = field_peaks_results[0]
    field_peaks_zprofile = field_peaks_results[2]
    field_peaks_plane = field_peaks_results[3]
    del field_peaks_results
    
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
        plt.xlabel(trs.choose("Wavelength [nm]", "Longitud de onda [nm]"))
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
        plt.xlabel(trs.choose("Wavelength [nm]", "Longitud de onda [nm]"))
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
        plt.plot(t_plane, zprofile_integral)
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
        plt.ylabel(trs.choose(r"Normalized Electric Field Maximum $max[ E_z(z) ]$",
                              r"Máximo del campo eléctrico normalizado $max[ E_z(z) ]$"))
        
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
        plt.ylabel(trs.choose(r"Normalized Electric Field $E_z(y=z=0)$",
                              r"Campo eléctrico normalizado $E_z(y=z=0)$"))
        
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
        plt.ylabel(trs.choose(r"Normalized Electric Field $E_z(y=z=0)$",
                              r"Campo eléctrico normalizado $E_z(y=z=0)$"))
        
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
        cbar.set_label(trs.choose("Normalized electric field $E_z$",
                                  "Campo eléctrico normalizado $E_z$"))
            
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
        cbar.set_label(trs.choose("Normalized electric field $E_z$",
                                  "Campo eléctrico normalizado $E_z$"))
            
        # plt.show()
        plt.xlabel(trs.choose("Distance Y [nm]", "Distancia Y [nm]"))
        plt.ylabel(trs.choose("Distance Z [nm]", "Distancia Z [nm]"))
        
        plt.savefig(sa.file("CroppedYZPlane.png"))
    
    #%% MAKE PLANE GIF
    
    if make_gifs and pm.assign(1):
        
        # What should be parameters
        nframes = min(maxnframes, results_plane.shape[-1])
        nframes_step = int(results_plane.shape[-1] / nframes)
        call_series = lambda i : results_plane[:,:,i]
        label_function = lambda i : trs.choose('Tiempo: {:.1f} uMP',
                                               'Time: {:.1f} MPu').format(i*period_plane)
        
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
            
            ax.text(-.1, -.105, label_function(k), transform=ax.transAxes)
            plt.show()
            plt.xlabel(trs.choose("Distance Y [MPu]", "Distancia Y [uMP]"))
            plt.ylabel(trs.choose("Distance Z [MPu]", "Distancia Z [uMP]"))
            plt.annotate(trs.choose(f"1 Meep Unit = {from_um_factor * 1e3:.0f} nm",
                                    f"1 Unidad de Meep = {from_um_factor * 1e3:.0f} nm"),
                         (300, 11), xycoords='figure points')
            
            cax = ax.inset_axes([1.04, 0, 0.07, 1], #[1.04, 0.2, 0.05, 0.6], 
                                transform=ax.transAxes)
            cbar = fig.colorbar(ims, ax=ax, cax=cax)
            cbar.set_label(trs.choose("Normalized electric field $E_z$",
                                      "Campo eléctrico normalizado $E_z$"))
            return ax
        
        def make_gif_plane(gif_filename):
            pics = []
            for i in range(nframes):
                ax = make_pic_plane(i*nframes_step)
                plt.savefig('temp_pic.png') 
                pics.append(mim.imread('temp_pic.png')) 
                print(str(i+1)+'/'+str(nframes))
            mim.mimsave(gif_filename+'.gif', pics, fps=5)
            os.remove('temp_pic.png')
            print('Saved gif')
        
        make_gif_plane(sa.file("PlaneX=0"))
        plt.close(fig)
        # del fig, ax, lims, nframes_step, nframes, call_series, label_function
            
    #%% MAKE PROFILE GIF
    
    if make_gifs and pm.assign(0):
        
        # What should be parameters
        nframes = min(maxnframes, results_plane.shape[-1])
        nframes_step = int(results_plane.shape[-1] / nframes)
        call_series = lambda i : results_plane[:,:,i]
        label_function = lambda i : trs.choose('Tiempo: {:.1f} uMP',
                                               'Time: {:.1f} MPu').format(i*period_plane)
        
        # Animation base
        fig = plt.figure()
        ax = plt.subplot()
        lims = (np.min(zprofile_results), np.max(zprofile_results))
        
        def make_pic_line(k):
            ax.clear()
            
            ax.plot(z_plane, zprofile_results[...,k])
            plt.axvline(-cell_width/2 + pml_width, color="k", linestyle="dashed")
            plt.axvline(cell_width/2 - pml_width, color="k", linestyle="dashed")
            plt.axvline(source_center, color="r", linestyle="dashed")
            plt.axhline(0, color="k", linewidth=1)
        
            ax.text(-.1, -.105, label_function(k), transform=ax.transAxes)
            plt.xlabel(trs.choose("Position Z [MPu]", "Position Z [uMP]"))
            plt.ylabel(trs.choose(r"Normalized Electric Field $E_z(x=y=0)$",
                                  r"Campo eléctrico normalizado $E_z(x=y=0)$"))
            plt.annotate(trs.choose(f"1 Meep Unit = {from_um_factor * 1e3:.0f} nm",
                                    f"1 Unidad de Meep = {from_um_factor * 1e3:.0f} nm"),
                         (300, 11), xycoords='figure points')
            plt.ylim(*lims)
            
            plt.show()
            return ax
        
        def make_gif_line(gif_filename):
            pics = []
            for i in range(nframes):
                ax = make_pic_line(i*nframes_step)
                plt.savefig('temp_pic.png') 
                pics.append(mim.imread('temp_pic.png')) 
                print(str(i+1)+'/'+str(nframes))
            mim.mimsave(gif_filename+'.gif', pics, fps=5)
            os.remove('temp_pic.png')
            print('Saved gif')
        
        make_gif_line(sa.file("AxisZ=0"))
        plt.close(fig)
        # del fig, ax, lims, nframes_step, nframes, call_series, label_function
        
    #%% MAKE LINES GIF
    
    if make_gifs and pm.assign(1):
        
        # What should be parameters
        nframes = min(maxnframes, results_plane.shape[-1])
        nframes_step = int(results_plane.shape[-1] / nframes)
        call_series = lambda i : results_plane[:,:,i]
        label_function = lambda i : trs.choose('Tiempo: {:.1f} uMP',
                                               'Time: {:.1f} MPu').format(i*period_plane)
        
        # Animation base
        fig = plt.figure()
        ax = plt.subplot()
        lims = (np.min(results_line), np.max(results_line))
        
        def make_pic_line(k):
            ax.clear()
            
            ax.plot(x_line, results_line[...,k])
            plt.axvline(-cell_width/2 + pml_width, color="k", linestyle="dashed")
            plt.axvline(cell_width/2 - pml_width, color="k", linestyle="dashed")
            plt.axhline(0, color="k", linewidth=1)
        
            ax.text(-.1, -.105, label_function(k), transform=ax.transAxes)
            plt.xlabel(trs.choose("Position X [MPu]", "Position X [uMP]"))
            plt.ylabel(trs.choose(r"Normalized Electric Field $E_z(y=z=0)$",
                                  r"Campo eléctrico normalizado $E_z(y=z=0)$"))
            plt.annotate(trs.choose(f"1 Meep Unit = {from_um_factor * 1e3:.0f} nm",
                                    f"1 Unidad de Meep = {from_um_factor * 1e3:.0f} nm"),
                         (300, 11), xycoords='figure points')
            plt.ylim(*lims)
            
            plt.show()
            return ax
        
        def make_gif_line(gif_filename):
            pics = []
            for i in range(nframes):
                ax = make_pic_line(i*nframes_step)
                plt.savefig('temp_pic.png') 
                pics.append(mim.imread('temp_pic.png')) 
                print(str(i+1)+'/'+str(nframes))
            mim.mimsave(gif_filename+'.gif', pics, fps=5)
            os.remove('temp_pic.png')
            print('Saved gif')
        
        make_gif_line(sa.file("AxisX=0"))
        plt.close(fig)
        # del fig, ax, lims, nframes_step, nframes, call_series, label_function
    
    #%%
    
    f.close()
    g.close()
    try:
        h.close()
    except:
        pass