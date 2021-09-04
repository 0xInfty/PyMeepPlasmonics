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
import meep as mp
import numpy as np
import os
from scipy.signal import find_peaks
import v_meep as vm
import v_save as vs
import v_utilities as vu

#%% PARAMETERS

"""
series = "Periods532"
folder = "Field/NPMonoch/TestPeriods/Water"

hfield = False

make_plots = True
make_gifs = False

english = False
"""

#%% 

def plots_monoch_field(series, folder, hfield=False, 
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
    
    from_um_factor = f["Ez"].attrs["from_um_factor"]
    
    r = f["Ez"].attrs["r"]
    material = f["Ez"].attrs["material"]
    
    wlen = f["Ez"].attrs["wlen"]
    
    cell_width = f["Ez"].attrs["cell_width"]
    pml_width = f["Ez"].attrs["pml_width"]
    source_center = f["Ez"].attrs["source_center"]
    
    until_time = f["Ez"].attrs["until_time"]
    period_line = f["Ez"].attrs["period_line"]
    period_plane = f["Ez"].attrs["period_plane"]
    
    #%% GENERAL USEFUL FUNCTIONS
    
    t_line_index = lambda t0 : np.argmin(np.abs(t_line - t0))        
    x_line_index = lambda x0 : np.argmin(np.abs(x_line - x0))
    
    t_plane_index = lambda t0 : np.argmin(np.abs(t_plane - t0))
    y_plane_index = lambda y0 : np.argmin(np.abs(y_plane - y0))
    z_plane_index = lambda z0 : np.argmin(np.abs(z_plane - z0))
    
    #%%
    
    source_results = np.asarray(results_line[x_line_index(source_center), :])
    
    def crop_field_zyplane(field):
        cropped = field[: y_plane_index(cell_width/2 - pml_width), :]
        cropped = cropped[y_plane_index(-cell_width/2 + pml_width) :, :]
        cropped = cropped[:,: z_plane_index(cell_width/2 - pml_width)]
        cropped = cropped[:, z_plane_index(-cell_width/2 + pml_width) :]
        return cropped        
        
    def crop_field_zprofile(field):
        cropped = field[: z_plane_index(cell_width/2 - pml_width)]
        cropped = cropped[z_plane_index(-cell_width/2 + pml_width) :]
        return cropped
    
    def integrate_field_zprofile(field_profile):
        integral = np.sum(crop_field_zprofile(field_profile)) * period_plane
        return integral
    
    def detect_sign_field_zprofile(field_profile):
        return -np.sign(field_profile[0])
    
    def find_peaks_field_zprofile(field_profile):
        sign = detect_sign_field_zprofile(field_profile)
        peaks = find_peaks(sign * crop_field_zprofile(field_profile))[0]
        peaks = np.array(peaks) + z_plane_index(-cell_width/2 + pml_width)
        if len(peaks) > 2: peaks = peaks[[0,-1]]
        try:
            peaks[0] = min(peaks[0], z_plane_index(-r) - 1)
            peaks[1] = max(peaks[1], z_plane_index(r))
            return peaks
        except:
            return (None, None)
    
    def get_max_index_zprofile(field_profile):
        try: 
            return min(find_peaks_field_zprofile(field_profile)[0],
                       z_plane_index(-r))
        except IndexError:
            return None
    
    def get_max_field_zprofile(field_profile):
        try: 
            return np.mean( field_profile[ find_peaks_field_zprofile(field_profile) ] )
        except RuntimeWarning:
            return None
    
    def get_max_resonance_index(zprofile_results):
        z_resonance_max = np.array([ get_max_field_zprofile(zprof) for zprof in zprofile_results.T])
        first_index = int(np.argwhere( np.isnan(z_resonance_max) == False )[0])
        t_resonance_max_index = find_peaks(z_resonance_max[first_index:], 
                                           height=(np.max(z_resonance_max[first_index:])/2, None))[0]
        t_resonance_max_index = np.array(t_resonance_max_index) + first_index
        # t_resonance_max = t_plane[t_resonance_max_index]
        # z_resonance_max = z_resonance_max[t_resonance_max_index]
        return t_resonance_max_index #, t_resonance_max, z_resonance_max
        
    zprofile_results = np.asarray(results_plane[y_plane_index(0), :, :])
    
    zprofile_integral = np.array([ integrate_field_zprofile(zprof) for zprof in zprofile_results.T])
    
    zprofile_max = np.array([ get_max_field_zprofile(zprof) for zprof in zprofile_results.T])
    
    resonance_max_index = get_max_resonance_index(zprofile_results)
    
    #%% SHOW SOURCE AND FOURIER
    
    if make_plots and pm.assign(0): 

        plt.figure()        
        plt.title(trs.choose('Monochromatic wave {:.0f} nm on {} Sphere With {:.1f} nm Diameter',
                             'Onda monocromática {:.0f} nm sobre esfera de {} con diámetro {:.1f} nm'
                             ).format(wlen*from_um_factor*1e3, material, 2*r*from_um_factor*1e3 ))
        plt.plot(t_line, source_results)
        plt.xlabel(trs.choose("Time [Mp.u.]", "Tiempo [u.Mp.]"))
        plt.ylabel(trs.choose(r"Electric Field $E_z$ [a.u.]",
                              r"Campo eléctrico $E_z$ [u.a.]"))
        
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
        
    #%% SHOW PROFILE AND RESONANCE
    
    if make_plots and pm.assign(0): 

        plt.figure()        
        plt.title(trs.choose('Monochromatic wave {:.0f} nm on {} Sphere With {:.1f} nm Diameter',
                             'Onda monocromática {:.0f} nm sobre esfera de {} con diámetro {:.1f} nm'
                             ).format(wlen*from_um_factor*1e3, material, 2*r*from_um_factor*1e3 ))
        plt.plot(t_plane, zprofile_integral)
        plt.xlabel(trs.choose("Time [Mp.u.]", "Tiempo [u.Mp.]"))
        plt.ylabel(trs.choose(r"Electric Field Integral $\int E_z(z) \; dz$ [a.u.]",
                              r"Integral del campo eléctrico $\int E_z(z) \; dz$ [u.a.]"))
        
        plt.savefig(sa.file("Integral.png"))
        
        plt.figure()
        plt.title(trs.choose('Monochromatic wave {:.0f} nm on {} Sphere With {:.1f} nm Diameter',
                             'Onda monocromática {:.0f} nm sobre esfera de {} con diámetro {:.1f} nm'
                             ).format(wlen*from_um_factor*1e3, material, 2*r*from_um_factor*1e3 ))
        plt.plot(t_plane, zprofile_max)
        plt.xlabel(trs.choose("Time [Mp.u.]", "Tiempo [u.Mp.]"))
        plt.ylabel(trs.choose(r"Electric Field Maximum $max[ E_z(z) ]$ [a.u.]",
                              r"Máximo del campo eléctrico $max[ E_z(z) ]$ [u.a.]"))
        
        plt.savefig(sa.file("Maximum.png"))
        
        plt.figure()        
        plt.suptitle(trs.choose('Monochromatic wave {:.0f} nm on {} Sphere With {:.1f} nm Diameter',
                                'Onda monocromática {:.0f} nm sobre esfera de {} con diámetro {:.1f} nm'
                                ).format(wlen*from_um_factor*1e3, material, 2*r*from_um_factor*1e3 ))
        ax = plt.subplot()
        ax2 = plt.twinx()
        l2, = ax.plot(t_plane, zprofile_integral / np.max(np.abs(zprofile_integral)), 
                      "C0", linewidth=2, alpha=0.5, zorder=0)
        l, = ax.plot(t_line, source_results  / np.max(np.abs(source_results)), 
                     "k", linewidth=1)
        ax.set_xlabel(trs.choose("Time [Mp.u.]", "Tiempo [u.Mp.]"))
        ax.set_ylabel(trs.choose(r"Normalized Electric Field $E_z$ [a.u.]",
                                 r"Campo eléctrico normalizado $E_z$ [u.a.]"))
        ax2.set_ylabel(trs.choose(r"Normalized Electric Field Integral $\int E_z(z) \; dz$ [a.u.]",
                                  r"Integral normalizada del campo eléctrico $\int E_z(z) \; dz$ [u.a.]"))
        plt.legend([l, l2], ["Source", "Resonance"])
        
        plt.savefig(sa.file("IntegralSource.png"))
        
        plt.figure()        
        plt.suptitle(trs.choose('Monochromatic wave {:.0f} nm on {} Sphere With {:.1f} nm Diameter',
                                'Onda monocromática {:.0f} nm sobre esfera de {} con diámetro {:.1f} nm'
                                ).format(wlen*from_um_factor*1e3, material, 2*r*from_um_factor*1e3 ))
        ax = plt.subplot()
        ax2 = plt.twinx()
        l2, = ax.plot(t_plane, zprofile_max, 
                      "C0", linewidth=2, alpha=0.5, zorder=0)
        l, = ax.plot(t_line, source_results, "k", linewidth=1)
        ax.set_xlabel(trs.choose("Time [Mp.u.]", "Tiempo [u.Mp.]"))
        ax.set_ylabel(trs.choose(r"Electric Field $E_z$ [a.u.]",
                                 r"Campo eléctrico $E_z$ [u.a.]"))
        ax2.set_ylabel(trs.choose(r"Electric Field Maximum $max[ E_z(z) ]$ [a.u.]",
                                  r"Máximo del campo eléctrico $max[ E_z(z) ]$ [u.a.]"))
        plt.legend([l, l2], ["Source", "Resonance"])
        
        plt.savefig(sa.file("MaximumSource.png"))
        
        def draw_pml_box():
            plt.hlines(-cell_width/2 + pml_width, 
                       -cell_width/2 + pml_width, 
                       cell_width/2 - pml_width,
                       linestyle=":", color='k')
            plt.hlines(cell_width/2 - pml_width, 
                       -cell_width/2 + pml_width, 
                       cell_width/2 - pml_width,
                       linestyle=":", color='k')
            plt.vlines(-cell_width/2 + pml_width, 
                       -cell_width/2 + pml_width, 
                       cell_width/2 - pml_width,
                       linestyle=":", color='k')
            plt.vlines(cell_width/2 - pml_width, 
                       -cell_width/2 + pml_width, 
                       cell_width/2 - pml_width,
                       linestyle=":", color='k')
        
        Y, Z = np.meshgrid(y_plane, z_plane)
        lims = (np.min(results_plane), np.max(results_plane))
        
        plt.figure()
        ax = plt.subplot()
        ax.set_aspect("equal")
        plt.title(trs.choose('Monochromatic wave {:.0f} nm on {} Sphere With {:.1f} nm Diameter',
                             'Onda monocromática {:.0f} nm sobre esfera de {} con diámetro {:.1f} nm'
                             ).format(wlen*from_um_factor*1e3, material, 2*r*from_um_factor*1e3 ))
        ax.pcolormesh(Y, Z, results_plane[:,:,resonance_max_index[-2]].T, 
                      cmap='bwr', shading='gouraud', vmin=lims[0], vmax=lims[1])
        draw_pml_box()
        plt.show()
        plt.xlabel(trs.choose("Distance Y [Mp.u.]", "Distancia Y [u.Mp.]"))
        plt.ylabel(trs.choose("Distance Z [Mp.u.]", "Distancia Z [u.Mp.]"))
        
        plt.savefig(sa.file("Resonance.png"))
    
    #%% MAKE PLANE GIF
    
    if make_gifs and pm.assign(1):
        
        # What should be parameters
        nframes = min(maxnframes, results_plane.shape[-1])
        nframes_step = int(results_plane.shape[-1] / nframes)
        call_series = lambda i : results_plane[:,:,i]
        label_function = lambda i : trs.choose('Tiempo: {:.1f} u.Mp.',
                                               'Time: {:.1f} Mp.u.').format(i*period_plane)
        Y, Z = np.meshgrid(y_plane, z_plane)
        
        # Animation base
        fig = plt.figure()
        ax = plt.subplot()
        ax.set_aspect('equal')
        lims = (np.min(results_plane), np.max(results_plane))
        
        def draw_pml_box():
            plt.hlines(-cell_width/2 + pml_width, 
                       -cell_width/2 + pml_width, 
                       cell_width/2 - pml_width,
                       linestyle=":", color='k')
            plt.hlines(cell_width/2 - pml_width, 
                       -cell_width/2 + pml_width, 
                       cell_width/2 - pml_width,
                       linestyle=":", color='k')
            plt.vlines(-cell_width/2 + pml_width, 
                       -cell_width/2 + pml_width, 
                       cell_width/2 - pml_width,
                       linestyle=":", color='k')
            plt.vlines(cell_width/2 - pml_width, 
                       -cell_width/2 + pml_width, 
                       cell_width/2 - pml_width,
                       linestyle=":", color='k')
        
        def make_pic_plane(i):
            ax.clear()
            ax.pcolormesh(Y, Z, call_series(i).T, cmap='bwr', shading='gouraud',
                          vmin=lims[0], vmax=lims[1])
            ax.text(-.1, -.105, label_function(i), transform=ax.transAxes)
            draw_pml_box()
            plt.show()
            plt.xlabel(trs.choose("Distance Y [Mp.u.]", "Distancia Y [u.Mp.]"))
            plt.ylabel(trs.choose("Distance Z [Mp.u.]", "Distancia Z [u.Mp.]"))
            plt.annotate(trs.choose(f"1 Meep Unit = {from_um_factor * 1e3:.0f} nm",
                                    f"1 Unidad de Meep = {from_um_factor * 1e3:.0f} nm"),
                         (5, 5), xycoords='figure points')
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
    
    ### HASTA ACÁ LLEGUÉ
        
    #%% MAKE PROFILE GIF
    
    if make_gifs and pm.assign(0):
        
        # What should be parameters
        nframes = min(maxnframes, zprofile_results.shape[-1])
        nframes_step = int(zprofile_results.shape[-1] / nframes)
        # nframes = int(zprofile_results.shape[-1]/nframes_step)
        call_series = lambda i : zprofile_results[:, i]
        label_function = lambda i : trs.choose('Tiempo: {:.1f} u.Mp.',
                                               'Time: {:.1f} Mp.u.').format(i*period_plane)
        
        # Animation base
        fig = plt.figure()
        ax = plt.subplot()
        lims = (np.min(zprofile_results), np.max(zprofile_results))
        shape = call_series(0).shape[0]
        
        def draw_pml_box():
            plt.vlines(-cell_width/2 + pml_width, *lims,
                        linestyle=":", color='k')
            plt.vlines(cell_width/2 - pml_width, *lims,
                        linestyle=":", color='k')
            plt.hlines(0, min(z_plane), max(z_plane),
                       color='k', linewidth=1)
        
        def make_pic_line(i):
            ax.clear()
            plt.plot(z_plane, call_series(i))
            ax.set_ylim(*lims)
            ax.text(-.12, -.1, label_function(i), transform=ax.transAxes)
            draw_pml_box()
            plt.xlabel(trs.choose("Distance Z [Mp.u.]", "Distancia Z (u.Mp.)"))
            plt.ylabel(trs.choose(r"Electric Field Profile $E_z|_{z=0}$ [a.u.]",
                                  r"Perfil del campo eléctrico $E_z|_{z=0}$ [u.a.]"))
            plt.xlim(min(z_plane), max(z_plane))
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
        
        make_gif_line(sa.file("AxisZ"))
        plt.close(fig)
        # del fig, ax, lims, nframes_step, nframes, call_series, label_function
        
    #%% MAKE LINES GIF
    
    if make_gifs and pm.assign(1):
        
        # What should be parameters
        nframes = min(maxnframes, results_line.shape[-1])
        nframes_step = int(results_line.shape[-1] / nframes)
        call_series = lambda i : results_line[:,i]
        label_function = lambda i : trs.choose('Tiempo: {:.1f} u.Mp.',
                                               'Time: {:.1f} Mp.u.').format(i*period_plane)
                
        # Animation base
        fig = plt.figure()
        ax = plt.subplot()
        lims = (np.min(results_line), np.max(results_line))
        shape = call_series(0).shape[0]
        
        def draw_pml_box():
            plt.vlines(-cell_width/2 + pml_width, 
                        -cell_width/2 + pml_width, 
                        cell_width/2 - pml_width,
                        linestyle=":", color='k')
            plt.vlines(cell_width/2 - pml_width, 
                        -cell_width/2 + pml_width, 
                        cell_width/2 - pml_width,
                        linestyle=":", color='k')
            plt.hlines(0, min(x_line), max(x_line),
                       color='k', linewidth=1)
        
        def make_pic_line(i):
            ax.clear()
            plt.plot(x_line, call_series(i))
            ax.set_ylim(*lims)
            ax.text(-.12, -.1, label_function(i), transform=ax.transAxes)
            draw_pml_box()
            plt.xlabel(trs.choose("Distance X [Mp.u.]", "Distancia X (u.Mp.)"))
            plt.ylabel(trs.choose(r"Electric Field Profile $E_z|_{x=0}$ [a.u.]",
                                  r"Perfil del campo eléctrico $E_z|_{x=0}$ [u.a.]"))
            plt.xlim(min(x_line), max(x_line))
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
        
        make_gif_line(sa.file("AxisX"))
        plt.close(fig)
        # del fig, ax, lims, nframes_step, nframes, call_series, label_function
    
    #%%
    
    f.close()
    g.close()