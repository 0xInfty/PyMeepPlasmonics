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

def plots_monoch_field(series, folder, units=False, hfield=False, 
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
    wlen = f["Ez"].attrs["wlen"]
    submerged_index = f["Ez"].attrs["submerged_index"]
    
    cell_width = f["Ez"].attrs["cell_width"]
    pml_width = f["Ez"].attrs["pml_width"]
    source_center = f["Ez"].attrs["source_center"]
    
    until_time = f["Ez"].attrs["until_time"]
    period_line = f["Ez"].attrs["period_line"]
    period_plane = f["Ez"].attrs["period_plane"]
    period = submerged_index * wlen

    units = f["Ez"].attrs["units"]
    
    if units:
        plot_title_base = trs.choose('Monochromatic wave ', 
                                     "Onda monocromática de ") + f' {wlen * from_um_factor * 1e3:.0f} nm'
    else:
        plot_title_base = trs.choose('Dimnesionless monochromatic wave', 
                                     "Onda monocromática adimensional")
    
    #%% GENERAL USEFUL FUNCTIONS
    
    t_line_index = lambda t0 : np.argmin(np.abs(t_line - t0))        
    x_line_index = lambda x0 : np.argmin(np.abs(x_line - x0))
    
    t_plane_index = lambda t0 : np.argmin(np.abs(t_plane - t0))
    y_plane_index = lambda y0 : np.argmin(np.abs(y_plane - y0))
    z_plane_index = lambda z0 : np.argmin(np.abs(z_plane - z0))
    
    #%%
    
    source_results = np.asarray(results_line[x_line_index(source_center), :])
    
    def crop_field_xline(field):
        cropped = field[: x_line_index(cell_width/2 - pml_width)]
        cropped = cropped[x_line_index(-cell_width/2 + pml_width) :]
        return cropped
    
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
    
    xaxis_results = np.asarray([crop_field_xline(field) for field in np.array(results_line).T]).T
       
    zprofile_results = np.asarray(results_plane[y_plane_index(0), :, :])
    
    zprofile_integral = np.array([ integrate_field_zprofile(zprof) for zprof in zprofile_results.T])
    
    #%% SHOW SOURCE AND FOURIER
    
    if make_plots and pm.assign(0): 

        plt.figure()        
        plt.title(plot_title_base)
        plt.plot(t_line, source_results)
        plt.xlabel(trs.choose("Time [Mp.u.]", "Tiempo [u.Mp.]"))
        plt.ylabel(trs.choose(r"Electric Field $E_z$ [a.u.]",
                              r"Campo eléctrico $E_z$ [u.a.]"))
        
        plt.savefig(sa.file("Source.png"))
        
        fourier = np.abs(np.fft.rfft(source_results))
        fourier_freq = np.fft.rfftfreq(len(source_results), d=period_line)
        if units:
            fourier_wlen = from_um_factor * 1e3 / fourier_freq
        else:
            fourier_wlen = 1 / fourier_freq
        fourier_max_wlen = fourier_wlen[ np.argmax(fourier) ]
        
        plt.figure()
        plt.title(plot_title_base)
        plt.plot(fourier_wlen, fourier, 'k')
        if units:
            plt.xlabel(trs.choose("Wavelength [nm]", "Longitud de onda [nm]"))
        else:
            plt.xlabel(trs.choose("Wavelength [Mp.u.]", "Longitud de onda [u.Mp.]"))
        plt.ylabel(trs.choose(r"Electric Field Fourier $\mathcal{F}\;(E_z)$ [u.a.]",
                              r"Transformada del campo eléctrico $\mathcal{F}\;(E_z)$ [u.a.]"))
        
        if units:
            plt.annotate(trs.choose(f"Maximum at {fourier_max_wlen:.2f} nm",
                                    f"Máximo en {fourier_max_wlen:.2f} nm"),
                         (5, 5), xycoords='figure points')
        else:
            plt.annotate(trs.choose(f"Maximum at {fourier_max_wlen:.2f}",
                                    f"Máximo en {fourier_max_wlen:.2f}"),
                         (5, 5), xycoords='figure points')            
        
        plt.savefig(sa.file("SourceFFT.png"))
        
        if units: plt.xlim([350, 850])
        else: plt.xlim([0, 2])
                
        plt.savefig(sa.file("SourceFFTZoom.png"))
                
    #%% SHOW X AXIS FIELD

    if make_plots and pm.assign(1):
        plt.figure()
        T, X = np.meshgrid(
            t_line,
            x_line[x_line_index(-cell_width/2 + pml_width):x_line_index(cell_width/2 - pml_width)])
        plt.contourf(T, X, xaxis_results, 100, interpolation='spline36', cmap='RdBu')
        plt.xlabel(trs.choose("Time [Mp.u.]", "Tiempo [u.Mp.]"))
        plt.ylabel(trs.choose("X Distance [Mp.u.]", "Distancia en X [u.Mp.]"))
        
        plt.savefig(sa.file("CroppedXAxis.png"))
        
        plt.figure()
        T, X = np.meshgrid(t_line, x_line)
        plt.contourf(T, X, results_line, 100, interpolation='spline36', cmap='RdBu')
        xlims = plt.xlim()
        plt.hlines(-cell_width/2 + pml_width, *xlims, color="k", linestyle="dashed")
        plt.hlines(cell_width/2 - pml_width, *xlims, color="k", linestyle="dashed")
        plt.xlim(*xlims)
        plt.xlabel(trs.choose("Time [Mp.u.]", "Tiempo [u.Mp.]"))
        plt.ylabel(trs.choose("X Distance [Mp.u.]", "Distancia en X [u.Mp.´]"))
        
        plt.savefig(sa.file("XAxis.png"))
    
    
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
            if units:
                plt.annotate(trs.choose(f"1 Meep Unit = {from_um_factor * 1e3:.0f} nm",
                                        f"1 Unidad de Meep = {from_um_factor * 1e3:.0f} nm"),
                             (5, 300), xycoords='figure points')
            else:
                plt.annotate(trs.choose(r"1 Meep Unit = $\lambda$",
                                        r"1 Unidad de Meep = $\lambda$"),
                             (50, 310), xycoords='figure points')
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
    
    #%% MAKE LINES GIF
    
    if make_gifs and pm.assign(0):
        
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
            plt.vlines(-cell_width/2 + pml_width, *lims,
                        linestyle=":", color='k')
            plt.vlines(cell_width/2 - pml_width, *lims,
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