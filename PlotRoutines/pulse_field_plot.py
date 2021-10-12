#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis of single run of planewave pulse of Gaussian frequency profile.

See also
--------
Routines/pulse_field
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
series = "TestPulseField"
folder = "Test"

hfield = False

make_plots = True
make_gifs = False

english = False
maxnframes = 300
"""

#%% 

def plots_pulse_field(series, folder, units=False, hfield=False, 
                      make_plots=True, make_gifs=False, 
                      english=False, maxnframes=300):
        
    #%% SETUP
    
    # Computation
    pm = vm.ParallelManager()
    n_processes, n_cores, n_nodes = pm.specs
    parallel = pm.parallel
    
    # Saving directories
    sa = vm.SavingAssistant(series, folder)
    
    trs = vu.BilingualManager(english=english)
    
    #%% EXIT IF NOTHING TO BE DONE
    
    if not make_plots and not make_gifs:
        return
    
    #%% GET READY TO LOAD DATA
        
    f = pm.hdf_file(sa.file("Field-Lines.h5"), "r+")
    results_line = f["Ez"]
    t_line = np.array(f["T"])
    x_line = np.array(f["X"])
    
    flux_wlens, flux_intensity = np.loadtxt(sa.file("Results.txt"), unpack=True)
    
    params = dict(f["Ez"].attrs)
    
    from_um_factor = params["from_um_factor"]
    wlen_range = params["wlen_range"]
    submerged_index = params["submerged_index"]
    
    cell_width = params["cell_width"]
    pml_width = params["pml_width"]
    source_center = params["source_center"]
    
    until_after_sources = params["until_after_sources"]
    period_line = params["period_line"]
    # period = submerged_index * wlen

    units = params["units"]
    
    #%% PLOT CONFIGURATION
    
    if units:
        plot_title_base = trs.choose('Planewave pulse with ', 
                                     "Pulso de frentes de onda plano con ")
        plot_title_base += r"$\lambda_0$ = "
        plot_title_base += f'{np.mean(wlen_range) * from_um_factor * 1e3:.0f} nm, '
        plot_title_base += r"$\Delta\lambda$ = "
        plot_title_base += f'{float(np.diff(wlen_range)) * from_um_factor * 1e3:.0f} nm '
    else:
        plot_title_base = trs.choose('Planewave pulse with ', 
                                     "Pulso de frentes de onda plano con ")
        plot_title_base += r"$\Delta\lambda$ = "
        plot_title_base += f'{float(np.diff(wlen_range)) * from_um_factor * 1e3:.2f} '
        plot_title_base += trs.choose("MPu", "uMP")
    
    #%% POSITION RECONSTRUCTION
    
    t_line_index = vma.def_index_function(t_line)
    x_line_index = vma.def_index_function(x_line)
    
    x_line_cropped = x_line[:x_line_index(cell_width/2 - pml_width)+1]
    x_line_cropped = x_line_cropped[x_line_index(-cell_width/2 + pml_width):]

    #%% DATA EXTRACTION
    
    source_results = vma.get_source_from_line(results_line, x_line_index, source_center)
    
    center_results = results_line[x_line_index(0),:]
        
    results_cropped_line = vma.crop_field_xprofile(results_line, x_line_index,
                                                   cell_width, pml_width)
    
    #%% SHOW SOURCE AND FOURIER
    
    if make_plots and pm.assign(0): 

        plt.figure()        
        plt.title(plot_title_base)
        plt.plot(t_line, source_results)
        plt.xlabel(trs.choose("Time [MPu]", "Tiempo [uMP]"))
        plt.ylabel(trs.choose(r"Electric Field $E_z(y=z=0)$ [au]", 
                              r"Campo eléctrico $E_z(y=z=0)$ [ua]"))
        
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
        plt.plot(fourier_wlen, fourier)
        if units:
            plt.xlabel(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
        else:
            plt.xlabel(trs.choose("Wavelength [MPu]", "Longitud de onda [uMP]"))

        plt.ylabel(trs.choose(r"Electric Field Fourier $\mathcal{F}\;(E_z)$ [ua]",
                              r"Transformada del campo eléctrico $\mathcal{F}\;(E_z)$ [ua]"))
        
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
               
    #%% SHOW FLUX
    
    if make_plots and pm.assign(1):
        
        plt.figure()
        plt.plot(flux_wlens, flux_intensity)
        
        if units:
            plt.xlabel(trs.choose(r"Wavelength $\lambda$ [nm]", r"Longitud de onda $\lambda$ [nm]"))
        else:
            plt.xlabel(trs.choose("Wavelength [MPu]", "Longitud de onda [uMP]"))
        plt.ylabel(trs.choose(r"Electromagnetic Flux $\frac{1}{A} \int_A \langle\,S_x\,\rangle \,dx\,dy$ [au]",
                              r"Flujo electromagnético $\frac{1}{A} \int_A \langle\,S_x\,\rangle \,dx\,dy$ [ua]"))
        
        if units:
            plt.annotate(trs.choose(f"Maximum at {flux_wlens[np.argmax(flux_intensity)]:.2f} nm",
                                    f"Máximo en {flux_wlens[np.argmax(flux_intensity)]:.2f} nm"),
                         (5, 5), xycoords='figure points')
        else:
            plt.annotate(trs.choose(f"Maximum at {flux_wlens[np.argmax(flux_intensity)]:.2f} MPu",
                                    f"Máximo en {flux_wlens[np.argmax(flux_intensity)]:.2f} uMP"),
                         (5, 5), xycoords='figure points') 
                
        plt.savefig(sa.file("Flux.png"))
        
    #%% SHOW FIELD AT FLUX WALL
    
    if make_plots and pm.assign(0):
        
        plt.figure()
        plt.plot(t_line, source_results, color="C0", alpha=0.4, linewidth=2,
                 label=trs.choose("Source", "Fuente"))
        plt.plot(t_line, center_results, color="C0", linewidth=1.5,
                 label=trs.choose("Flux wall", "Pared de flujo"))
        plt.xlabel(trs.choose("Time [MPu]", "Tiempo [uMP]"))
        plt.ylabel(trs.choose(r"Electric Field $E_z(y=z=0)$ [au]", 
                              r"Campo eléctrico $E_z(y=z=0)$ [ua]"))
        plt.legend()
        
        plt.savefig(sa.file("FluxWallField.png"))        
            
    #%% SHOW X AXIS FIELD

    if make_plots and pm.assign(1):
        
        plt.figure()
        T, X = np.meshgrid(t_line, x_line_cropped)
        plt.contourf(T, X, results_cropped_line, 100, cmap='RdBu')
        plt.xlabel(trs.choose("Time [MPu]", "Tiempo [uMP]"))
        plt.ylabel(trs.choose("Position $X$ [MPu]", "Posición  $X$ [uMP]"))
        plt.grid(False)
        
        plt.savefig(sa.file("CroppedXAxis.png"))
        
        plt.figure()
        T, X = np.meshgrid(t_line, x_line)
        plt.contourf(T, X, results_line, 100, cmap='RdBu')
        xlims = plt.xlim()
        plt.hlines(-cell_width/2 + pml_width, *xlims, color="k", linestyle="dashed")
        plt.hlines(cell_width/2 - pml_width, *xlims, color="k", linestyle="dashed")
        plt.xlim(*xlims)
        plt.xlabel(trs.choose("Time [MPu]", "Tiempo [uMP]"))
        plt.ylabel(trs.choose("Position $X$ [MPu]", "Posición  $X$ [uMP]"))
        plt.grid(False)
        
        plt.savefig(sa.file("XAxis.png"))
        
    #%% MAKE LINES GIF
    
    if make_gifs and pm.assign(0):
        
        # What should be parameters
        nframes = min(maxnframes, results_line.shape[-1])                   
        frames_index = np.linspace(0, results_line.shape[-1]-1, nframes)
        frames_index = [int(round(i)) for i in frames_index]
        
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
            if units:
                plt.annotate(trs.choose(f"1 MPu = {from_um_factor * 1e3:.0f} nm",
                                        f"1 uMP = {from_um_factor * 1e3:.0f} nm"),
                             (355, 9), xycoords='figure points') # 50, 300
            else:
                plt.annotate(trs.choose(r"1 MPu = $\lambda_0$",
                                        r"1 uMP = $\lambda_0$"),
                             (355, 9), xycoords='figure points') # 50, 310
            ax.text(0, -.11, trs.choose(f'Time: {t_line[k]:.1f} MPu',
                                        f'Tiempo: {t_line[k]:.1f} uMP'), 
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
        
    #%%
    
    f.close()