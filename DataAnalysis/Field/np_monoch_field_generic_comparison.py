#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:45:00 2020

@author: vall
"""

# Field of 120nm-diameter Au sphere given a visible monochromatic incident wave.

from socket import gethostname
if "Nano" in gethostname():
    syshome = "/home/nanofisica/Documents/Vale/ThesisPython"
elif "vall" in gethostname():
    syshome = "/home/vall/Documents/Thesis/ThesisPython"
else:
    raise ValueError("Your PC must be registered at the top of this code")

import sys
sys.path.append(syshome)

import imageio as mim
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plab
from matplotlib.ticker import AutoMinorLocator
import os
import v_materials as vmt
import v_meep as vm
import v_meep_analysis as vma
import v_plot as vp
import v_theory as vt
import v_save as vs
import v_utilities as vu

english = False
trs = vu.BilingualManager(english=english)

#%% PARAMETERS

# Saving directories
folder = ["Field/NPMonoch/AuSphere/VacWatField/Vacuum", 
          "Field/NPMonoch/AuSphere/VacWatField/Water"]
home = vs.get_home()

# Parameter for the test
test_param_string = "wlen"
test_param_in_params = True
test_param_position = 0
test_param_label = trs.choose("Wavelength", "Longitud de onda")

# Sorting and labelling data series
sorting_function = [lambda l : vu.sort_by_number(l, test_param_position)]*2
series_label = [lambda s : trs.choose(r"Vacuum", r"Vacío") + 
                rf" $\lambda$ = {vu.find_numbers(s)[test_param_position]:.0f} nm",
                lambda s : trs.choose("Water", "Agua") + 
                rf" $\lambda$ = {vu.find_numbers(s)[test_param_position]:.0f} nm"]
series_must = ["Res3"]*2 # leave "" per default
series_mustnt = ["Old"]*2 # leave "" per default

# Scattering plot options
plot_title_ending = trs.choose("Au 60 nm sphere", "esfera de Au de 60 nm")
series_legend = ["Vacuum", "Water"]
series_colors = [plab.cm.Reds, plab.cm.Blues]
series_linestyles = ["solid"]*2
plot_make_big = False
plot_file = lambda n : os.path.join(home, "DataAnalysis/Field/NPMonoch/AuSphere/VacWatField/WLen/WLen" + n)

#%% LOAD DATA

def file_definer(path):
    return lambda s, n : os.path.join(path, s, n)

path = [os.path.join(home, fold) for fold in folder]
file = [file_definer(pa) for pa in path]

series = []
files_line = []
files_plane = []
results_line = []
results_plane = []
t_line = []
x_line = []
t_plane = []
y_plane = []
z_plane = []
params = []

# for f, sf, sm, smn in zip(folder, sorting_function, series_must, series_mustnt):
for i in range(len(folder)):

    # path.append( os.path.join(home, f) )
    # file.append( lambda f, s : os.path.join(path[-1], f, s) )
    
    series.append( os.listdir(path[i]) )
    series[-1] = vu.filter_by_string_must(series[-1], series_must[i])
    if series_mustnt[i]!="": 
        series[-1] = vu.filter_by_string_must(series[-1], series_mustnt[i], False)
    series[-1] = sorting_function[i](series[-1])
    
    files_line.append( [] )
    files_plane.append( [] )
    for s in series[-1]:
        files_line[-1].append( h5.File(file[i](s, "Field-Lines.h5"), "r") )
        files_plane[-1].append( h5.File(file[i](s, "Field-Planes.h5"), "r") )
    del s
    
    results_line.append( [fi["Ez"] for fi in files_line[-1]] )
    results_plane.append( [fi["Ez"] for fi in files_plane[-1]] )
    params.append( [dict(fi["Ez"].attrs) for fi in files_line[-1]] )
    
    t_line.append( [np.asarray(fi["T"]) for fi in files_line[-1]] )
    x_line.append( [np.asarray(fi["X"]) for fi in files_line[-1]] )
    
    t_plane.append( [np.asarray(fi["T"]) for fi in files_plane[-1]] )
    y_plane.append( [np.asarray(fi["Y"]) for fi in files_plane[-1]] )
    z_plane.append( [np.asarray(fi["Z"]) for fi in files_plane[-1]] )
    
    for s, p in zip(series[-1], params[-1]):
        try:
            f = h5.File(file[-1](s, "Resources.h5"))
            p["used_ram"] = np.array(f["RAM"])
            p["used_swap"] = np.array(f["SWAP"])
            p["elapsed_time"] = np.array(f["ElapsedTime"])
        except FileNotFoundError:
            f = h5.File(file[-1](s, "RAM.h5"))
            p["used_ram"] = np.array(f["RAM"])
            p["used_swap"] = np.array(f["SWAP"])
            p["elapsed_time"] = p["elapsed"]
            del p["elapsed"]
    # del s, p
del i
            
from_um_factor = []
resolution = []
r = []
material = []
paper = []
index = []
wlen = []
cell_width = []
pml_width = []
source_center = []
period_plane = []
period_line = []
until_time = []
time_period_factor = []
norm_until_time = []
norm_amplitude = []
norm_period = []
norm_path = []
sysname = []
for p in params:
    from_um_factor.append( [pi["from_um_factor"] for pi in p] )
    resolution.append( [pi["resolution"] for pi in p] )
    r.append( [pi["r"] for pi in p] )
    material.append( [pi["material"] for pi in p])
    paper.append( [pi["paper"] for pi in p] )
    index.append( [pi["submerged_index"] for pi in p] )
    wlen.append( [pi["wlen"] for pi in p] )
    cell_width.append( [pi["cell_width"] for pi in p] )
    pml_width.append( [pi["pml_width"] for pi in p] )
    source_center.append( [pi["source_center"] for pi in p] )
    period_plane.append( [pi["period_plane"] for pi in p] )
    period_line.append( [pi["period_line"] for pi in p] )
    until_time.append( [pi["until_time"] for pi in p] )
    time_period_factor.append( [pi["time_period_factor"] for pi in p] )
    try:
        norm_until_time.append( [pi["norm_until_time"] for pi in p] )
        norm_amplitude.append( [pi["norm_amplitude"] for pi in p] )    
        norm_period.append( [pi["norm_period"] for pi in p] )
        norm_path.append( [pi["norm_path"] for pi in p] )
        requires_normalization = False
    except:
        requires_normalization = True
    sysname.append( [pi["sysname"] for pi in p] )
del p

#%%

files_line_norm = []

for i in range(len(series)):
    
    files_line_norm.append([])
    
    for j in range(len(series[i])):
            
        try:
            
            try:
                files_line_norm[-1].append( h5.File(file[i](series[i][j], "Field-Lines-Norm.h5"), "r") )
                
            except:
                files_line_norm[-1].append( h5.File(os.path.join(norm_path[i][j], "Field-Lines-Norm.h5"), "r") )
                
            print(f"Loading available normfield for {i},{j}")
        
        except:
            norm_path_ij = vm.check_normfield(params[i][j])
            
            try:
                files_line_norm[-1].append( h5.File(os.path.join(norm_path_ij[0], "Field-Lines-Norm.h5"), "r") )
                print(f"Loading compatible normfield for {i},{j}")
                norm_path[i][j] = norm_path_ij
            
            except:
                files_line_norm[-1].append( files_line[i][j] )
                print(f"Using NP data for {i},{j} normalization.",
                      "This is by all means not ideal!!",
                      "If possible, run again these simulations.")

del i, j
                
results_line_norm = [[ files_line_norm[i][j]["Ez"] for j in range(len(series[i]))] for i in range(len(series))]

t_line_norm = [[ np.asarray(files_line_norm[i][j]["T"]) for j in range(len(series[i]))] for i in range(len(series))]
x_line_norm = [[ np.asarray(files_line_norm[i][j]["X"]) for j in range(len(series[i]))] for i in range(len(series))]

if test_param_in_params:
    test_param = [[p[test_param_string] for p in par] for par in params]
else:
    test_param = [[vu.find_numbers(s)[test_param_position] for s in ser] for ser in series]

minor_division = [[fum * 1e3 / res for fum, res in zip(frum, reso)] for frum, reso in zip(from_um_factor, resolution)]
try:
    width_points = [[int(p["cell_width"] * p["resolution"]) for p in par] for par in params] 
    effective_width_points = [[(p["cell_width"] - 2 * params["pml_width"]) * p["resolution"] for p in par] for par in params]
except:
    width_points = [[2*int((p["pml_width"] + p["empty_width"] + p["r"]) * p["resolution"]) for p in par] for par in params] 
    effective_width_points = [[2*int((p["empty_width"] + p["r"]) * p["resolution"]) for p in par] for par in params]
grid_points = [[wp**3 for wp in wpoints] for wpoints in width_points]
memory_B = [[2 * 12 * gp * 32 for p, gp in zip(par, gpoints)] for par, gpoints in zip(params, grid_points)] # in bytes

elapsed_time = [[p["elapsed_time"] for p in par] for par in params]
total_elapsed_time = [[sum(p["elapsed_time"]) for p in par] for par in params]

used_ram = [[np.array(p["used_ram"])/(1024)**2 for p in par] for par in params]
total_used_ram = [[np.sum(used_ram[i][j], axis=1) for j in range(len(series[i]))] for i in range(len(series))]
used_swap = [[p["used_swap"] for p in par] for par in params]

#%% POSITION RECONSTRUCTION

t_line_index = [[vma.def_index_function(t_line[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
x_line_index = [[vma.def_index_function(x_line[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

t_plane_index = [[vma.def_index_function(t_plane[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
y_plane_index = [[vma.def_index_function(y_plane[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
z_plane_index = [[vma.def_index_function(z_plane[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

t_line_norm_index = [[vma.def_index_function(t_line_norm[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
x_line_norm_index = [[vma.def_index_function(x_line_norm[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

x_line_cropped = [[x_line[i][j][:x_line_index[i][j](cell_width[i][j]/2 - pml_width[i][j])+1] for j in range(len(series[i]))] for i in range(len(series))]
x_line_cropped = [[x_line_cropped[i][j][x_line_index[i][j](-cell_width[i][j]/2 + pml_width[i][j]):] for j in range(len(series[i]))] for i in range(len(series))]

y_plane_cropped = [[y_plane[i][j][:y_plane_index[i][j](cell_width[i][j]/2 - pml_width[i][j])+1] for j in range(len(series[i]))] for i in range(len(series))]
y_plane_cropped = [[y_plane_cropped[i][j][y_plane_index[i][j](-cell_width[i][j]/2 + pml_width[i][j]):] for j in range(len(series[i]))] for i in range(len(series))]

z_plane_cropped = [[z_plane[i][j][:z_plane_index[i][j](cell_width[i][j]/2 - pml_width[i][j])+1] for j in range(len(series[i]))] for i in range(len(series))]
z_plane_cropped = [[z_plane_cropped[i][j][z_plane_index[i][j](-cell_width[i][j]/2 + pml_width[i][j]):] for j in range(len(series[i]))] for i in range(len(series))]

#%% DATA EXTRACTION

source_results = [[vma.get_source_from_line(results_line_norm[i][j], x_line_norm_index[i][j], source_center[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

if not requires_normalization:
    
    period_results, amplitude_results = norm_period, norm_amplitude
    
else:
    
    period_results = [[vma.get_period_from_source(source_results[i][j], t_line_norm[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
    amplitude_results = [[vma.get_amplitude_from_source(source_results[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
    
    results_plane = [[np.asarray(results_plane[i][j]) / amplitude_results[i][j] for j in range(len(series[i]))] for i in range(len(series))]
    results_line = [[np.asarray(results_line[i][j]) / amplitude_results[i][j] for j in range(len(series[i]))] for i in range(len(series))]

sim_source_results = [[vma.get_source_from_line(results_line[i][j], x_line_index[i][j], source_center[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

zprofile_results = [[vma.get_zprofile_from_plane(results_plane[i][j], y_plane_index[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

zprofile_integral = [[vma.integrate_field_zprofile(zprofile_results[i][j], z_plane_index[i][j],
                                                   cell_width[i][j], pml_width[i][j], period_plane[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

zprofile_max = [[vma.find_zpeaks_zprofile(zprofile_results[i][j], z_plane_index[i][j],
                                          cell_width[i][j], pml_width[i][j])[1] for j in range(len(series[i]))] for i in range(len(series))]

field_peaks_all_index = []
field_peaks_single_index = []
field_peaks_zprofile = []
field_peaks_plane = []
for i in range(len(series)):
    field_peaks_all_index.append([])
    field_peaks_single_index.append([])
    field_peaks_zprofile.append([])
    field_peaks_plane.append([])
    for j in range(len(series[i])):
        ind, mxs, zprof, plan = vma.get_all_field_peaks_from_yzplanes(results_plane[i][j],
                                                                    y_plane_index[i][j], z_plane_index[i][j], 
                                                                    cell_width[i][j], pml_width[i][j])
        field_peaks_all_index[-1].append(ind)
        
        ind, mxs, zprof, plan = vma.get_single_field_peak_from_yzplanes(results_plane[i][j],
                                                                       y_plane_index[i][j], z_plane_index[i][j], 
                                                                       cell_width[i][j], pml_width[i][j])
        field_peaks_single_index[-1].append(ind)
        field_peaks_zprofile[-1].append(zprof)
        field_peaks_plane[-1].append(plan)
del i, j, ind, mxs, zprof, plan

#%% SHOW SOURCE AND FOURIER USED FOR NORMALIZATION

# colors = [["C0"], ["C4"], ["C3"]]
colors = [sc(np.linspace(0,1,len(s)+2))[2:] 
          for sc, s in zip(series_colors, series)]

plt.figure()
plt.suptitle(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
             plot_title_ending)

lines = []
for i in range(len(series)):
    for j in range(len(series[i])):
        l, = plt.plot(t_line_norm[i][j] / period_results[i][j], 
                      source_results[i][j] / amplitude_results[i][j],
                      label=series_label[i](series[i][j]),
                      color=colors[i][j])            
        lines.append(l)
plt.xlabel(trs.choose("Time in multiples of period", "Tiempo en múltiplos del período"))
plt.ylabel(trs.choose(r"Normalized Electric Field $E_z(y=z=0)$",
                      r"Campo eléctrico normalizado $E_z(y=z=0)$"))
plt.legend(ncol=2)

plt.savefig(plot_file("Source.png"))
        
fourier = [[np.abs(np.fft.rfft(source_results[i][j] / amplitude_results[i][j])) for j in range(len(series[i]))] for i in range(len(series))]
fourier_freq = [[np.fft.rfftfreq(len(source_results[i][j]), d=period_line[i][j])  for j in range(len(series[i]))] for i in range(len(series))]
fourier_wlen = [[from_um_factor[i][j] * 1e3 / fourier_freq[i][j]  for j in range(len(series[i]))] for i in range(len(series))]
# fourier_max_wlen = [[fourier_wlen[i][j][ np.argmax(fourier[i][j]) ]  for j in range(len(series[i]))] for i in range(len(series))]

plt.figure()
plt.suptitle(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
             plot_title_ending)
lines = []
for i in range(len(series)):
    for j in range(len(series[i])):
        plt.plot(fourier_wlen[i][j], fourier[i][j],
                 label=series_label[i](series[i][j]),
                 color=colors[i][j])
plt.xlabel(trs.choose("Wavelength [nm]", "Longitud de onda [nm]"))
plt.ylabel(trs.choose(r"Electric Field Fourier $\mathcal{F}\;(E_z)$",
                      r"Transformada del campo eléctrico $\mathcal{F}\;(E_z)$"))
plt.legend(ncol=2)

# plt.annotate(trs.choose(f"Maximum at {fourier_max_wlen:.2f} nm",
#                         f"Máximo en {fourier_max_wlen:.2f} nm"),
#              (5, 5), xycoords='figure points')

plt.savefig(plot_file("SourceFFT.png"))

plt.xlim([350, 850])
        
plt.savefig(plot_file("SourceFFTZoom.png"))

#%% SHOW SOURCE AND FOURIER DURING SIMULATION

# colors = [["C0"], ["C4"], ["C3"]]
colors = [sc(np.linspace(0,1,len(s)+2))[2:] 
          for sc, s in zip(series_colors, series)]

plt.figure()
plt.suptitle(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
             plot_title_ending)

lines = []
for i in range(len(series)):
    for j in range(len(series[i])):
        l, = plt.plot(t_line[i][j]/period_results[i][j], 
                      sim_source_results[i][j],
                      label=series_label[i](series[i][j]),
                      color=colors[i][j])            
        lines.append(l)
plt.xlabel(trs.choose("Time in multiples of period", "Tiempo en múltiplos del período"))
plt.ylabel(trs.choose(r"Normalized Electric Field $E_z(y=z=0)$",
                      r"Campo eléctrico normalizado $E_z(y=z=0)$"))
plt.legend(ncol=2)

plt.savefig(plot_file("SimSource.png"))
        
fourier = [[np.abs(np.fft.rfft(sim_source_results[i][j])) for j in range(len(series[i]))] for i in range(len(series))]
fourier_freq = [[np.fft.rfftfreq(len(sim_source_results[i][j]), d=period_line[i][j])  for j in range(len(series[i]))] for i in range(len(series))]
fourier_wlen = [[from_um_factor[i][j] * 1e3 / fourier_freq[i][j]  for j in range(len(series[i]))] for i in range(len(series))]
# fourier_max_wlen = [[fourier_wlen[i][j][ np.argmax(fourier[i][j]) ]  for j in range(len(series[i]))] for i in range(len(series))]

plt.figure()
plt.suptitle(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
             plot_title_ending)
lines = []
for i in range(len(series)):
    for j in range(len(series[i])):
        plt.plot(fourier_wlen[i][j], fourier[i][j],
                 label=series_label[i](series[i][j]),
                 color=colors[i][j])
plt.xlabel(trs.choose("Wavelength [nm]", "Longitud de onda [nm]"))
plt.ylabel(trs.choose(r"Electric Field Fourier $\mathcal{F}\;(E_z)$",
                      r"Transformada del campo eléctrico $\mathcal{F}\;(E_z)$"))
plt.legend(ncol=2)

# plt.annotate(trs.choose(f"Maximum at {fourier_max_wlen:.2f} nm",
#                         f"Máximo en {fourier_max_wlen:.2f} nm"),
#              (5, 5), xycoords='figure points')

plt.savefig(plot_file("SimSourceFFT.png"))

plt.xlim([350, 850])
        
plt.savefig(plot_file("SimSourceFFTZoom.png"))

#%% SHOW FIELD PEAKS INTENSIFICATION OSCILLATIONS

colors = [sc(np.linspace(0,1,len(s)+2))[2:] 
          for sc, s in zip(series_colors, series)]

plt.figure()        
plt.suptitle(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
             plot_title_ending)

for i in range(len(series)):
    for j in range(len(series[i])):
            plt.plot(t_line[i][j] / period_results[i][j], 
                     zprofile_integral[i][j], color=colors[i][j],
                     label=series_label[i](series[i][j]))
plt.xlabel(trs.choose("Time [Mp.u.]", "Tiempo [u.Mp.]"))
plt.ylabel(trs.choose(r"Electric Field Integral $\int E_z(z) \; dz$ [a.u.]",
                      r"Integral del campo eléctrico $\int E_z(z) \; dz$ [u.a.]"))

plt.legend()
plt.savefig(plot_file("Integral.png"))

plt.figure()        
plt.suptitle(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
             plot_title_ending)

for i in range(len(series)):
    for j in range(len(series[i])):
            plt.plot(t_line[i][j]/period_results[i][j], 
                     zprofile_max[i][j],
                     color=colors[i][j],
                     label=series_label[i](series[i][j]))
plt.xlabel(trs.choose("Time [Mp.u.]", "Tiempo [u.Mp.]"))
plt.ylabel(trs.choose(r"Normalized Electric Field Maximum $max[ E_z(z) ]$",
                      r"Máximo del campo eléctrico normalizado $max[ E_z(z) ]$"))

plt.legend()
plt.savefig(plot_file("Maximum.png"))

#%% GET THEORY (SCATTERING)

def wlen_range(material, surrounding_index):
    if material=="Au":
        if surrounding_index == 1: return (450, 750)
        elif surrounding_index == 1.33: return (550, 850)
    else:
        raise ValueError("Please, expand this function.")

scatt_max_wlen_theory = [[vmt.max_scatt_meep(r[i][j] * from_um_factor[i][j] * 1e3, 
                                             material[i][j], 
                                             paper[i][j], 
                                             wlen_range(material[i][j],
                                                        index[i][j]))[0] for j in range(len(series[i]))] for i in range(len(series))]

scatt_max_wlen_predict = [wlen[i][np.argmin(np.abs([wlen[i][j] * from_um_factor[i][j] * 1e3 - scatt_max_wlen_theory[i][j] for j in range(len(series[i]))]))] for i in range(len(series))]

#%% GET THEORY (FIELD)

rvec = []
for i in range(len(series)):
    rvec.append([])
    for j in range(len(series[i])):
        naux = zprofile_results[i][j].shape[-1]
        aux = np.zeros((naux, 3))
        aux[:,2] = np.linspace(-cell_width[i][j]/2 + pml_width[i][j], 
                                cell_width[i][j]/2 - pml_width[i][j], 
                                naux)
        rvec[-1].append(aux)
del aux, naux

E0 = np.array([0,0,1])

zprofile_cm_theory = []
zprofile_ku_theory = []
for i in range(len(series)):
    zprofile_cm_theory.append([])
    zprofile_ku_theory.append([])
    for j in range(len(series[i])):
        medium = vmt.import_medium(material=material[i][j], paper=paper[i][j],
                                   from_um_factor=from_um_factor[i][j])
        epsilon = medium.epsilon(1/wlen[i][j])[0,0]
        # E0 = np.array([0, 0, is_max_in_z_profile[j] * np.real(in_z_profile[j][max_in_z_profile[j], 0])])
        alpha_cm = vt.alpha_Clausius_Mosotti(epsilon, r[i][j], epsilon_ext=index[i][j]**2)
        alpha_ku = vt.alpha_Kuwata(epsilon, wlen[i][j], r[i][j], epsilon_ext=index[i][j]**2)
        theory_cm = np.array([vt.E(epsilon, alpha_cm, E0, 
                                   rv, r[i][j], epsilon_ext=index[i][j]**2) 
                              for rv in rvec[i][j]])[:,-1]
        theory_ku = np.array([vt.E(epsilon, alpha_ku, E0, 
                                   rv, r[i][j], epsilon_ext=index[i][j]**2) 
                              for rv in rvec[i][j]])[:,-1]
        zprofile_cm_theory[-1].append(theory_cm)
        zprofile_ku_theory[-1].append(theory_ku)
        
#%% PLOT MAXIMUM INTENSIFICATION PROFILE (THEORY)

# colors = [["C0"], ["C4"], ["C3"]]
colors = [sc(np.linspace(0,1,len(s)+2))[2:] 
          for sc, s in zip(series_colors, series)]

plt.figure()
plt.suptitle(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
             plot_title_ending)
l_series = []
l_origin = []
for i in range(len(series)):
    for j in range(len(series[i])):
        l_cm, = plt.plot(rvec[i][j][:,-1] * from_um_factor[i][j] * 1e3, 
                         np.abs(zprofile_cm_theory[i][j]),
                         label=series_label[i](series[i][j]),
                         color=colors[i][j], linestyle="dashed")
        l_ku, = plt.plot(rvec[i][j][:,-1]  * from_um_factor[i][j] * 1e3, 
                         np.abs(zprofile_ku_theory[i][j]),
                         label=series_label[i](series[i][j]),
                         color=colors[i][j], linestyle="dotted")
        l_series.append(l_cm)
        if i == 0 and j == len(series[0]) - 1:
            l_origin = [l_cm, l_ku]
plt.xlabel(trs.choose("Position Z [nm]", "Position Z [nm]"))
plt.ylabel(trs.choose(r"Normalized Electric Field $E_z(y=z=0)$",
                      r"Campo eléctrico normalizado $E_z(y=z=0)$"))
plt.legend(ncol=2)

first_legend = plt.legend(l_origin, trs.choose(["CM Theory", "Ku Theory"],
                                               ["Teoría CM", "Teoría Ku"]),
                          loc="center")
second_legend = plt.legend(
    l_series, 
    [l.get_label() for l in l_series],
    loc="upper center")
plt.gca().add_artist(first_legend)

plt.savefig(plot_file("FieldProfileTheory.png"))

#%% PLOT MAXIMUM INTENSIFICATION PROFILE (DATA)

# colors = [["C0"], ["C4"], ["C3"]]
colors = [sc(np.linspace(0,1,len(s)+2))[2:] 
          for sc, s in zip(series_colors, series)]

plt.figure()
plt.suptitle(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
             plot_title_ending)
for i in range(len(series)):
    for j in range(len(series[i])):
        plt.plot(z_plane_cropped[i][j] * from_um_factor[i][j] * 1e3, 
                 field_peaks_zprofile[i][j],
                 label=series_label[i](series[i][j]),
                 color=colors[i][j])
plt.xlabel(trs.choose("Position Z [nm]", "Position Z [nm]"))
plt.ylabel(trs.choose(r"Normalized Electric Field $E_z(y=z=0)$",
                      r"Campo eléctrico normalizado $E_z(y=z=0)$"))
plt.legend()

plt.savefig(plot_file("FieldProfileData.png"))

#%% PLOT MAXIMUM INTENSIFICATION PROFILE (ALL)

# colors = [["C0"], ["C4"], ["C3"]]
colors = [sc(np.linspace(0,1,len(s)+2))[2:] 
          for sc, s in zip(series_colors, series)]

plt.figure()
plt.suptitle(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
             plot_title_ending)
l_origin = []
l_series = []
for i in range(len(series)):
    for j in range(len(series[i])):
        l_mp, = plt.plot(z_plane_cropped[i][j] * from_um_factor[i][j] * 1e3, 
                         field_peaks_zprofile[i][j] ,
                         label=series_label[i](series[i][j]),
                         color=colors[i][j])
        l_cm, = plt.plot(rvec[i][j][:,-1] * from_um_factor[i][j] * 1e3, 
                         np.abs(zprofile_cm_theory[i][j]),
                         color=colors[i][j], linestyle="dashed")
        l_ku, = plt.plot(rvec[i][j][:,-1]  * from_um_factor[i][j] * 1e3, 
                         np.abs(zprofile_ku_theory[i][j]),
                         color=colors[i][j], linestyle="dotted")
        l_series.append(l_mp)
        if i == 0 and j == len(series[i])-1:
            l_origin = [l_mp, l_cm, l_ku]
plt.xlabel(trs.choose("Position Z [nm]", "Posición Z [nm]"))
plt.ylabel(trs.choose(r"Normalized Electric Field $E_z(y=z=0)$",
                      r"Campo eléctrico normalizado $E_z(y=z=0)$"))

plt.legend(ncol=2)

first_legend = plt.legend(l_origin, trs.choose(["MEEP Data", "CM Theory", "Ku Theory"],
                                               ["Data MEEP", "Teoría CM", "Teoría Ku"]),
                          loc="lower right")
second_legend = plt.legend(
    l_series, 
    [l.get_label() for l in l_series],
    loc="upper center")
plt.gca().add_artist(first_legend)

plt.savefig(plot_file("FieldProfileAll.png"))

#%% PLOT MAXIMUM INTENSIFICATION PROFILE (SUBPLOTS)

n = len(series)
m = max([len(s) for s in series])

if n*m <= 3:
    subfig_size = 6.5
if n*m <= 6:
    subfig_size = 4.5
else:
    subfig_size = 3.5

fig = plt.figure(figsize=(m*subfig_size, n*subfig_size))
axes = fig.subplots(ncols=m, nrows=n, sharex=True, sharey=True, gridspec_kw={"wspace":0})
plt.suptitle(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
             plot_title_ending)
        
for i in range(len(series)):
    for j in range(len(series[i])):
        axes[i][j].axvline(r[i][j] * from_um_factor[i][j] * 1e3, 
                           color="k", linestyle="dotted")
        axes[i][j].axvline(-r[i][j] * from_um_factor[i][j] * 1e3, 
                           color="k", linestyle="dotted")
        l_mp, = axes[i][j].plot(z_plane_cropped[i][j] * from_um_factor[i][j] * 1e3, 
                                field_peaks_zprofile[i][j],
                                label=series_label[i](series[i][j]),
                                color=colors[i][j])
        l_cm, = axes[i][j].plot(rvec[i][j][:,-1] * from_um_factor[i][j] * 1e3, 
                                np.abs(zprofile_cm_theory[i][j]),
                                color=colors[i][j], linestyle="dashed")
        l_ku, = axes[i][j].plot(rvec[i][j][:,-1]  * from_um_factor[i][j] * 1e3, 
                                np.abs(zprofile_ku_theory[i][j]),
                                color=colors[i][j], linestyle="dotted")
        xlims = axes[i][j].get_xlim()
        axes[i][j].axhline(0, color="k", linewidth=.5)
        axes[i][j].axvline(0, color="k", linewidth=.5)
        axes[i][j].set_xlim(xlims)
        axes[i][j].set_title(series_label[i](series[i][j]))
        if i==len(series)-1:
            axes[i][j].set_xlabel(trs.choose("Position Z [nm]", "Posición Z [nm]"))
        if j==0:
            axes[i][j].set_ylabel(trs.choose(r"Normalized Electric Field $E_z(y=z=0)$",
                                              r"Campo eléctrico normalizado $E_z(y=z=0)$"))
        if i==len(series)-1 and j==len(series[i])-1:
            axes[i][j].legend([l_mp, l_cm, l_ku], 
                              trs.choose(["MEEP Data", "CM Theory", "Ku Theory"],
                                         ["Data MEEP", "Teoría CM", "Teoría Ku"]),
                              loc="upper center")
        
        axes[i][j].xaxis.set_minor_locator(AutoMinorLocator())
        axes[i][j].yaxis.set_minor_locator(AutoMinorLocator())
        # axes[i][j].grid(True, axis="y", which="major")
        axes[i][j].grid(True, axis="y", which="both", alpha=.3)
        axes[i][j].set_title(series_label[i](series[i][j]))

plt.savefig(plot_file("FieldProfileAllSubplots.png"))

#%% PLOT MAXIMUM INTENSIFICATION FIELD (SUBPLOTS)

n = len(series)
m = max([len(s) for s in series])

if n*m <= 3:
    subfig_size = 6.5
if n*m <= 6:
    subfig_size = 4.5
else:
    subfig_size = 3.5

fig = plt.figure(figsize=(m*subfig_size, n*subfig_size))
axes = fig.subplots(ncols=m, nrows=n)
plt.suptitle(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
             plot_title_ending)
lims = [np.min([np.min([np.min(field_peaks_plane[i][j]) for j in range(len(series[i]))]) for i in range(len(series))]),
        np.max([np.max([np.max(field_peaks_plane[i][j]) for j in range(len(series[i]))]) for i in range(len(series))])]
lims = max([abs(l) for l in lims])
lims = [-lims, lims]

for i in range(len(series)):
    for j in range(len(series[i])):
        ims = axes[i][j].imshow(field_peaks_plane[i][j],
                                cmap='RdBu', #interpolation='spline36', 
                                vmin=lims[0], vmax=lims[1],
                                extent=[min(y_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3,
                                        max(y_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3,
                                        min(z_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3,
                                        max(z_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3])
        if i==len(series)-1:
            axes[i][j].set_xlabel(trs.choose("Position Y [nm]", "Posición Y [nm]"))
        if j==0:
            axes[i][j].set_ylabel(trs.choose("Position Z [nm]", "Posición Z [nm]"))
        if j==len(series[i])-1:
            cax = axes[i][j].inset_axes([1.04, 0, 0.07, 1], #[1.04, 0.2, 0.05, 0.6], 
                                        transform=axes[i][j].transAxes)
            cbar = fig.colorbar(ims, ax=axes[i][j], cax=cax)
            cbar.set_label(trs.choose("Normalized electric field $E_z$",
                                      "Campo eléctrico normalizado $E_z$"))
        # if i==0:
        #     axes[i][j].set_title(series_label[i](series[i][j]), y = 1.05)
        # else:
        #     axes[i][j].set_title(series_label[i](series[i][j]), y = -.3)
        axes[i][j].set_title(series_label[i](series[i][j]))
plt.savefig(plot_file("FieldPlaneAll.png"))

#%% PLOT PROFILE AND PLANE (SUBPLOTS)

n = len(series)
m = max([len(s) for s in series])

if n*m <= 3:
    subfig_size = 6.5
if n*m <= 6:
    subfig_size = 4.5
else:
    subfig_size = 3.5

fig = plt.figure(figsize=(m*subfig_size, n*subfig_size))
axes = fig.subplots(ncols=m, nrows=n, sharex=True, sharey=True, 
                    gridspec_kw={"wspace":0, "hspace":0})
plt.suptitle(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
             plot_title_ending)

lims = [np.min([np.min([np.min(field_peaks_plane[i][j]) for j in range(len(series[i]))]) for i in range(len(series))]),
        np.max([np.max([np.max(field_peaks_plane[i][j]) for j in range(len(series[i]))]) for i in range(len(series))])]
lims = max([abs(l) for l in lims])
lims = [-lims, lims]
        
l_meep = []
l_cmos = []
l_kuwa = []
axes_field = []
for i in range(len(series)):
    axes_field.append([])
    l_meep.append([])
    l_cmos.append([])
    l_kuwa.append([])
    for j in range(len(series[i])):
        axes[i][j].axvline(r[i][j] * from_um_factor[i][j] * 1e3, 
                           color="k", linestyle="dotted")
        axes[i][j].axvline(-r[i][j] * from_um_factor[i][j] * 1e3, 
                           color="k", linestyle="dotted")
        l_mp, = axes[i][j].plot(z_plane_cropped[i][j] * from_um_factor[i][j] * 1e3, 
                                field_peaks_zprofile[i][j],
                                label=series_label[i](series[i][j]),
                                color="k", linewidth=1.4) # color=colors[i][j]
        l_cm, = axes[i][j].plot(rvec[i][j][:,-1] * from_um_factor[i][j] * 1e3, 
                                np.abs(zprofile_cm_theory[i][j]),
                                linestyle=(0, (5, 1)), color="darkorchid") # color=colors[i][j]
        l_ku, = axes[i][j].plot(rvec[i][j][:,-1]  * from_um_factor[i][j] * 1e3, 
                                np.abs(zprofile_ku_theory[i][j]),
                                linestyle=(0, (5, 3)), color="deeppink") # color=colors[i][j]
        l_meep[-1].append(l_mp)
        l_cmos[-1].append(l_cm)
        l_kuwa[-1].append(l_ku)
        
        xlims = axes[i][j].get_xlim()
        axes[i][j].axhline(0, color="k", linewidth=.5)
        axes[i][j].axvline(0, color="k", linewidth=.5)
        axes[i][j].set_xlim(xlims)
        axes[i][j].set_title(series_label[i](series[i][j]))
        if i==len(series)-1:
            axes[i][j].set_xlabel(trs.choose("Position Z [nm]", "Posición Z [nm]"))
        if j==0:
            axes[i][j].set_ylabel(trs.choose(r"Normalized Electric Field $E_z(y=z=0)$",
                                              r"Campo eléctrico normalizado $E_z(y=z=0)$"))
        # else:
        #     axes[i][j].set_ytick_labels([])
        
        axes[i][j].xaxis.set_minor_locator(AutoMinorLocator())
        axes[i][j].yaxis.set_minor_locator(AutoMinorLocator())
        axes[i][j].grid(True, axis="y", which="both", alpha=.3)
        if i==0:
            axes[i][j].set_title(series_label[i](series[i][j]), y = 1.02)
        else:
            axes[i][j].set_title(series_label[i](series[i][j]), y = -.25)
        
        ax_field = vp.add_subplot_axes(axes[i][j], 
                                       [0.5 - 0.45/2, 0.72 - 0.45/2, 0.45, 0.45])
        axes_field[-1].append(ax_field)
        
        ims = ax_field.imshow(field_peaks_plane[i][j].T,
                              cmap='RdBu', #interpolation='spline36', 
                              vmin=lims[0], vmax=lims[1],
                              extent=[min(y_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3,
                                      max(y_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3,
                                      min(z_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3,
                                      max(z_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3])
        ax_field.set_xticks([])
        ax_field.set_yticks([])
        
        if i==len(series)-1 and j==len(series[i])-1:
            cax = axes[i][j].inset_axes([1.04, 0, 0.07, 2], #[1.04, 0, 0.07, 1], 
                                        transform=axes[i][j].transAxes)
            cbar = fig.colorbar(ims, ax=axes[i][j], cax=cax)
            cbar.set_label(trs.choose("Normalized electric field $E_z$",
                                      "Campo eléctrico normalizado $E_z$"))
            cbar.ax.minorticks_on()
        
leg = plt.legend(
    [l_meep[-1][-1], l_cmos[-1][-1], l_kuwa[-1][-1]],
    trs.choose(["MEEP Data", "CM Theory", "Ku Theory"], ["Data MEEP", "Teoría CM", "Teoría Ku"]),
    bbox_to_anchor=(2.6, -1.45), #(2.5, 1.4), 
    loc="center right", frameon=False)

fig.set_size_inches([13.5,  7.9]) # ([17.5,  7.9])
plt.savefig(plot_file("AllSubplots.png"))

#%% PLOT PROFILE AND PLANE (SUBPLOTS) [ABSOLUTE VALUE]

n = len(series)
m = max([len(s) for s in series])

if n*m <= 3:
    subfig_size = 6.5
if n*m <= 6:
    subfig_size = 4.5
else:
    subfig_size = 3.5

fig = plt.figure(figsize=(m*subfig_size, n*subfig_size))
axes = fig.subplots(ncols=m, nrows=n, sharex=True, sharey=True, 
                    gridspec_kw={"wspace":0, "hspace":0})
plt.suptitle(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
             plot_title_ending)

# lims = [np.min([np.min([np.min(np.power(np.abs(field_peaks_plane[i][j]),2)) for j in range(len(series[i]))]) for i in range(len(series))]), 
#         np.max([np.max([np.max(np.power(np.abs(field_peaks_plane[i][j]),2)) for j in range(len(series[i]))]) for i in range(len(series))])] 
lims = [np.min([np.min([np.min(np.abs(field_peaks_plane[i][j])) for j in range(len(series[i]))]) for i in range(len(series))]), 
        np.max([np.max([np.max(np.abs(field_peaks_plane[i][j])) for j in range(len(series[i]))]) for i in range(len(series))])] 
lims = max([abs(l) for l in lims])
lims = [0, lims]
        
l_meep = []
l_cmos = []
l_kuwa = []
axes_field = []
for i in range(len(series)):
    axes_field.append([])
    l_meep.append([])
    l_cmos.append([])
    l_kuwa.append([])
    for j in range(len(series[i])):
        axes[i][j].axvline(r[i][j] * from_um_factor[i][j] * 1e3, 
                           color="k", linestyle="dotted")
        axes[i][j].axvline(-r[i][j] * from_um_factor[i][j] * 1e3, 
                           color="k", linestyle="dotted")
        l_mp, = axes[i][j].plot(z_plane_cropped[i][j] * from_um_factor[i][j] * 1e3, 
                                np.abs(field_peaks_zprofile[i][j]),
                                label=series_label[i](series[i][j]),
                                color="k", linewidth=1.4) # color=colors[i][j]
        l_cm, = axes[i][j].plot(rvec[i][j][:,-1] * from_um_factor[i][j] * 1e3, 
                                np.abs(zprofile_cm_theory[i][j]),
                                linestyle=(0, (5, 1)), color="darkorchid") # color=colors[i][j]
        l_ku, = axes[i][j].plot(rvec[i][j][:,-1]  * from_um_factor[i][j] * 1e3, 
                                np.abs(zprofile_ku_theory[i][j]),
                                linestyle=(0, (5, 3)), color="deeppink") # color=colors[i][j]
        l_meep[-1].append(l_mp)
        l_cmos[-1].append(l_cm)
        l_kuwa[-1].append(l_ku)
        
        xlims = axes[i][j].get_xlim()
        axes[i][j].axhline(0, color="k", linewidth=.5)
        axes[i][j].axvline(0, color="k", linewidth=.5)
        axes[i][j].set_xlim(xlims)
        axes[i][j].set_title(series_label[i](series[i][j]))
        if i==len(series)-1:
            axes[i][j].set_xlabel(trs.choose("Position Z [nm]", "Posición Z [nm]"))
        if j==0:
            axes[i][j].set_ylabel(trs.choose(r"Normalized Electric Field $|E_z|(y=z=0)$",
                                             r"Campo eléctrico normalizado $|E_z|(y=z=0)$"))
        # else:
        #     axes[i][j].set_ytick_labels([])
        
        axes[i][j].xaxis.set_minor_locator(AutoMinorLocator())
        axes[i][j].yaxis.set_minor_locator(AutoMinorLocator())
        axes[i][j].grid(True, axis="y", which="both", alpha=.3)
        if i==0:
            axes[i][j].set_title(series_label[i](series[i][j]), y = 1.02)
        else:
            axes[i][j].set_title(series_label[i](series[i][j]), y = -.25)
        
        ax_field = vp.add_subplot_axes(axes[i][j], 
                                       [0.5 - 0.45/2, 0.65 - 0.45/2, 0.45, 0.45])
        axes_field[-1].append(ax_field)
        
        ims = ax_field.imshow(np.abs(field_peaks_plane[i][j]).T, #np.power(np.abs(field_peaks_plane[i][j]),2).T
                              cmap='Reds', #interpolation='spline36', 
                              vmin=lims[0], vmax=lims[1],
                              extent=[min(y_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3,
                                      max(y_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3,
                                      min(z_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3,
                                      max(z_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3])
        ax_field.set_xticks([])
        ax_field.set_yticks([])
        
        if i==len(series)-1 and j==len(series[i])-1:
            cax = axes[i][j].inset_axes([1.04, 0, 0.07, 2], #[1.04, 0, 0.07, 1], 
                                        transform=axes[i][j].transAxes)
            cbar = fig.colorbar(ims, ax=axes[i][j], cax=cax)
            cbar.set_label(trs.choose(r"Normalized Squared Electric Field $|E_z|^2(y=z=0)$",
                                      r"Cuadrado del campo eléctrico normalizado $|E_z|^2(y=z=0)$"))
            cbar.ax.minorticks_on()
        
leg = plt.legend(
    [l_meep[-1][-1], l_cmos[-1][-1], l_kuwa[-1][-1]],
    trs.choose(["MEEP Data", "CM Theory", "Ku Theory"], ["Data MEEP", "Teoría CM", "Teoría Ku"]),
    bbox_to_anchor=(2.6, -1.3), #(2.5, 1.4), 
    loc="center right", frameon=False)

fig.set_size_inches([13.5,  7.9]) # [17.5,  7.9]
plt.savefig(plot_file("AllSubplotsAbs.png"))

#%% MAKE PROFILE GIF

maxnframes = 300

# What should be parameters
nframes = min(maxnframes, np.max([[zprofile_results[i][j].shape[-1] for j in range(len(series[i]))] for i in range(len(series))]))
nframes = int( vu.round_to_multiple(nframes, np.max(time_period_factor) ) )
nframes_period = int(nframes/np.max(time_period_factor))

iref = -1
jref = 0
label_function = lambda k : trs.choose('Time: {:.1f} of period',
                                       'Tiempo: {:.1f} del período').format(
                                           t_line[iref][jref][k]/period_results[iref][jref])

# Animation base
n = len(series)
m = max([len(s) for s in series])

if n*m <= 3:
    subfig_size = 6.5
if n*m <= 6:
    subfig_size = 4.5
else:
    subfig_size = 3.5

fig = plt.figure(figsize=(m*subfig_size, n*subfig_size))
axes = fig.subplots(ncols=m, nrows=n)
lims = [np.min([np.min([np.min(zprofile_results[i][j]) for j in range(len(series[i]))]) for i in range(len(series))]), 
        np.max([np.max([np.max(zprofile_results[i][j]) for j in range(len(series[i]))]) for i in range(len(series))])] 

def draw_pml_box(i, j):
    axes[i][j].axvline((-cell_width[i][j]/2 + pml_width[i][j]) * from_um_factor[i][j] * 1e3, 
                        linestyle=":", color='k')
    axes[i][j].axvline((cell_width[i][j]/2 - pml_width[i][j]) * from_um_factor[i][j] * 1e3, 
                       linestyle=":", color='k')
    axes[i][j].axhline(0, color='k', linewidth=1)

def get_cut_points(i, j):
    cut_points = []
    for k in range(int(time_period_factor[i][j])):
        cut_points.append( np.argmin(np.abs(t_line[i][j] / period_results[i][j] - (k + 1))) )
    return [0, *cut_points]
cut_points_index = [[get_cut_points(i, j) for j in range(len(series[i]))] for i in range(len(series))]

nframes = int( vu.round_to_multiple(nframes, np.max(time_period_factor) ) )
nframes_period = int(nframes/np.max(time_period_factor))

def set_frame_numbers(i, j):
    frames_index = []
    for k in range(len(cut_points_index[i][j])):
        if k < len(cut_points_index[i][j])-1:
            intermediate_frames = np.linspace(
                cut_points_index[i][j][k],
                cut_points_index[i][j][k+1],
                nframes_period+1)[:-1]
            intermediate_frames = [int(fr) for fr in intermediate_frames]
            frames_index = [*frames_index, *intermediate_frames]
    return frames_index
frames_index = [[set_frame_numbers(i, j) for j in range(len(series[i]))] for i in range(len(series))]

def make_pic_line(k):
    for i in range(len(series)):
        for j in range(len(series[i])):
            kij = frames_index[i][j][k]
            axes[i][j].clear()
            axes[i][j].plot(z_plane[i][j] * from_um_factor[i][j] * 1e3, 
                            zprofile_results[i][j][:, kij],
                            color=colors[i][j])
            axes[i][j].axhline(zprofile_max[i][j][kij] ,
                               linestyle="dashed", color=colors[i][j])
            axes[i][j].axhline(0, linewidth=1, color="k")
            axes[i][j].set_ylim(*lims)
            if i==len(series)-1 and j==0:
                axes[i][j].text(-.2, -.2, label_function(kij), 
                                transform=axes[i][j].transAxes)
            draw_pml_box(i, j)
            if i==len(series)-1:
                axes[i][j].set_xlabel(trs.choose("Distance Z [nm]", "Distancia Z [nm]"))
            if j==0:
                axes[i][j].set_ylabel(trs.choose(r"Normalized Electric Field Profile $E_z|_{z=0}$",
                                                 r"Campo eléctrico normalizado $E_z|_{z=0}$"))
            axes[i][j].set_xlim(min(z_plane[i][j]) * from_um_factor[i][j] * 1e3, 
                                max(z_plane[i][j]) * from_um_factor[i][j] * 1e3)
            plt.show()
    return axes

def make_gif_line(gif_filename):
    pics = []
    for k in range(nframes):
        axes = make_pic_line(k)
        plt.savefig('temp_pic.png') 
        pics.append(mim.imread('temp_pic.png')) 
        print(str(k+1)+'/'+str(nframes))
    mim.mimsave(gif_filename+'.gif', pics, fps=5)
    os.remove('temp_pic.png')
    print('Saved gif')

make_gif_line(plot_file("AxisZ"))
plt.close(fig)
# del fig, ax, lims, nframes_step, nframes, call_series, label_function

#%% MAKE YZ PLANE FIELD GIF

maxnframes = 300

# What should be parameters
nframes = min(maxnframes, np.max([[zprofile_results[i][j].shape[-1] for j in range(len(series[i]))] for i in range(len(series))]))
nframes = int( vu.round_to_multiple(nframes, np.max(time_period_factor) ) )
nframes_period = int(nframes/np.max(time_period_factor))

iref = -1
jref = 0
label_function = lambda k : trs.choose('Time: {:.1f} of period',
                                       'Tiempo: {:.1f} del período').format(
                                           t_line[iref][jref][k]/period_results[iref][jref])

# Animation base
fig = plt.figure(figsize=(m*subfig_size, n*subfig_size))
axes = fig.subplots(ncols=m, nrows=n)
plt.suptitle(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
             plot_title_ending)
lims = [np.min([np.min([np.min(results_plane[i][j]) for j in range(len(series[i]))]) for i in range(len(series))]),
        np.max([np.max([np.max(results_plane[i][j]) for j in range(len(series[i]))]) for i in range(len(series))])]
lims = max([abs(l) for l in lims])
lims = [-lims, lims]

def draw_pml_box(i, j):
    axes[i][j].axvline((-cell_width[i][j]/2 + pml_width[i][j]) * from_um_factor[i][j] * 1e3, 
                        linestyle=":", color='k')
    axes[i][j].axvline((cell_width[i][j]/2 - pml_width[i][j]) * from_um_factor[i][j] * 1e3, 
                       linestyle=":", color='k')
    axes[i][j].axhline((-cell_width[i][j]/2 + pml_width[i][j]) * from_um_factor[i][j] * 1e3, 
                        linestyle=":", color='k')
    axes[i][j].axhline((cell_width[i][j]/2 - pml_width[i][j]) * from_um_factor[i][j] * 1e3, 
                       linestyle=":", color='k')

def get_cut_points(i, j):
    cut_points = []
    for k in range(int(time_period_factor[i][j])):
        cut_points.append( np.argmin(np.abs(t_line[i][j] / period_results[i][j] - (k + 1))) )
    return [0, *cut_points]
cut_points_index = [[get_cut_points(i, j) for j in range(len(series[i]))] for i in range(len(series))]

nframes = int( vu.round_to_multiple(nframes, np.max(time_period_factor) ) )
nframes_period = int(nframes/np.max(time_period_factor))

def set_frame_numbers(i, j):
    frames_index = []
    for k in range(len(cut_points_index[i][j])):
        if k < len(cut_points_index[i][j])-1:
            intermediate_frames = np.linspace(
                cut_points_index[i][j][k],
                cut_points_index[i][j][k+1],
                nframes_period+1)[:-1]
            intermediate_frames = [int(fr) for fr in intermediate_frames]
            frames_index = [*frames_index, *intermediate_frames]
    return frames_index
frames_index = [[set_frame_numbers(i, j) for j in range(len(series[i]))] for i in range(len(series))]

def make_pic_line(k):
    for i in range(len(series)):
        for j in range(len(series[i])):
            kij = frames_index[i][j][k]
            axes[i][j].clear()
            axes[i][j].imshow(results_plane[i][j][...,kij].T,
                              cmap='RdBu', #interpolation='spline36', 
                              vmin=lims[0], vmax=lims[1],
                              extent=[min(y_plane[i][j]) * from_um_factor[i][j] * 1e3,
                                      max(y_plane[i][j]) * from_um_factor[i][j] * 1e3,
                                      min(z_plane[i][j]) * from_um_factor[i][j] * 1e3,
                                      max(z_plane[i][j]) * from_um_factor[i][j] * 1e3])
            if i==len(series)-1:
                axes[i][j].set_xlabel(trs.choose("Position Y [nm]", "Posición Y [nm]"))
            if j==0:
                axes[i][j].set_ylabel(trs.choose("Position Z [nm]", "Posición Z [nm]"))
            if j==len(series[i])-1:
                cax = axes[i][j].inset_axes([1.04, 0, 0.07, 1], #[1.04, 0.2, 0.05, 0.6], 
                                            transform=axes[i][j].transAxes)
                cbar = fig.colorbar(ims, ax=axes[i][j], cax=cax)
                cbar.set_label(trs.choose("Normalized electric field $E_z$",
                                          "Campo eléctrico normalizado $E_z$"))
            if i==len(series)-1 and j==0:
                axes[i][j].text(-.2, -.2, label_function(kij), 
                                transform=axes[i][j].transAxes)
            draw_pml_box(i, j)
            plt.show()
    return axes

def make_gif_line(gif_filename):
    pics = []
    for k in range(nframes):
        axes = make_pic_line(k)
        plt.savefig('temp_pic.png') 
        pics.append(mim.imread('temp_pic.png')) 
        print(str(k+1)+'/'+str(nframes))
    mim.mimsave(gif_filename+'.gif', pics, fps=5)
    os.remove('temp_pic.png')
    print('Saved gif')

make_gif_line(plot_file("PlanesYZ"))
plt.close(fig)
# del fig, ax, lims, nframes_step, nframes, call_series, label_function

#%% MAKE CROPPED PROFILE GIF

zprofile_cropped = [[vma.crop_field_zprofile(zprofile_results[i][j], 
                                             z_plane_index[i][j], 
                                             cell_width[i][j], pml_width[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

maxnframes = 300

# What should be parameters
nframes = min(maxnframes, np.max([[zprofile_cropped[i][j].shape[-1] for j in range(len(series[i]))] for i in range(len(series))]))
nframes = int( vu.round_to_multiple(nframes, np.max(time_period_factor) ) )
nframes_period = int(nframes/np.max(time_period_factor))

iref = -1
jref = 0
label_function = lambda k : trs.choose('Time: {:.1f} of period',
                                       'Tiempo: {:.1f} del período').format(
                                           t_line[iref][jref][k]/period_results[iref][jref])

# Animation base
n = len(series)
m = max([len(s) for s in series])

if n*m <= 3:
    subfig_size = 6.5
if n*m <= 6:
    subfig_size = 4.5
else:
    subfig_size = 3.5

fig = plt.figure(figsize=(m*subfig_size, n*subfig_size))
axes = fig.subplots(ncols=m, nrows=n)
lims = [np.min([np.min([np.min(zprofile_cropped[i][j]) for j in range(len(series[i]))]) for i in range(len(series))]), 
        np.max([np.max([np.max(zprofile_cropped[i][j]) for j in range(len(series[i]))]) for i in range(len(series))])] 

def get_cut_points(i, j):
    cut_points = []
    for k in range(int(time_period_factor[i][j])):
        cut_points.append( np.argmin(np.abs(t_line[i][j] / period_results[i][j] - (k + 1))) )
    return [0, *cut_points]
cut_points_index = [[get_cut_points(i, j) for j in range(len(series[i]))] for i in range(len(series))]

nframes = int( vu.round_to_multiple(nframes, np.max(time_period_factor) ) )
nframes_period = int(nframes/np.max(time_period_factor))

def set_frame_numbers(i, j):
    frames_index = []
    for k in range(len(cut_points_index[i][j])):
        if k < len(cut_points_index[i][j])-1:
            intermediate_frames = np.linspace(
                cut_points_index[i][j][k],
                cut_points_index[i][j][k+1],
                nframes_period+1)[:-1]
            intermediate_frames = [int(fr) for fr in intermediate_frames]
            frames_index = [*frames_index, *intermediate_frames]
    return frames_index
frames_index = [[set_frame_numbers(i, j) for j in range(len(series[i]))] for i in range(len(series))]

def make_pic_line(k):
    for i in range(len(series)):
        for j in range(len(series[i])):
            kij = frames_index[i][j][k]
            axes[i][j].clear()
            axes[i][j].plot(z_plane_cropped[i][j] * from_um_factor[i][j] * 1e3, 
                            zprofile_cropped[i][j][:, kij],
                            color=colors[i][j])
            axes[i][j].axvline(r[i][j] * from_um_factor[i][j] * 1e3, 
                               color="k", linestyle="dotted")
            axes[i][j].axvline(-r[i][j] * from_um_factor[i][j] * 1e3, 
                               color="k", linestyle="dotted")
            axes[i][j].axhline(zprofile_max[i][j][kij] ,
                               linestyle="dashed", color=colors[i][j])
            axes[i][j].axhline(0, color="k", linewidth=1)
            axes[i][j].set_ylim(*lims)
            if i==len(series)-1 and j==0:
                axes[i][j].text(-.2, -.2, label_function(kij), 
                                transform=axes[i][j].transAxes)
            if i==len(series)-1:
                axes[i][j].set_xlabel(trs.choose("Distance Z [nm]", "Distancia Z [nm]"))
            if j==0:
                axes[i][j].set_ylabel(trs.choose(r"Normalized Electric Field Profile $E_z|_{z=0}$",
                                                 r"Campo eléctrico normalizado $E_z|_{z=0}$"))
            axes[i][j].set_xlim(min(z_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3, 
                                max(z_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3)
            plt.show()
    return axes

def make_gif_line(gif_filename):
    pics = []
    for k in range(nframes):
        axes = make_pic_line(k)
        plt.savefig('temp_pic.png') 
        pics.append(mim.imread('temp_pic.png')) 
        print(str(k+1)+'/'+str(nframes))
    mim.mimsave(gif_filename+'.gif', pics, fps=5)
    os.remove('temp_pic.png')
    print('Saved gif')
#%%

make_gif_line(plot_file("CroppedAxisZ"))
plt.close(fig)
# del fig, ax, lims, nframes_step, nframes, call_series, label_function

#%% MAKE YZ PLANE FIELD GIF

plane_cropped = [[vma.crop_field_yzplane(results_plane[i][j], 
                                         y_plane_index[i][j], z_plane_index[i][j], 
                                         cell_width[i][j], pml_width[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

maxnframes = 300

# What should be parameters
nframes = min(maxnframes, np.max([[plane_cropped[i][j].shape[-1] for j in range(len(series[i]))] for i in range(len(series))]))
nframes = int( vu.round_to_multiple(nframes, np.max(time_period_factor) ) )
nframes_period = int(nframes/np.max(time_period_factor))

iref = -1
jref = 0
label_function = lambda k : trs.choose('Time: {:.1f} of period',
                                       'Tiempo: {:.1f} del período').format(
                                           t_line[iref][jref][k]/period_results[iref][jref])

# Animation base
fig = plt.figure(figsize=(m*subfig_size, n*subfig_size))
axes = fig.subplots(ncols=m, nrows=n)
plt.suptitle(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
             plot_title_ending)
lims = [np.min([np.min([np.min(plane_cropped[i][j]) for j in range(len(series[i]))]) for i in range(len(series))]),
        np.max([np.max([np.max(plane_cropped[i][j]) for j in range(len(series[i]))]) for i in range(len(series))])]
lims = max([abs(l) for l in lims])
lims = [-lims, lims]

def get_cut_points(i, j):
    cut_points = []
    for k in range(int(time_period_factor[i][j])):
        cut_points.append( np.argmin(np.abs(t_line[i][j] / period_results[i][j] - (k + 1))) )
    return [0, *cut_points]
cut_points_index = [[get_cut_points(i, j) for j in range(len(series[i]))] for i in range(len(series))]

nframes = int( vu.round_to_multiple(nframes, np.max(time_period_factor) ) )
nframes_period = int(nframes/np.max(time_period_factor))

def set_frame_numbers(i, j):
    frames_index = []
    for k in range(len(cut_points_index[i][j])):
        if k < len(cut_points_index[i][j])-1:
            intermediate_frames = np.linspace(
                cut_points_index[i][j][k],
                cut_points_index[i][j][k+1],
                nframes_period+1)[:-1]
            intermediate_frames = [int(fr) for fr in intermediate_frames]
            frames_index = [*frames_index, *intermediate_frames]
    return frames_index
frames_index = [[set_frame_numbers(i, j) for j in range(len(series[i]))] for i in range(len(series))]

def make_pic_line(k):
    for i in range(len(series)):
        for j in range(len(series[i])):
            kij = frames_index[i][j][k]
            axes[i][j].clear()
            axes[i][j].imshow(plane_cropped[i][j][...,kij].T,
                              cmap='RdBu', #interpolation='spline36', 
                              vmin=lims[0], vmax=lims[1],
                              extent=[min(y_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3,
                                      max(y_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3,
                                      min(z_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3,
                                      max(z_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3])
            if i==len(series)-1:
                axes[i][j].set_xlabel(trs.choose("Position Y [nm]", "Posición Y [nm]"))
            if j==0:
                axes[i][j].set_ylabel(trs.choose("Position Z [nm]", "Posición Z [nm]"))
            if j==len(series[i])-1:
                cax = axes[i][j].inset_axes([1.04, 0, 0.07, 1], #[1.04, 0.2, 0.05, 0.6], 
                                            transform=axes[i][j].transAxes)
                cbar = fig.colorbar(ims, ax=axes[i][j], cax=cax)
                cbar.set_label(trs.choose("Normalized electric field $E_z$",
                                          "Campo eléctrico normalizado $E_z$"))
            if i==len(series)-1 and j==0:
                axes[i][j].text(-.2, -.2, label_function(kij), 
                                transform=axes[i][j].transAxes)
            plt.show()
    return axes

def make_gif_line(gif_filename):
    pics = []
    for k in range(nframes):
        axes = make_pic_line(k)
        plt.savefig('temp_pic.png') 
        pics.append(mim.imread('temp_pic.png')) 
        print(str(k+1)+'/'+str(nframes))
    mim.mimsave(gif_filename+'.gif', pics, fps=5)
    os.remove('temp_pic.png')
    print('Saved gif')
   
make_gif_line(plot_file("CroppedPlanesYZ"))
plt.close(fig)
# del fig, ax, lims, nframes_step, nframes, call_series, label_function

#%% MAKE PROFILE AND PLANE GIF (SUBPLOTS)

zprofile_cropped = [[vma.crop_single_field_zprofile(zprofile_results[i][j], z_plane_index[i][j], 
                                             cell_width[i][j], pml_width[i][j])
                     for j in range(len(series[i]))] for i in range(len(series))]
plane_cropped = [[vma.crop_single_field_yzplane(results_plane[i][j], 
                                         y_plane_index[i][j], z_plane_index[i][j], 
                                         cell_width[i][j], pml_width[i][j])
                     for j in range(len(series[i]))] for i in range(len(series))]


maxnframes = 300

# What should be parameters
nframes = min(maxnframes, np.max([[zprofile_cropped[i][j].shape[-1] for j in range(len(series[i]))] for i in range(len(series))]))
nframes = int( vu.round_to_multiple(nframes, np.max(time_period_factor) ) )
nframes_period = int(nframes/np.max(time_period_factor))

iref = -1
jref = 0
label_function = lambda k : trs.choose('     Time:\n{:.1f} of period',
                                       '     Tiempo:\n{:.1f} del período').format(
                                           t_line[iref][jref][k]/period_results[iref][jref])

# Animation base
n = len(series)
m = max([len(s) for s in series])

if n*m <= 3:
    subfig_size = 6.5
if n*m <= 6:
    subfig_size = 4.5
else:
    subfig_size = 3.5

fig = plt.figure(figsize=(m*subfig_size, n*subfig_size))
fig.set_size_inches([13.5,  7.9])
axes = fig.subplots(ncols=m, nrows=n, sharex=True, sharey=True, 
                    gridspec_kw={"wspace":0, "hspace":0})
axes_field = [[vp.add_subplot_axes(axes[i][j], 
                                   [0.5 - 0.35/2, 0.8 - 0.35/2, 0.35, 0.35])
               for j in range(len(series[i]))] for i in range(len(series))]
colors = [sc(np.linspace(0,1,len(s)+2))[2:] 
          for sc, s in zip(series_colors, series)]

plt.suptitle(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
             plot_title_ending)
lims = [np.min([np.min([np.min(plane_cropped[i][j]) for j in range(len(series[i]))]) for i in range(len(series))]), 
        np.max([np.max([np.max(plane_cropped[i][j]) for j in range(len(series[i]))]) for i in range(len(series))])] 
lims = max([abs(l) for l in lims])
lims = [-lims, lims]

def get_cut_points(i, j):
    cut_points = []
    for k in range(int(time_period_factor[i][j])):
        cut_points.append( np.argmin(np.abs(t_line[i][j] / period_results[i][j] - (k + 1))) )
    return [0, *cut_points]
cut_points_index = [[get_cut_points(i, j) for j in range(len(series[i]))] for i in range(len(series))]

nframes = int( vu.round_to_multiple(nframes, np.max(time_period_factor) ) )
nframes_period = int(nframes/np.max(time_period_factor))

def set_frame_numbers(i, j):
    frames_index = []
    for k in range(len(cut_points_index[i][j])):
        if k < len(cut_points_index[i][j])-1:
            intermediate_frames = np.linspace(
                cut_points_index[i][j][k],
                cut_points_index[i][j][k+1],
                nframes_period+1)[:-1]
            intermediate_frames = [int(fr) for fr in intermediate_frames]
            frames_index = [*frames_index, *intermediate_frames]
    return frames_index
frames_index = [[set_frame_numbers(i, j) for j in range(len(series[i]))] for i in range(len(series))]

def make_pic_line(k):       
    for i in range(len(series)):
        for j in range(len(series[i])):
            kij = frames_index[i][j][k]
            
            axes[i][j].clear()
            axes_field[i][j].clear()
            
            axes[i][j].axvline(r[i][j] * from_um_factor[i][j] * 1e3, 
                               color="k", linestyle="dotted")
            axes[i][j].axvline(-r[i][j] * from_um_factor[i][j] * 1e3, 
                               color="k", linestyle="dotted")
            l_mp, = axes[i][j].plot(z_plane_cropped[i][j] * from_um_factor[i][j] * 1e3, 
                                    zprofile_cropped[i][j][...,kij],
                                    label=series_label[i](series[i][j]),
                                    color="k", linewidth=1.4) # color=colors[i][j]
            l_cm, = axes[i][j].plot(rvec[i][j][:,-1] * from_um_factor[i][j] * 1e3, 
                                    np.abs(zprofile_cm_theory[i][j]),
                                    linestyle=(0, (5, 1)), color="darkorchid") # color=colors[i][j]
            l_ku, = axes[i][j].plot(rvec[i][j][:,-1]  * from_um_factor[i][j] * 1e3, 
                                    np.abs(zprofile_ku_theory[i][j]),
                                    linestyle=(0, (5, 3)), color="deeppink") # color=colors[i][j]
            axes[i][j].plot(rvec[i][j][:,-1] * from_um_factor[i][j] * 1e3, 
                            -np.abs(zprofile_cm_theory[i][j]),
                            linestyle=(0, (5, 1)), color="darkorchid") # color=colors[i][j]
            axes[i][j].plot(rvec[i][j][:,-1]  * from_um_factor[i][j] * 1e3, 
                            -np.abs(zprofile_ku_theory[i][j]),
                            linestyle=(0, (5, 3)), color="deeppink") # color=colors[i][j]
            
            axes[i][j].axhline(0, color="k", linewidth=.5)
            axes[i][j].axvline(0, color="k", linewidth=.5)
            axes[i][j].set_title(series_label[i](series[i][j]))
            axes[i][j].set_ylim(lims)
            if i==len(series)-1 and j==0:
                axes[i][j].text(-.15, -.2, label_function(kij), 
                                transform=axes[i][j].transAxes)
            if i==len(series)-1:
                axes[i][j].set_xlabel(trs.choose("Position Z [nm]", "Posición Z [nm]"))
            if j==0:
                axes[i][j].set_ylabel(trs.choose(r"Normalized Electric Field $E_z(y=z=0)$",
                                                  r"Campo eléctrico normalizado $E_z(y=z=0)$"))
            # else:
            #     axes[i][j].set_ytick_labels([])
            
            axes[i][j].xaxis.set_minor_locator(AutoMinorLocator())
            axes[i][j].yaxis.set_minor_locator(AutoMinorLocator())
            axes[i][j].grid(True, axis="y", which="both", alpha=.3)
            if i==0:
                axes[i][j].set_title(series_label[i](series[i][j]), y = 1.02)
            else:
                axes[i][j].set_title(series_label[i](series[i][j]), y = -.25)
                        
            ims = axes_field[i][j].imshow(plane_cropped[i][j][...,kij].T,
                                          cmap='RdBu', #interpolation='spline36', 
                                          vmin=lims[0], vmax=lims[1],
                                          extent=[min(y_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3,
                                                  max(y_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3,
                                                  min(z_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3,
                                                  max(z_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3])
            axes_field[i][j].set_xticks([])
            axes_field[i][j].set_yticks([])
            
            if i==len(series)-1 and j==len(series[i])-1:
                cax = axes[i][j].inset_axes([1.04, 0, 0.07, 2], #[1.04, 0, 0.07, 1], 
                                            transform=axes[i][j].transAxes)
                cbar = fig.colorbar(ims, ax=axes[i][j], cax=cax)
                cbar.set_label(trs.choose("Normalized electric field $E_z$",
                                          "Campo eléctrico normalizado $E_z$"))
                cbar.ax.minorticks_on()
            
    leg = plt.legend(
        [l_mp, l_cm, l_ku],
        trs.choose(["MEEP Data", "CM Theory", "Ku Theory"], ["Data MEEP", "Teoría CM", "Teoría Ku"]),
        bbox_to_anchor=(2.85, -1.8), #(2.5, 1.4), 
        loc="center right", frameon=False)
    
    return axes

def make_gif_line(gif_filename):
    pics = []
    for k in range(nframes):
        axes = make_pic_line(k)
        plt.savefig('temp_pic.png') 
        pics.append(mim.imread('temp_pic.png')) 
        print(str(k+1)+'/'+str(nframes))
    mim.mimsave(gif_filename+'.gif', pics, fps=5)
    os.remove('temp_pic.png')
    print('Saved gif')  

#%%

make_gif_line(plot_file("All2"))
plt.close(fig)
