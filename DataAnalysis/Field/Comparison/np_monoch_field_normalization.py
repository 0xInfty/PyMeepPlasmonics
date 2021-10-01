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
import os
from scipy.signal import find_peaks
import v_materials as vmt
import v_theory as vt
import v_save as vs
import v_utilities as vu
import v_meep_analysis as vma

english = False
trs = vu.BilingualManager(english=english)

#%% PARAMETERS

# Saving directories
folder = ["Field/NPMonoch/AuSphere/VacWatTest/TestNorm/Vacuum", 
          "Field/NPMonoch/AuSphere/VacWatTest/TestNorm/Water"]
home = vs.get_home()

# Parameter for the test
test_param_string = "wlen"
test_param_in_params = True
test_param_position = -1
test_param_label = trs.choose("Wavelength", "Longitud de onda")

# Sorting and labelling data series
sorting_function = [lambda l : vu.sort_by_number(l, test_param_position)]*2
series_label = [lambda s : trs.choose(r"Vacuum", r"Vacío") + 
                rf" $\lambda$ = {vu.find_numbers(s)[test_param_position]:.0f} nm",
                lambda s : trs.choose("Water", "Agua") + 
                rf" $\lambda$ = {vu.find_numbers(s)[test_param_position]:.0f} nm"]
series_must = [""]*2 # leave "" per default
series_mustnt = ["Old"]*2 # leave "" per default

# Scattering plot options
plot_title_ending = trs.choose("Au 60 nm sphere", "esfera de Au de 60 nm")
series_legend = ["Vacuum", "Water"]
series_colors = [plab.cm.Reds, plab.cm.Blues]
series_linestyles = ["solid"]*2
plot_make_big = False
plot_file = lambda n : os.path.join(home, "DataAnalysis/Field/NPMonoch/AuSphere/VacWatField/TestNorm/Norm" + n)

#%% LOAD DATA

path = []
file = []
series = []
files_line = []
files_plane = []
files_line_norm = []
results_line = []
results_plane = []
results_line_norm = []
t_line = []
x_line = []
t_plane = []
y_plane = []
z_plane = []
t_line_norm = []
x_line_norm = []
params = []

for f, sf, sm, smn in zip(folder, sorting_function, series_must, series_mustnt):

    path.append( os.path.join(home, f) )
    file.append( lambda f, s : os.path.join(path[-1], f, s) )
    
    series.append( os.listdir(path[-1]) )
    series[-1] = vu.filter_by_string_must(series[-1], sm)
    if smn!="": series[-1] = vu.filter_by_string_must(series[-1], smn, False)
    series[-1] = sf(series[-1])
    
    files_line.append( [] )
    files_plane.append( [] )
    for s in series[-1]:
        files_line[-1].append( h5.File(file[-1](s, "Field-Lines.h5"), "r") )
        files_plane[-1].append( h5.File(file[-1](s, "Field-Planes.h5"), "r") )
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
            p["RAM"] = np.array(f["RAM"])
            p["SWAP"] = np.array(f["SWAP"])
            p["elapsed_time"] = np.array(f["ElapsedTime"])
        except FileNotFoundError:
            f = h5.File(file[-1](s, "RAM.h5"))
            p["RAM"] = np.array(f["RAM"])
            p["SWAP"] = np.array(f["SWAP"])
            p["elapsed_time"] = p["elapsed"]
            del p["elapsed"]
    del s, p

    files_line_norm.append( [] )
    for s in series[-1]:
        files_line_norm[-1].append( h5.File(file[-1](s, "Field-Lines-Norm.h5"), "r") )
    del s
    
    results_line_norm.append( [fi["Ez"] for fi in files_line_norm[-1]] )
    
    t_line_norm.append( [np.asarray(fi["T"]) for fi in files_line_norm[-1]] )
    x_line_norm.append( [np.asarray(fi["X"]) for fi in files_line_norm[-1]] )

    # files_line_norm.append( [] )
    # results_line_norm.append( [] )
    # t_line_norm.append( [] )
    # x_line_norm.append( [] )
    # for s, p in zip(series[-1], params[-1]):
    #     # try:
    #     if p["normalize"]:
    #         files_line_norm[-1].append( h5.File(file[-1](s, "Field-Lines-Norm.h5"), "r") )
    #         results_line_norm[-1].append( files_line_norm[-1][-1]["Ez"] )
    #         t_line_norm[-1].append( files_line_norm[-1][-1]["T"] )
    #         x_line_norm[-1].append( files_line_norm[-1][-1]["X"] )
    #         # file_line_norm.close()
    #         # del file_line_norm
    #     #     else:
    #     #         results_line_norm[-1].append( None )
    #     #         t_line_norm[-1].append( None )
    #     #         x_line_norm[-1].append( None )
    #     # except:
    #     #         results_line_norm[-1].append( None )
    #     #         t_line_norm[-1].append( None )
    #     #         x_line_norm[-1].append( None )

del f, sf, sm, smn
            
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
norm_until_time = []
sysname = []
for p in params:
    from_um_factor.append( [pi["from_um_factor"] for pi in p] )
    resolution.append( [pi["resolution"] for pi in p] )
    r.append( [pi["r"] for pi in p] )
    material.append( [pi["material"] for pi in p])
    index.append( [pi["submerged_index"] for pi in p] )
    wlen.append( [pi["wlen"] for pi in p] )
    cell_width.append( [pi["cell_width"] for pi in p] )
    pml_width.append( [pi["pml_width"] for pi in p] )
    source_center.append( [pi["source_center"] for pi in p] )
    period_plane.append( [pi["period_plane"] for pi in p] )
    period_line.append( [pi["period_line"] for pi in p] )
    until_time.append( [pi["until_time"] for pi in p] )
    sysname.append( [pi["sysname"] for pi in p] )
    try:
        paper.append( [pi["paper"] for pi in p] )
    except:
        paper.append( ["R" for pi in p] )
    norm_until_time.append( [] )
    for pi in p:
        # try:
        if pi["normalize"]:
                # try:
                    # norm_until_time[-1].append( p["norm_until_time"] )
                # except:
            aux_until_time = pi["norm_time_period_factor"] * pi["wlen"]
            aux_until_time = vu.round_to_multiple(aux_until_time, 
                                                  pi["courant"]/pi["resolution"], 
                                                  round_up=True)
            norm_until_time[-1].append( aux_until_time )
            # else:
                # norm_until_time[-1].append( None )
        # except:
            # norm_until_time[-1].append( None )
# del p

t_line_norm = [[np.arange(0, files_line_norm[i][j]["Ez"].shape[-1]) * period_line[i][j] for j in range(len(series[i]))] for i in range(len(series))]

if test_param_in_params:
    test_param = [[p[test_param_string] for p in par] for par in params]
else:
    test_param = [[vu.find_numbers(s)[test_param_position] for s in ser] for ser in series]

minor_division = [[fum * 1e3 / res for fum, res in zip(frum, reso)] for frum, reso in zip(from_um_factor, resolution)]
width_points = [[int(p["cell_width"] * p["resolution"]) for p in par] for par in params] 
grid_points = [[wp**3 for wp in wpoints] for wpoints in width_points]
memory_B = [[2 * 12 * gp * 32 for p, gp in zip(par, gpoints)] for par, gpoints in zip(params, grid_points)] # in bytes

#%% POSITION RECONSTRUCTION

t_line_index = [[vma.def_index_function(t_line[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
x_line_index = [[vma.def_index_function(x_line[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

t_plane_index = [[vma.def_index_function(t_plane[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
y_plane_index = [[vma.def_index_function(y_plane[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
z_plane_index = [[vma.def_index_function(z_plane[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

t_line_norm_index = [[vma.def_index_function(t_line_norm[i][j]) for j in range(len(series[i]))] for i in range(len(series))]
x_line_norm_index = [[vma.def_index_function(x_line_norm[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

x_line_cropped = [[x_line[i][j][:x_line_index[i][j](cell_width[i][j]/2 - pml_width[i][j])] for j in range(len(series[i]))] for i in range(len(series))]
x_line_cropped = [[x_line_cropped[i][j][x_line_index[i][j](-cell_width[i][j]/2 + pml_width[i][j]):] for j in range(len(series[i]))] for i in range(len(series))]

y_plane_cropped = [[y_plane[i][j][:y_plane_index[i][j](cell_width[i][j]/2 - pml_width[i][j])] for j in range(len(series[i]))] for i in range(len(series))]
y_plane_cropped = [[y_plane_cropped[i][j][y_plane_index[i][j](-cell_width[i][j]/2 + pml_width[i][j]):] for j in range(len(series[i]))] for i in range(len(series))]

z_plane_cropped = [[z_plane[i][j][:z_plane_index[i][j](cell_width[i][j]/2 - pml_width[i][j])] for j in range(len(series[i]))] for i in range(len(series))]
z_plane_cropped = [[z_plane_cropped[i][j][z_plane_index[i][j](-cell_width[i][j]/2 + pml_width[i][j]):] for j in range(len(series[i]))] for i in range(len(series))]

x_line_norm_cropped = [[x_line_norm[i][j][:x_line_norm_index[i][j](cell_width[i][j]/2 - pml_width[i][j])] for j in range(len(series[i]))] for i in range(len(series))]
x_line_norm_cropped = [[x_line_norm_cropped[i][j][x_line_norm_index[i][j](-cell_width[i][j]/2 + pml_width[i][j]):] for j in range(len(series[i]))] for i in range(len(series))]

#%% DATA EXTRACTION

source_results = [[vma.get_source_from_line(results_line_norm[i][j], x_line_norm_index[i][j], source_center[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

period_results = [[vma.get_period_from_source(source_results[i][j], t_line_norm[i][j], peaks_sep_sensitivity=.15) for j in range(len(series[i]))] for i in range(len(series))]

amplitude_results = [[vma.get_amplitude_from_source(source_results[i][j], peaks_sep_sensitivity=.15) for j in range(len(series[i]))] for i in range(len(series))]

# source_results = [[vma.get_source_from_line(results_line[i][j], x_line_index[i][j], source_center[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

# period_results = [[vma.get_period_from_source(source_results[i][j], t_line[i][j], peaks_sep_sensitivity=.15) for j in range(len(series[i]))] for i in range(len(series))]

# amplitude_results = [[vma.get_amplitude_from_source(source_results[i][j], peaks_sep_sensitivity=.15) for j in range(len(series[i]))] for i in range(len(series))]

zprofile_results = [[vma.get_zprofile_from_plane(results_plane[i][j], y_plane_index[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

zprofile_integral = [[vma.integrate_field_zprofile(zprofile_results[i][j], z_plane_index[i][j],
                                                   cell_width[i][j], pml_width[i][j], period_plane[i][j]) for j in range(len(series[i]))] for i in range(len(series))]

zprofile_max = [[vma.find_zpeaks_zprofile(zprofile_results[i][j], z_plane_index[i][j],
                                               r[i][j], cell_width[i][j], pml_width[i][j])[1] for j in range(len(series[i]))] for i in range(len(series))]

field_peaks_zprofile = [[vma.get_single_field_peak_from_yzplanes(results_plane[i][j],
                                                              y_plane_index[i][j], z_plane_index[i][j], 
                                                              r[i][j], cell_width[i][j], pml_width[i][j])[1] for j in range(len(series[i]))] for i in range(len(series))]

field_peaks_plane = [[vma.get_single_field_peak_from_yzplanes(results_plane[i][j],
                                                           y_plane_index[i][j], z_plane_index[i][j], 
                                                           r[i][j], cell_width[i][j], pml_width[i][j])[2] for j in range(len(series[i]))] for i in range(len(series))]

#%% SHOW SOURCE AND FOURIER

# colors = [["C0"], ["C4"], ["C3"]]
colors = [sc(np.linspace(0,1,len(s)+2))[2:] 
          for sc, s in zip(series_colors, series)]

plt.figure()
plt.suptitle(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
             plot_title_ending)

lines = []
for i in range(len(series)):
    for j in range(len(series[i])):
        # l, = plt.plot(t_line[i][j]/period_results[i][j], 
        l, = plt.plot(t_line_norm[i][j]/period_results[i][j], 
                      source_results[i][j]/amplitude_results[i][j], 
                      label=series_label[i](series[i][j]),
                      color=colors[i][j])
        lines.append(l)
plt.xlabel(trs.choose("Time in multiples of period", "Tiempo en múltiplos del período"))
plt.ylabel(trs.choose(r"Normalized Electric Field $E_z(y=z=0)$ [a.u.]",
                      r"Campo eléctrico normalizado $E_z(y=z=0)$ [u.a.]"))
plt.legend(ncol=2)

plt.savefig(plot_file("Source.png"))
        
fourier = [[np.abs(np.fft.rfft(source_results[i][j])) for j in range(len(series[i]))] for i in range(len(series))]
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
plt.ylabel(trs.choose(r"Electric Field Fourier $\mathcal{F}\;(E_z)$ [u.a.]",
                      r"Transformada del campo eléctrico $\mathcal{F}\;(E_z)$ [u.a.]"))
plt.legend(ncol=2)

# plt.annotate(trs.choose(f"Maximum at {fourier_max_wlen:.2f} nm",
#                         f"Máximo en {fourier_max_wlen:.2f} nm"),
#              (5, 5), xycoords='figure points')

plt.savefig(plot_file("SourceFFT.png"))

plt.xlim([350, 850])
        
plt.savefig(plot_file("SourceFFTZoom.png"))

#%% SHOW RESONANCE OSCILLATIONS

colors = [sc(np.linspace(0,1,len(s)+2))[2:] 
          for sc, s in zip(series_colors, series)]

plt.figure()        
plt.suptitle(trs.choose('Monochromatic source on ', 'Fuente monocromática sobre ') + 
             plot_title_ending)

for i in range(len(series)):
    for j in range(len(series[i])):
            plt.plot(t_line[i][j]/period_results[i][j], 
                     zprofile_integral[i][j], color=colors[i][j],
                     label=series_label[i](series[i][j]))
plt.xlabel(trs.choose("Time [MPu]", "Tiempo [uMP]"))
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
                     zprofile_max[i][j]/amplitude_results[i][j], 
                     color=colors[i][j],
                     label=series_label[i](series[i][j]))
# plt.plot(t_plane, zprofile_max)
plt.xlabel(trs.choose("Time [MPu]", "Tiempo [uMP]"))
plt.ylabel(trs.choose(r"Electric Field Maximum $max[ E_z(z) ]$ [a.u.]",
                      r"Máximo del campo eléctrico $max[ E_z(z) ]$ [u.a.]"))

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
plt.ylabel(trs.choose(r"Normalized Electric Field $E_z(y=z=0)$ [a.u.]",
                      r"Campo eléctrico normalizado $E_z(y=z=0)$ [u.a.]"))
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
# l_cm = []
# l_ku = []
for i in range(len(series)):
    for j in range(len(series[i])):
        plt.plot(z_plane_cropped[i][j], field_peaks_zprofile[i][j],
                 label=series_label[i](series[i][j]),
                 color=colors[i][j])
plt.xlabel(trs.choose("Position Z [nm]", "Position Z [nm]"))
plt.ylabel(trs.choose(r"Normalized Electric Field $E_z(y=z=0)$ [a.u.]",
                      r"Campo eléctrico normalizado $E_z(y=z=0)$ [u.a.]"))
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
                         field_peaks_zprofile[i][j] / amplitude_results[i][j],
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
plt.ylabel(trs.choose(r"Normalized Electric Field $E_z(y=z=0)$ [a.u.]",
                      r"Campo eléctrico normalizado $E_z(y=z=0)$ [u.a.]"))

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
        l_mp, = axes[i][j].plot(z_plane_cropped[i][j] * from_um_factor[i][j] * 1e3, 
                                field_peaks_zprofile[i][j] / amplitude_results[i][j],
                                label=series_label[i](series[i][j]),
                                color=colors[i][j])
        l_cm, = axes[i][j].plot(rvec[i][j][:,-1] * from_um_factor[i][j] * 1e3, 
                                np.abs(zprofile_cm_theory[i][j]),
                                color=colors[i][j], linestyle="dashed")
        l_ku, = axes[i][j].plot(rvec[i][j][:,-1]  * from_um_factor[i][j] * 1e3, 
                                np.abs(zprofile_ku_theory[i][j]),
                                color=colors[i][j], linestyle="dotted")
        xlims = axes[i][j].get_xlim()
        axes[i][j].axhline(0, color="k", linewidth=1)
        axes[i][j].set_xlim(xlims)
        axes[i][j].set_title(series_label[i](series[i][j]))
        # axes[i][j].set_ylim(*lims)
        if i==len(series)-1:
            axes[i][j].set_xlabel(trs.choose("Position Z [nm]", "Posición Z [nm]"))
        if j==0:
            axes[i][j].set_ylabel(trs.choose(r"Normalized Electric Field $E_z(y=z=0)$ [a.u.]",
                                              r"Campo eléctrico normalizado $E_z(y=z=0)$ [u.a.]"))
        if i==len(series)-1 and j==len(series[i])-1:
            axes[i][j].legend([l_mp, l_cm, l_ku], 
                              trs.choose(["MEEP Data", "CM Theory", "Ku Theory"],
                                         ["Data MEEP", "Teoría CM", "Teoría Ku"]),
                              loc="upper center")
        # axes[i][j].grid(True, axis="y")

plt.savefig(plot_file("FieldProfileAllSubplots.png"))

#%% PLOT MAXIMUM INSTENSIFICATION FIELD

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
lims = [np.min([np.min([np.min(field_peaks_plane[i][j] / amplitude_results[i][j]) for j in range(len(series[i]))]) for i in range(len(series))]),
        np.max([np.max([np.max(field_peaks_plane[i][j] / amplitude_results[i][j]) for j in range(len(series[i]))]) for i in range(len(series))])]
lims = max([abs(l) for l in lims])
lims = [-lims, lims]

for i in range(len(series)):
    for j in range(len(series[i])):
        axes[i][j].imshow(field_peaks_plane[i][j].T / amplitude_results[i][j], 
                          cmap='RdBu', #interpolation='spline36', 
                          vmin=lims[0], vmax=lims[1],
                          extent=[min(y_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3,
                                  max(y_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3,
                                  min(z_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3,
                                  max(z_plane_cropped[i][j]) * from_um_factor[i][j] * 1e3])
        if i == len(series)-1 and j == 0:
            axes[i][j].set_xlabel(trs.choose("Position Y [nm]", "Posición Y [nm]"))
            axes[i][j].set_ylabel(trs.choose("Position Z [nm]", "Posición Z [nm]"))
        axes[i][j].set_title(series_label[i](series[i][j]))
plt.savefig(plot_file("FieldPlaneAll.png"))

#%% MAKE PROFILE GIF

maxnframes = 300

# What should be parameters
iref = -1
jref = 0
nframes = min(maxnframes, np.max([[zprofile_results[i][j].shape[-1] for j in range(len(series[i]))] for i in range(len(series))]))
nframes_step = int(round(zprofile_results[iref][jref].shape[-1] / nframes))
label_function = lambda k : trs.choose('Time: {:.1f} of period',
                                       'Tiempo: {:.1f} del período').format(t_line[-1][0][k]/period_results[-1][0])

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
lims = [np.min([np.min([np.min(zprofile_results[i][j] / amplitude_results[i][j]) for j in range(len(series[i]))]) for i in range(len(series))]),
        np.max([np.max([np.max(zprofile_results[i][j] / amplitude_results[i][j]) for j in range(len(series[i]))]) for i in range(len(series))])]

def draw_pml_box(i, j):
    axes[i][j].axvline((-cell_width[i][j]/2 + pml_width[i][j]) * from_um_factor[i][j] * 1e3, 
                        linestyle=":", color='k')
    axes[i][j].axvline((cell_width[i][j]/2 - pml_width[i][j]) * from_um_factor[i][j] * 1e3, 
                       linestyle=":", color='k')
    axes[i][j].axhline(0, color='k', linewidth=1)

def make_pic_line(k):
    tk = t_line[iref][jref][k * nframes_step] / period_results[iref][jref]
    for i in range(len(series)):
        for j in range(len(series[i])):
            kij = np.argmin(np.abs(t_line[i][j] / period_results[i][j] - tk))
            axes[i][j].clear()
            axes[i][j].plot(z_plane[i][j] * from_um_factor[i][j] * 1e3, 
                            zprofile_results[i][j][:, kij] / amplitude_results[i][j],
                            color=colors[i][j])
            axes[i][j].axhline(zprofile_max[i][j][kij] / amplitude_results[i][j], 
                               linestyle="dashed", color=colors[i][j])
            axes[i][j].set_ylim(*lims)
            if i==len(series)-1 and j==0:
                axes[i][j].text(-.2, -.2, label_function(kij), 
                                transform=axes[i][j].transAxes)
            draw_pml_box(i, j)
            if i==len(series)-1:
                axes[i][j].set_xlabel(trs.choose("Distance Z [nm]", "Distancia Z [nm]"))
            if j==0:
                axes[i][j].set_ylabel(trs.choose(r"Electric Field Profile $E_z|_{z=0}$ [a.u.]",
                                                 r"Perfil del campo eléctrico $E_z|_{z=0}$ [u.a.]"))
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

#%% MAKE CROPPED PROFILE GIF

zprofile_cropped_results = [[vma.crop_single_field_zprofile(zprofile_results[i][j], 
                                                     z_plane_index[i][j], 
                                                     cell_width[i][j], 
                                                     pml_width[i][j])
                               for j in range(len(series[i]))] for i in range(len(series))]

maxnframes = 300

# What should be parameters
iref = -1
jref = 0
nframes = min(maxnframes, np.max([[zprofile_results[i][j].shape[-1] for j in range(len(series[i]))] for i in range(len(series))]))
nframes_step = int(round(zprofile_results[iref][jref].shape[-1] / nframes))
label_function = lambda k : trs.choose('Time: {:.1f} of period',
                                       'Tiempo: {:.1f} del período').format(t_line[-1][0][k]/period_results[-1][0])

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
lims = [np.min([np.min([np.min(zprofile_cropped_results[i][j] / amplitude_results[i][j]) for j in range(len(series[i]))]) for i in range(len(series))]),
        np.max([np.max([np.max(zprofile_cropped_results[i][j] / amplitude_results[i][j]) for j in range(len(series[i]))]) for i in range(len(series))])]

def draw_pml_box(i, j):
    axes[i][j].axvline((-cell_width[i][j]/2 + pml_width[i][j]) * from_um_factor[i][j] * 1e3, 
                        linestyle=":", color='k')
    axes[i][j].axvline((cell_width[i][j]/2 - pml_width[i][j]) * from_um_factor[i][j] * 1e3, 
                       linestyle=":", color='k')
    axes[i][j].axhline(0, color='k', linewidth=1)

def make_pic_line(k):
    tk = t_line[iref][jref][k * nframes_step] / period_results[iref][jref]
    for i in range(len(series)):
        for j in range(len(series[i])):
            kij = np.argmin(np.abs(t_line[i][j] / period_results[i][j] - tk))
            axes[i][j].clear()
            axes[i][j].plot(z_plane_cropped[i][j] * from_um_factor[i][j] * 1e3, 
                            zprofile_cropped_results[i][j][:, kij] / amplitude_results[i][j],
                            color=colors[i][j])
            axes[i][j].axhline(zprofile_max[i][j][kij] / amplitude_results[i][j], 
                               linestyle="dashed", color=colors[i][j])
            axes[i][j].set_ylim(*lims)
            if i==len(series)-1 and j==0:
                axes[i][j].text(-.2, -.2, label_function(kij), 
                                transform=axes[i][j].transAxes)
            draw_pml_box(i, j)
            if i==len(series)-1:
                axes[i][j].set_xlabel(trs.choose("Distance Z [nm]", "Distancia Z [nm]"))
            if j==0:
                axes[i][j].set_ylabel(trs.choose(r"Electric Field Profile $E_z|_{z=0}$ [a.u.]",
                                                 r"Perfil del campo eléctrico $E_z|_{z=0}$ [u.a.]"))
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

make_gif_line(plot_file("CroppedAxisZ"))
plt.close(fig)


# # What should be parameters
# nframes_step = 1
# all_nframes = [int(rp.shape[0]/nframes_step) for rp in results_plane_in_fase]
# nframes = max(all_nframes)
# jmax = np.where(np.asarray(all_nframes)==nframes)[0][0]
# call_index = lambda i, j : int(i * all_nframes[j] / nframes)
# call_series = lambda i, j : z_profile_in_fase[j][i,:]
# label_function = lambda i : 'Tiempo: {:.1f}'.format(i * period_plane[jmax] / wlen[jmax]) + ' T'

# # Animation base
# fig = plt.figure()
# ax = plt.subplot()
# lims_y = (min([np.min(zp) for zp in z_profile_in_fase]), 
#           max([np.max(zp) for zp in z_profile_in_fase]))
# shape = [call_series(0,i).shape[0] for i in range(n)]
# color = ['#1f77b4', '#ff7f0e', '#2ca02c']

# def draw_pml_box(j):
#     plt.vlines((-cell_width[j]/2 + pml_width[j])/cell_width[j], *lims_y,
#                linestyle=":", color=color[j])
#     plt.vlines((cell_width[j]/2 - pml_width[j])/cell_width[j], *lims_y,
#                linestyle=":", color=color[j])

# def make_pic_line(i, max_vals, max_index):
#     ax.clear()
#     for j in range(n):
#         data = call_series(call_index(i,j),j)
#         search = [space_to_index(-cell_width[j]/2 + pml_width[j], j),
#                   space_to_index(cell_width[j]/2 - pml_width[j], j)]
#         max_data = max(data[search[0]:search[-1]])
#         if max_data>max_vals[j]:
#             max_index = j
#             max_vals[j] = max_data
#         plt.plot(np.linspace(-0.5, 0.5, shape[j]), 
#                  call_series(call_index(i,j),j))
#         draw_pml_box(j)
#         plt.hlines(max_vals[j], -.5, .5, color=color[j], linewidth=.5)
#     ax.set_xlim(-.5, .5)
#     ax.set_ylim(*lims_y)
#     ax.text(-.12, -.1, label_function(i), transform=ax.transAxes)
#     plt.legend(["$\lambda$ = {} nm, $D$ = {:.0f} nm".format(wl*10, cw*10) for wl, cw in zip(wlen, cell_width)])
#     # plt.hlines(max_vals[max_index], -.5, .5, color=color[max_index], linewidth=.5)
#     plt.hlines(0, -.5, .5, color='k', linewidth=.5)
#     plt.xlabel("Distancia en z (D)")
#     plt.ylabel("Campo eléctrico Ez (u.a.)")
#     plt.show()
#     return ax, max_vals, max_index

# def make_gif_line(gif_filename):
#     max_vals = [0,0,0]
#     max_index = 0
#     pics = []
#     for i in range(nframes):
#         ax, max_vals, max_index = make_pic_line(i*nframes_step, 
#                                                max_vals, max_index)
#         plt.savefig('temp_pic.png') 
#         pics.append(mim.imread('temp_pic.png')) 
#         print(str(i+1)+'/'+str(nframes))
#     mim.mimsave(gif_filename+'.gif', pics, fps=8)
#     os.remove('temp_pic.png')
#     print('Saved gif')

# make_gif_line(file("AxisZ2"))
# plt.close(fig)
# # del fig, ax, lims, nframes_step, nframes, call_series, label_functio

# #%% CROPPPED PLANE IN FASE GIF

# # What should be parameters
# nframes_step = 1
# all_nframes = [int(rp.shape[0]/nframes_step) for rp in in_results_plane_in_fase]
# nframes = max(all_nframes)
# jmax = np.where(np.asarray(all_nframes)==nframes)[0][0]
# call_index = lambda i, j : int(i * all_nframes[j] / nframes)
# call_series = lambda i, j : in_results_plane_in_fase[j][i,:,:].T
# label_function = lambda i : 'Tiempo: {:.1f}'.format(i * period_plane[jmax] / wlen[jmax]) + ' T'

# # Animation base
# fig = plt.figure(figsize=(n*6.4, 6.4))
# axes = fig.subplots(ncols=n)
# lims = lambda j : (np.min(in_results_plane_in_fase[j]),
#                    np.max(in_results_plane_in_fase[j]))

# def make_pic_plane(i):
#     for j in range(n):
#         axes[j].clear()
#         axes[j].imshow(call_series(call_index(i, j), j), 
#                        interpolation='spline36', cmap='RdBu', 
#                        vmin=lims(j)[0], vmax=lims(j)[1])
#         axes[j].set_xlabel("Distancia en y (u.a.)")
#         axes[j].set_ylabel("Distancia en z (u.a.)")
#         axes[j].set_title("$\lambda$={:.0f} nm".format(wlen[j]*10))
#     axes[0].text(-.1, -.105, label_function(i), transform=axes[0].transAxes)
#     plt.show()
#     return axes

# def make_gif_plane(gif_filename):
#     pics = []
#     for i in range(nframes):
#         axes = make_pic_plane(i*nframes_step)
#         plt.savefig('temp_pic.png') 
#         pics.append(mim.imread('temp_pic.png')) 
#         print(str(i+1)+'/'+str(nframes))
#     mim.mimsave(gif_filename+'.gif', pics, fps=5)
#     os.remove('temp_pic.png')
#     print('Saved gif')

# make_gif_plane(file("CroppedPlaneX=0"))
# plt.close(fig)
# # del fig, ax, lims, nframes_step, nframes, call_series, label_function

# #%% CROPPED Z LINES GIF

# # What should be parameters
# nframes_step = 1
# all_nframes = [int(rp.shape[0]/nframes_step) for rp in in_results_plane_in_fase]
# nframes = max(all_nframes)
# jmax = np.where(np.asarray(all_nframes)==nframes)[0][0]
# call_index = lambda i, j : int(i * all_nframes[j] / nframes)
# call_series = lambda i, j : in_z_profile_in_fase[j][i,:]
# label_function = lambda i : 'Tiempo: {:.1f}'.format(i * period_plane[jmax] / wlen[jmax]) + ' T'

# # Animation base
# fig = plt.figure()
# ax = plt.subplot()
# lims_y = (min([np.min(zp) for zp in in_z_profile_in_fase]), 
#           max([np.max(zp) for zp in in_z_profile_in_fase]))
# shape = [call_series(0,i).shape[0] for i in range(n)]
# color = ['#1f77b4', '#ff7f0e', '#2ca02c']

# def make_pic_line(i, max_vals, max_index):
#     ax.clear()
#     for j in range(n):
#         data = call_series(call_index(i,j),j)
#         max_data = max(data)
#         if max_data>max_vals[j]:
#             max_index = j
#             max_vals[j] = max_data
#         plt.plot(np.linspace(-0.5, 0.5, shape[j]), 
#                  call_series(call_index(i,j),j))
#         plt.hlines(max_vals[j], -.5, .5, color=color[j], linewidth=.5)
#     ax.set_xlim(-.5, .5)
#     ax.set_ylim(*lims_y)
#     ax.text(-.12, -.1, label_function(i), transform=ax.transAxes)
#     plt.legend(["$\lambda$ = {} nm, $D$ = {:.0f} nm".format(wl*10, cw*10) for wl, cw in zip(wlen, cell_width)])
#     # plt.hlines(max_vals[max_index], -.5, .5, color=color[max_index], linewidth=.5)
#     plt.hlines(0, -.5, .5, color='k', linewidth=.5)
#     plt.xlabel("Distancia en z (D)")
#     plt.ylabel("Campo eléctrico Ez (u.a.)")
#     plt.show()
#     return ax, max_vals, max_index

# def make_gif_line(gif_filename):
#     max_vals = [0,0,0]
#     max_index = 0
#     pics = []
#     for i in range(nframes):
#         ax, max_vals, max_index = make_pic_line(i*nframes_step, 
#                                                max_vals, max_index)
#         plt.savefig('temp_pic.png') 
#         pics.append(mim.imread('temp_pic.png')) 
#         print(str(i+1)+'/'+str(nframes))
#     mim.mimsave(gif_filename+'.gif', pics, fps=8)
#     os.remove('temp_pic.png')
#     print('Saved gif')

# make_gif_line(file("CroppedAxisZ"))
# plt.close(fig)