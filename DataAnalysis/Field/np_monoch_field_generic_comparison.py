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
import PyMieScatt as ps
from scipy.signal import find_peaks
import v_analysis as va
from v_materials import import_medium
import v_save as vs
import v_utilities as vu

#%% PARAMETERS

# Saving directories
folder = ["Field/NPMonoch/TestPeriods/Vacuum", 
          "Field/NPMonoch/TestPeriods/Water"]
home = vs.get_home()

# Parameter for the test
test_param_string = "wlen"
test_param_in_params = True
test_param_position = -1
test_param_label = "Wavelength"

# Sorting and labelling data series
sorting_function = [lambda l : vu.sort_by_number(l, test_param_position)]*2
series_label = [lambda s : f"Vacuum $\lambda$ = {vu.find_numbers(s)[test_param_position]:.2f} nm",
                lambda s : f"Water $\lambda$ = {vu.find_numbers(s)[test_param_position]:.2f} nm"]
series_must = [""]*2 # leave "" per default
series_mustnt = [""]*2 # leave "" per default

# Scattering plot options
plot_title = "Au 103 nm sphere in water"
series_legend = ["Vacuum", "Water"]
series_colors = [plab.cm.Reds, plab.cm.Blues]
series_linestyles = ["solid"]*2
plot_make_big = False
plot_file = lambda n : os.path.join(home, "DataAnalysis/Field/Periods" + n)

#%% PARAMETERS

# # Directories
# data_series = ["AllWaterField60WLen405", 
#                "AllWaterField60WLen532",
#                "AllWaterField60WLen642"]
# data_folder = "AuMieMediums/AllWaterField"
# home = vs.get_home()

# n = len(data_series)

# # Saving directory
# series = "AuSphereField60"
# folder = "AuMieMediums/AllWaterField"

# #%% FILES MANAGEMENT

# data_file = lambda f,s : os.path.join(home, data_folder, s, f)

# path = os.path.join(home, folder, "{}Results".format(series))
# if not os.path.isdir(path): vs.new_dir(path)
# file = lambda f : os.path.join(path, f)

#%% LOAD DATA

path = []
file = []
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
    
    results_line.append( [fi["Ez"] for fi in files_line[-1]] )
    results_plane.append( [fi["Ez"] for fi in files_plane[-1]] )
    params.append( [dict(fi["Ez"].attrs) for fi in files_line[-1]] )
    
    t_line.append( [np.asarray(fi["T"]) for fi in files_line[-1]] )
    x_line.append( [np.asarray(fi["X"]) for fi in files_line[-1]] )
    
    t_plane.append( [np.asarray(fi["T"]) for fi in files_plane[-1]] )
    y_plane.append( [np.asarray(fi["Y"]) for fi in files_plane[-1]] )
    z_plane.append( [np.asarray(fi["Z"]) for fi in files_plane[-1]] )
    
from_um_factor = []
resolution = []
r = []
paper = []
index = []
cell_width = []
pml_width = []
source_center = []
period_plane = []
period_line = []
until_time = []
sysname = []
for p in params:
    from_um_factor.append( [pi["from_um_factor"] for pi in p] )
    resolution.append( [pi["resolution"] for pi in p] )
    r.append( [pi["r"] for pi in p] )
    paper.append( [pi["paper"] for pi in p])
    index.append( [pi["submerged_index"] for pi in p] )
    cell_width.append( [pi["cell_width"] for pi in p] )
    pml_width.append( [pi["pml_width"] for pi in p] )
    source_center.append( [pi["source_center"] for pi in p] )
    period_plane.append( [pi["period_plane"] for pi in p] )
    period_line.append( [pi["period_line"] for pi in p] )
    until_time.append( [pi["until_time"] for pi in p] )
    sysname.append( [pi["sysname"] for pi in p] )
    try:
        paper.append( [pi["paper"] for pi in p])
    except:
        paper.append( ["R" for pi in p] )

if test_param_in_params:
    test_param = [[p[test_param_string] for p in par] for par in params]
else:
    test_param = [[vu.find_numbers(s)[test_param_position] for s in ser] for ser in series]

minor_division = [[fum * 1e3 / res for fum, res in zip(frum, reso)] for frum, reso in zip(from_um_factor, resolution)]
width_points = [[int(p["cell_width"] * p["resolution"]) for p in par] for par in params] 
grid_points = [[wp**3 for wp in wpoints] for wpoints in width_points]
memory_B = [[2 * 12 * gp * 32 for p, gp in zip(par, gpoints)] for par, gpoints in zip(params, grid_points)] # in bytes

#%% GENERAL USEFUL FUNCTIONS

t_line_index = [[lambda t0 : np.argmin(np.abs(np.asarray(tl) - t0)) for tl in tlin] for tlin in t_line]
x_line_index = [[lambda x0 : np.argmin(np.abs(np.asarray(xl) - x0)) for xl in xlin] for xlin in x_line]

t_plane_index = [[lambda t0 : np.argmin(np.abs(np.asarray(tp) - t0)) for tp in tpln] for tpln in t_plane]
y_plane_index = [[lambda y0 : np.argmin(np.abs(np.asarray(yp) - y0)) for yp in ypln] for ypln in y_plane]
z_plane_index = [[lambda z0 : np.argmin(np.abs(np.asarray(zp) - z0)) for zp in zpln] for zpln in z_plane]    

#%%

source_results = [[np.asarray(results_line[i][j][x_line_index[i][j](source_center[i][j]), :]) for j in range(len(series[i]))] for i in range(len(series))]

def get_period_from_source(source, i, j):
    last_stable_periods = 5
    peaks = find_peaks(source_results[i][j])[0]
    
    keep_periods_from = 0
    periods = np.array(t_line[i][j][peaks[1:]] - t_line[i][j][peaks[:-1]])
    last_periods = periods[-last_stable_periods:]
    for k, per in enumerate(periods):
        if np.abs(per - np.mean(last_periods)) / per > .05:
            keep_periods_from = max(keep_periods_from, k+1)
    stable_periods = periods[keep_periods_from:]
    return np.mean(stable_periods)

def get_amplitude_from_source(source, i, j):
    # cut_points = 20
    last_stable_periods = 5
    # peaks = find_peaks(source_results[i][j][cut_points:])[0] + cut_points
    
    peaks = find_peaks(source_results[i][j])[0]
    
    keep_periods_from = 0
    heights = source_results[i][j][peaks]
    last_heights = source_results[i][j][peaks[-last_stable_periods:]]
    for k, h in enumerate(heights):
        if np.abs(h - np.mean(last_heights)) / h > .05:
            keep_periods_from = max(keep_periods_from, k+1)
    stable_heights = source_results[i][j][peaks[keep_periods_from:]]
    return np.mean(stable_heights)

period_results = [[get_period_from_source(source_results[i][j], i, j) for j in range(len(series[i]))] for i in range(len(series))] # Meep units

amplitude_results = [[get_amplitude_from_source(source_results[i][j], i, j) for j in range(len(series[i]))] for i in range(len(series))] # Meep units

def crop_field_zyplane(field, i, j):
    cropped = field[: y_plane_index[i][j](cell_width[i][j]/2 - pml_width[i][j]), :]
    cropped = cropped[y_plane_index[i][j](-cell_width[i][j]/2 + pml_width[i][j]) :, :]
    cropped = cropped[:,: z_plane_index[i][j](cell_width[i][j]/2 - pml_width[i][j])]
    cropped = cropped[:, z_plane_index[i][j](-cell_width[i][j]/2 + pml_width[i][j]) :]
    return cropped        
    
def crop_field_zprofile(field, i, j):
    cropped = field[: z_plane_index[i][j](cell_width[i][j]/2 - pml_width[i][j])]
    cropped = cropped[z_plane_index[i][j](-cell_width[i][j]/2 + pml_width[i][j]) :]
    return cropped

def integrate_field_zprofile(field_profile, i, j):
    integral = np.sum(crop_field_zprofile(field_profile, i, j)) * period_plane[i][j]
    return integral

def detect_sign_field_zprofile(field_profile, i, j):
    return -np.sign(field_profile[0])

def find_peaks_field_zprofile(field_profile, i, j):
    sign = detect_sign_field_zprofile(field_profile, i, j)
    peaks = find_peaks(sign * crop_field_zprofile(field_profile, i, j))[0]
    peaks = np.array(peaks) + z_plane_index[i][j](-cell_width[i][j]/2 + pml_width[i][j])
    if len(peaks) > 2: peaks = peaks[[0,-1]]
    try:
        peaks[0] = min(peaks[0], z_plane_index[i][j](-r[i][j]) - 1)
        peaks[1] = max(peaks[1], z_plane_index[i][j](r[i][j]))
        return peaks
    except:
        return (None, None)

def get_max_index_zprofile(field_profile, i, j):
    try: 
        return min(find_peaks_field_zprofile(field_profile)[0],
                   z_plane_index[i][j](-r[i][j]))
    except IndexError:
        return None

def get_max_field_zprofile(field_profile, i, j):
    try: 
        return np.mean( field_profile[ find_peaks_field_zprofile(field_profile, i, j) ] )
    except RuntimeWarning:
        return None

def get_max_resonance_index(zprofile_results, i, j):
    z_resonance_max = np.array([ get_max_field_zprofile(zprof, i, j) for zprof in zprofile_results.T])
    first_index = int(np.argwhere( np.isnan(z_resonance_max) == False )[0])
    t_resonance_max_index = find_peaks(z_resonance_max[first_index:], 
                                       height=(np.max(z_resonance_max[first_index:])/2, None))[0]
    t_resonance_max_index = np.array(t_resonance_max_index) + first_index
    # t_resonance_max = t_plane[t_resonance_max_index]
    # z_resonance_max = z_resonance_max[t_resonance_max_index]
    return t_resonance_max_index #, t_resonance_max, z_resonance_max

source_results = [[np.asarray(results_line[i][j])[x_line_index[i][j](source_center[i][j]), :] for j in range(len(series[i]))] for i in range(len(series))]

zprofile_results = [[np.asarray(results_plane[i][j])[y_plane_index[i][j](0), :, :] for j in range(len(series[i]))] for i in range(len(series))]

zprofile_integral = [[np.array([ integrate_field_zprofile(zprof, i, j) for zprof in zprofile_results[i][j].T]) for j in range(len(series[i]))] for i in range(len(series))]

zprofile_max = [[np.array([ get_max_field_zprofile(zprof, i, j) for zprof in zprofile_results[i][j].T]) for j in range(len(series[i]))] for i in range(len(series))]

resonance_max_index = [[get_max_resonance_index(zprofile_results[i][j], i, j) for j in range(len(series[i]))] for i in range(len(series))]

#%% SHOW SOURCE AND FOURIER

# colors = [["C0"], ["C4"], ["C3"]]
colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

plt.figure()

# plt.title(trs.choose('Monochromatic wave {:.0f} nm on {} Sphere With {:.1f} nm Diameter',
#                      'Onda monocromática {:.0f} nm sobre esfera de {} con diámetro {:.1f} nm'
#                      ).format(wlen*from_um_factor*1e3, material, 2*r*from_um_factor*1e3 ))

lines = []
for i in range(len(series)):
    for j in range(len(series[i])):
        l, = plt.plot(t_line[i][j]/period_results[i][j], 
                      source_results[i][j]/amplitude_results[i][j], 
                      label=f"{i},{j}")
        lines.append(l)
        
        
#%%

plt.xlabel(trs.choose("Time [Mp.u.]", "Tiempo [u.Mp.]"))
plt.ylabel(trs.choose(r"Electric Field $E_z$ [a.u.]",
                      r"Campo eléctrico $E_z$ [u.a.]"))

plt.savefig(plot_file("Source.png"))

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

plt.savefig(plot_file("SourceFFT.png"))

plt.xlim([350, 850])
        
plt.savefig(plot_file("SourceFFTZoom.png"))

#%% CROP SIGNALS TO JUST INSIDE THE BOX

z_profile = [results_plane[j][:, space_to_index(0,j), :] for j in range(n)]

in_results_line = []
in_results_plane = []
in_z_profile = []
for j in range(n):
    in_results_plane.append(results_plane[j][:,
        space_to_index(-cell_width[j]/2 + pml_width[j], j) : space_to_index(cell_width[j]/2 - pml_width[j], j),
        space_to_index(-cell_width[j]/2 + pml_width[j], j) : space_to_index(cell_width[j]/2 - pml_width[j], j)])
    in_results_line.append(results_line[j][:,
        space_to_index(-cell_width[j]/2 + pml_width[j], j) : space_to_index(cell_width[j]/2 - pml_width[j], j)])
    in_z_profile.append(z_profile[j][:, 
        space_to_index(-cell_width[j]/2 + pml_width[j], j) : space_to_index(cell_width[j]/2 - pml_width[j], j)])

in_index_to_space = lambda i, j : i/resolution[j] - (cell_width[j]-2*pml_width[j])/2
in_space_to_index = lambda x, j : round(resolution[j] * (x + (cell_width[j]-2*pml_width[j])/2))

#%% FIND MAXIMUM

max_in_z_profile = []
is_max_in_z_profile = []
for j in range(n):
    this_max = np.argmax( in_z_profile[j][:, in_space_to_index(r[j], j)])
    this_min = np.argmin( in_z_profile[j][:, in_space_to_index(r[j], j)])
    this_abs_max = np.argmax( np.abs( in_z_profile[j][:, in_space_to_index(r[j], j)]) )
    if this_abs_max == this_max:
        is_max_in_z_profile.append( +1 )
    else:
        is_max_in_z_profile.append( -1 )
    max_in_z_profile.append( this_abs_max )
del this_max, this_min

#%% GET THEORY (CLAUSSIUS-MOSOTTI)

rvec = []
for j in range(len(wlen)):
    naux = len(in_z_profile[j][0, :])
    aux = np.zeros((naux, 3))
    aux[:,2] = np.linspace(-cell_width[j]/2 + pml_width[j], 
                           cell_width[j]/2 - pml_width[j], 
                           naux)
    rvec.append(aux)
del aux, naux

E0 = np.array([0,0,1])

in_theory_cm_line = []
in_theory_k_line = []
for j in range(len(wlen)):
    medium = vm.import_medium("Au", from_um_factor[j])
    epsilon = medium.epsilon(1/wlen[j])[0,0]
    # E0 = np.array([0, 0, is_max_in_z_profile[j] * np.real(in_z_profile[j][max_in_z_profile[j], 0])])
    alpha_cm = vt.alpha_Clausius_Mosotti(epsilon, r[j], epsilon_ext=index[j]**2)
    alpha_k = vt.alpha_Kuwata(epsilon, wlen[j], r[j], epsilon_ext=index[j]**2)
    in_theory_cm_line.append(
        np.array([vt.E(epsilon, alpha_cm, E0, rv, r[j], 
                       epsilon_ext=index[j]**2) for rv in rvec[j]])[:,-1])
    in_theory_k_line.append(
        np.array([vt.E(epsilon, alpha_k, E0, rv, r[j],
                       epsilon_ext=index[j]**2) for rv in rvec[j]])[:,-1])

#%% PLOT MAXIMUM INTESIFICATION PROFILE (LINES)

# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
#           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
colors = ['#1f77b4', '#2ca02c', '#d62728']

l_meep = []
l_cm = []
l_k = []
fig = plt.figure()
plt.title("Máxima intensificación del campo eléctrico en agua en dirección de la polarización")
for j in range(n):
    l_meep.append(
        plt.plot(np.linspace(10*(-cell_width[j]/2 + pml_width[j]), 
                         10*(cell_width[j]/2 - pml_width[j]),
                         in_z_profile[j].shape[1]), 
             is_max_in_z_profile[j] * np.real(in_z_profile[j][max_in_z_profile[j], :]),
             colors[j],
             label=f"Meep $\lambda$ = {wlen[j]*from_um_factor[j]*1e3:1.0f} nm")[0])
    l_cm.append(
        plt.plot(np.linspace(10*(-cell_width[j]/2 + pml_width[j]), 
                         10*(cell_width[j]/2 - pml_width[j]),
                         in_z_profile[j].shape[1]), 
             np.real(in_theory_cm_line[j]), colors[j], linestyle='dashed',
             label=f"Clausius-Mosotti Theory $\lambda$ = {wlen[j]*from_um_factor[j]*1e3:1.0f} nm")[0])
    l_k.append(
        plt.plot(np.linspace(10*(-cell_width[j]/2 + pml_width[j]), 
                         10*(cell_width[j]/2 - pml_width[j]),
                         in_z_profile[j].shape[1]), 
             np.real(in_theory_k_line[j]), colors[j], linestyle='dotted',
             label=f"Kuwata Theory $\lambda$ = {wlen[j]*from_um_factor[j]*1e3:1.0f} nm")[0])

legend1 = plt.legend(
    [l_meep[0], l_cm[0], l_k[0]],
    ["Resultados de MEEP", "Teoría con Clausius-Mossotti", "Teoría con Kuwata"],
    loc='upper left')
plt.legend(l_meep, 
           [f"$\lambda$ = {wlen[j]*from_um_factor[j]*1e3:1.0f} nm" for j in range(n)],
           loc='upper right')
plt.gca().add_artist(legend1)
plt.xlabel("Distancia en z [nm])")
plt.ylabel("Campo eléctrico Ez [u.a.]")
fig.set_size_inches([10.03,  4.8 ])

plt.savefig(file("MaxFieldProfile.png"))

#%% PLOT MAXIMUM INTESIFICATION PROFILE (LINES, JUST THE RESULTS)

# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
#           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
colors = ['#1f77b4', '#2ca02c', '#d62728']

l_meep = []
fig = plt.figure()
plt.title("Máxima intensificación del campo en agua en dirección de la polarización")
for j in range(n):
    plt.plot(np.linspace(10*(-cell_width[j]/2 + pml_width[j]), 
                         10*(cell_width[j]/2 - pml_width[j]),
                         in_z_profile[j].shape[1]), 
             is_max_in_z_profile[j] * np.real(in_z_profile[j][max_in_z_profile[j], :]),
             colors[j],
             label=f"Meep $\lambda$ = {wlen[j]*from_um_factor[j]*1e3:1.0f} nm")[0]
plt.legend([f"$\lambda$ = {wlen[j]*from_um_factor[j]*1e3:1.0f} nm" for j in range(n)],
           loc='upper right')
plt.xlabel("Distancia en z [nm])")
plt.ylabel("Campo eléctrico Ez [u.a.]")
# fig.set_size_inches([10.03,  4.8 ])

plt.savefig(file("MaxFieldProfileResults.png"))


#%% PLOT MAXIMUM INTENSIFCATION FIELD (PLANE)

fig = plt.figure(figsize=(n*6.4, 6.4))
axes = fig.subplots(ncols=n)
lims = [np.min([in_results_plane[j][max_in_z_profile[j],:,:] for j in range(n)]),
       np.max([in_results_plane[j][max_in_z_profile[j],:,:] for j in range(n)])]      
lims = max([abs(l) for l in lims])
lims = [-lims, lims]
call_series = lambda j : in_results_plane[j][max_in_z_profile[j],:,:].T


for j in range(n):
    axes[j].imshow(call_series(j), 
                   interpolation='spline36', cmap='RdBu', 
                   vmin=lims[0], vmax=lims[1])
    axes[j].set_xlabel("Distancia en y (u.a.)", fontsize=18)
    axes[j].set_ylabel("Distancia en z (u.a.)", fontsize=18)
    axes[j].set_title("$\lambda$={} nm".format(wlen[j]*10), fontsize=22)
    plt.setp(axes[j].get_xticklabels(), fontsize=16)
    plt.setp(axes[j].get_yticklabels(), fontsize=16)
plt.savefig(file("MaxFieldPlane.png"))

#%% THEORY SCATTERING

medium = vm.import_medium("Au", from_um_factor[0])

wlens = 10*np.linspace(min(wlen), max(wlen), 500)
freqs = 1e3*from_um_factor[0]/wlens
scatt_eff_theory = [ps.MieQ(np.sqrt(medium.epsilon(f)[0,0]*medium.mu(f)[0,0]), 
                            1e3*from_um_factor[0]/f,
                            2*r[0]*1e3*from_um_factor[0],
                            nMedium=index[0], # Refraction Index of Medium
                            asDict=True)['Qsca'] 
                    for f in freqs]

wlen_max = wlens[np.argmax(scatt_eff_theory)]
e_wlen_max = np.mean([wlens[i+1]-wlens[i] for i in range(499)])

#%% PUT FULL SIGNALS IN FASE

source_field = [results_line[j][:, space_to_index(-cell_width[j]/2 + pml_width[j], j)] 
                for j in range(n)]

source_field_in_fase = []
init_index = []
end_index = []
for sf, rl in zip(source_field, results_line):
    peaks = find_peaks(sf)[0]
    period = round(np.mean(np.diff(peaks)))
    init = peaks[1]
    sfslice = sf[init:init+10*period]
    source_field_in_fase.append(sfslice)
    init_index.append(init)
    end_index.append(init+10*period)
del sfslice, period, init

results_plane_in_fase = [rp[i:f,:,:] for rp, i, f in zip(results_plane, 
                                                         init_index, 
                                                         end_index)]
z_profile_in_fase = [results_plane[j][init_index[j]:end_index[j], space_to_index(0,j), :] for j in range(n)]

#%% CROP FULL SIGNALS IN FASE

in_results_plane_in_fase = []
in_z_profile_in_fase = []
for j in range(len(f)):
    in_results_plane_in_fase.append(results_plane_in_fase[j][:,
        space_to_index(-cell_width[j]/2 + pml_width[j], j) : space_to_index(cell_width[j]/2 - pml_width[j], j),
        space_to_index(-cell_width[j]/2 + pml_width[j], j) : space_to_index(cell_width[j]/2 - pml_width[j], j)])
    in_z_profile_in_fase.append(z_profile_in_fase[j][:,
        space_to_index(-cell_width[j]/2 + pml_width[j], j) : space_to_index(cell_width[j]/2 - pml_width[j], j)])

#%% SHOW SOURCE

# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
#           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
colors = ['#1f77b4', '#2ca02c', '#d62728']

plt.figure()
for sfs, c in zip(source_field_in_fase, colors):
    plt.plot(np.linspace(0,10,len(sfs)), sfs, c)
    plt.xlabel("Tiempo ($T=\lambda/c$)")
    plt.ylabel("Campo eléctrico Ez (u.Meep)")
plt.legend(["$\lambda$ = {:.0f} nm".format(wl*10) for wl in wlen])

plt.savefig(file("Source.png"))

#%% PLANE IN FASE GIF

# What should be parameters
nframes_step = 1
all_nframes = [int(rp.shape[0]/nframes_step) for rp in results_plane_in_fase]
nframes = max(all_nframes)
jmax = np.where(np.asarray(all_nframes)==nframes)[0][0]
call_index = lambda i, j : int(i * all_nframes[j] / nframes)
call_series = lambda i, j : results_plane_in_fase[j][i,:,:].T
label_function = lambda i : 'Tiempo: {:.1f}'.format(i * period_plane[jmax] / wlen[jmax]) + ' T'

# Animation base
fig = plt.figure(figsize=(n*6.4, 6.4))
axes = fig.subplots(ncols=n)
lims = lambda j : (np.min(results_plane_in_fase[j]),
                   np.max(results_plane_in_fase[j]))
    
def draw_pml_box(j):
    axes[j].hlines(space_to_index(-cell_width[j]/2 + pml_width[j], j), 
                   space_to_index(-cell_width[j]/2 + pml_width[j], j), 
                   space_to_index(cell_width[j]/2 - pml_width[j], j),
                   linestyle=":", color='k')
    axes[j].hlines(space_to_index(cell_width[j]/2 - pml_width[j], j), 
                   space_to_index(-cell_width[j]/2 + pml_width[j], j), 
                   space_to_index(cell_width[j]/2 - pml_width[j], j),
                   linestyle=":", color='k')
    axes[j].vlines(space_to_index(-cell_width[j]/2 + pml_width[j], j), 
                   space_to_index(-cell_width[j]/2 + pml_width[j], j), 
                   space_to_index(cell_width[j]/2 - pml_width[j], j),
                   linestyle=":", color='k')
    axes[j].vlines(space_to_index(cell_width[j]/2 - pml_width[j], j), 
                   space_to_index(-cell_width[j]/2 + pml_width[j], j), 
                   space_to_index(cell_width[j]/2 - pml_width[j], j),
                   linestyle=":", color='k')

def make_pic_plane(i):
    for j in range(n):
        axes[j].clear()
        axes[j].imshow(call_series(call_index(i, j), j), 
                       interpolation='spline36', cmap='RdBu', 
                       vmin=lims(j)[0], vmax=lims(j)[1])
        axes[j].set_xlabel("Distancia en y (u.a.)")
        axes[j].set_ylabel("Distancia en z (u.a.)")
        axes[j].set_title("$\lambda$={} nm".format(wlen[j]*10))
        draw_pml_box(j)
    axes[0].text(-.1, -.105, label_function(i), transform=axes[0].transAxes)
    plt.show()
    return axes

def make_gif_plane(gif_filename):
    pics = []
    for i in range(nframes):
        axes = make_pic_plane(i*nframes_step)
        plt.savefig('temp_pic.png') 
        pics.append(mim.imread('temp_pic.png')) 
        print(str(i+1)+'/'+str(nframes))
    mim.mimsave(gif_filename+'.gif', pics, fps=5)
    os.remove('temp_pic.png')
    print('Saved gif')

make_gif_plane(file("PlaneX=0"))
plt.close(fig)
# del fig, ax, lims, nframes_step, nframes, call_series, label_function

#%% MAKE Z LINES GIF

# What should be parameters
nframes_step = 1
all_nframes = [int(rp.shape[0]/nframes_step) for rp in results_plane_in_fase]
nframes = max(all_nframes)
jmax = np.where(np.asarray(all_nframes)==nframes)[0][0]
call_index = lambda i, j : int(i * all_nframes[j] / nframes)
call_series = lambda i, j : z_profile_in_fase[j][i,:]
label_function = lambda i : 'Tiempo: {:.1f}'.format(i * period_plane[jmax] / wlen[jmax]) + ' T'

# Animation base
fig = plt.figure()
ax = plt.subplot()
lims_y = (min([np.min(zp) for zp in z_profile_in_fase]), 
          max([np.max(zp) for zp in z_profile_in_fase]))
shape = [call_series(0,i).shape[0] for i in range(n)]
color = ['#1f77b4', '#ff7f0e', '#2ca02c']

def draw_pml_box(j):
    plt.vlines((-cell_width[j]/2 + pml_width[j])/cell_width[j], *lims_y,
               linestyle=":", color=color[j])
    plt.vlines((cell_width[j]/2 - pml_width[j])/cell_width[j], *lims_y,
               linestyle=":", color=color[j])

def make_pic_line(i, max_vals, max_index):
    ax.clear()
    for j in range(n):
        data = call_series(call_index(i,j),j)
        search = [space_to_index(-cell_width[j]/2 + pml_width[j], j),
                  space_to_index(cell_width[j]/2 - pml_width[j], j)]
        max_data = max(data[search[0]:search[-1]])
        if max_data>max_vals[j]:
            max_index = j
            max_vals[j] = max_data
        plt.plot(np.linspace(-0.5, 0.5, shape[j]), 
                 call_series(call_index(i,j),j))
        draw_pml_box(j)
        plt.hlines(max_vals[j], -.5, .5, color=color[j], linewidth=.5)
    ax.set_xlim(-.5, .5)
    ax.set_ylim(*lims_y)
    ax.text(-.12, -.1, label_function(i), transform=ax.transAxes)
    plt.legend(["$\lambda$ = {} nm, $D$ = {:.0f} nm".format(wl*10, cw*10) for wl, cw in zip(wlen, cell_width)])
    # plt.hlines(max_vals[max_index], -.5, .5, color=color[max_index], linewidth=.5)
    plt.hlines(0, -.5, .5, color='k', linewidth=.5)
    plt.xlabel("Distancia en z (D)")
    plt.ylabel("Campo eléctrico Ez (u.a.)")
    plt.show()
    return ax, max_vals, max_index

def make_gif_line(gif_filename):
    max_vals = [0,0,0]
    max_index = 0
    pics = []
    for i in range(nframes):
        ax, max_vals, max_index = make_pic_line(i*nframes_step, 
                                               max_vals, max_index)
        plt.savefig('temp_pic.png') 
        pics.append(mim.imread('temp_pic.png')) 
        print(str(i+1)+'/'+str(nframes))
    mim.mimsave(gif_filename+'.gif', pics, fps=8)
    os.remove('temp_pic.png')
    print('Saved gif')

make_gif_line(file("AxisZ2"))
plt.close(fig)
# del fig, ax, lims, nframes_step, nframes, call_series, label_functio

#%% CROPPPED PLANE IN FASE GIF

# What should be parameters
nframes_step = 1
all_nframes = [int(rp.shape[0]/nframes_step) for rp in in_results_plane_in_fase]
nframes = max(all_nframes)
jmax = np.where(np.asarray(all_nframes)==nframes)[0][0]
call_index = lambda i, j : int(i * all_nframes[j] / nframes)
call_series = lambda i, j : in_results_plane_in_fase[j][i,:,:].T
label_function = lambda i : 'Tiempo: {:.1f}'.format(i * period_plane[jmax] / wlen[jmax]) + ' T'

# Animation base
fig = plt.figure(figsize=(n*6.4, 6.4))
axes = fig.subplots(ncols=n)
lims = lambda j : (np.min(in_results_plane_in_fase[j]),
                   np.max(in_results_plane_in_fase[j]))

def make_pic_plane(i):
    for j in range(n):
        axes[j].clear()
        axes[j].imshow(call_series(call_index(i, j), j), 
                       interpolation='spline36', cmap='RdBu', 
                       vmin=lims(j)[0], vmax=lims(j)[1])
        axes[j].set_xlabel("Distancia en y (u.a.)")
        axes[j].set_ylabel("Distancia en z (u.a.)")
        axes[j].set_title("$\lambda$={:.0f} nm".format(wlen[j]*10))
    axes[0].text(-.1, -.105, label_function(i), transform=axes[0].transAxes)
    plt.show()
    return axes

def make_gif_plane(gif_filename):
    pics = []
    for i in range(nframes):
        axes = make_pic_plane(i*nframes_step)
        plt.savefig('temp_pic.png') 
        pics.append(mim.imread('temp_pic.png')) 
        print(str(i+1)+'/'+str(nframes))
    mim.mimsave(gif_filename+'.gif', pics, fps=5)
    os.remove('temp_pic.png')
    print('Saved gif')

make_gif_plane(file("CroppedPlaneX=0"))
plt.close(fig)
# del fig, ax, lims, nframes_step, nframes, call_series, label_function

#%% CROPPED Z LINES GIF

# What should be parameters
nframes_step = 1
all_nframes = [int(rp.shape[0]/nframes_step) for rp in in_results_plane_in_fase]
nframes = max(all_nframes)
jmax = np.where(np.asarray(all_nframes)==nframes)[0][0]
call_index = lambda i, j : int(i * all_nframes[j] / nframes)
call_series = lambda i, j : in_z_profile_in_fase[j][i,:]
label_function = lambda i : 'Tiempo: {:.1f}'.format(i * period_plane[jmax] / wlen[jmax]) + ' T'

# Animation base
fig = plt.figure()
ax = plt.subplot()
lims_y = (min([np.min(zp) for zp in in_z_profile_in_fase]), 
          max([np.max(zp) for zp in in_z_profile_in_fase]))
shape = [call_series(0,i).shape[0] for i in range(n)]
color = ['#1f77b4', '#ff7f0e', '#2ca02c']

def make_pic_line(i, max_vals, max_index):
    ax.clear()
    for j in range(n):
        data = call_series(call_index(i,j),j)
        max_data = max(data)
        if max_data>max_vals[j]:
            max_index = j
            max_vals[j] = max_data
        plt.plot(np.linspace(-0.5, 0.5, shape[j]), 
                 call_series(call_index(i,j),j))
        plt.hlines(max_vals[j], -.5, .5, color=color[j], linewidth=.5)
    ax.set_xlim(-.5, .5)
    ax.set_ylim(*lims_y)
    ax.text(-.12, -.1, label_function(i), transform=ax.transAxes)
    plt.legend(["$\lambda$ = {} nm, $D$ = {:.0f} nm".format(wl*10, cw*10) for wl, cw in zip(wlen, cell_width)])
    # plt.hlines(max_vals[max_index], -.5, .5, color=color[max_index], linewidth=.5)
    plt.hlines(0, -.5, .5, color='k', linewidth=.5)
    plt.xlabel("Distancia en z (D)")
    plt.ylabel("Campo eléctrico Ez (u.a.)")
    plt.show()
    return ax, max_vals, max_index

def make_gif_line(gif_filename):
    max_vals = [0,0,0]
    max_index = 0
    pics = []
    for i in range(nframes):
        ax, max_vals, max_index = make_pic_line(i*nframes_step, 
                                               max_vals, max_index)
        plt.savefig('temp_pic.png') 
        pics.append(mim.imread('temp_pic.png')) 
        print(str(i+1)+'/'+str(nframes))
    mim.mimsave(gif_filename+'.gif', pics, fps=8)
    os.remove('temp_pic.png')
    print('Saved gif')

make_gif_line(file("CroppedAxisZ"))
plt.close(fig)