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
import os
import PyMieScatt as ps
from scipy.signal import find_peaks
import v_meep as vm
import v_save as vs
import v_theory as vt

#%% PARAMETERS

# Directories
data_series = ["AllWaterField60WLen405", 
               "AllWaterField60WLen532",
               "AllWaterField60WLen642"]
data_folder = "AuMieMediums/AllWaterField"
home = vs.get_home()

n = len(data_series)

# Saving directory
series = "AuSphereField60"
folder = "AuMieMediums/AllWaterField"


#%% FILES MANAGEMENT

data_file = lambda f,s : os.path.join(home, data_folder, s, f)

path = os.path.join(home, folder, "{}Results".format(series))
if not os.path.isdir(path): vs.new_dir(path)
file = lambda f : os.path.join(path, f)

#%% GET READY TO LOAD DATA

f = [h5.File(data_file("Lines.h5", s), "r") for s in data_series]
g = [h5.File(data_file("Planes.h5", s), "r") for s in data_series]

results_line = [fi["Ez"] for fi in f]
results_plane = [gi["Ez"] for gi in g]

from_um_factor = [rp.attrs["from_um_factor"] for rp in results_plane]
resolution = [rp.attrs["resolution"] for rp in results_plane]
r = [rp.attrs["r"] for rp in results_plane]
cell_width = [rp.attrs["cell_width"] for rp in results_plane]
pml_width = [rp.attrs["pml_width"] for rp in results_plane]
period_plane = [rp.attrs["period_plane"] for rp in results_plane]
wlen = [rp.attrs["wlen"] for rp in results_plane]
index = [rp.attrs["submerged_index"] for rp in results_plane]
air_width = [cell_width[j]/2 - pml_width[j] - r[j] for j in range(n)]

index_to_space = lambda i, j : i/resolution[j] - cell_width[j]/2
space_to_index = lambda x, j : round(resolution[j] * (x + cell_width[j]/2))

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