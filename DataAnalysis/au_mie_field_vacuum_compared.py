#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:45:00 2020

@author: vall
"""

# Field of 120nm-diameter Au sphere given a visible monochromatic incident wave.

import imageio as mim
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
import v_save as vs

#%% PARAMETERS

# Directories
data_series = ["2020101703", "2020101702", "2020101801"]
data_folder = "AuSphereFieldResults"
home = "/home/vall/Documents/Thesis/ThesisPython/"

n = len(data_series)

# Saving directory
series = "2020102001"
folder = "AuSphereComparedResults"

#%% FILES MANAGEMENT

data_file = lambda f,s : os.path.join(home, data_folder, str("{}Results".format(s)), f)

path = os.path.join(home, folder, "{}Results".format(series))
if not os.path.isdir(path): vs.new_dir(path)
file = lambda f : os.path.join(path, f)

#%% GET READY TO LOAD DATA

f = [h5.File(data_file("Lines.h5", s), "r") for s in data_series]
g = [h5.File(data_file("Planes.h5", s), "r") for s in data_series]

results_line = [fi["Ez"] for fi in f]
results_plane = [gi["Ez"] for gi in g]

resolution = [rp.attrs["resolution"] for rp in results_plane]
r = [rp.attrs["r"] for rp in results_plane]
cell_width = [rp.attrs["cell_width"] for rp in results_plane]
pml_width = [rp.attrs["pml_width"] for rp in results_plane]
period_plane = [rp.attrs["period_plane"] for rp in results_plane]
wlen = [rp.attrs["wlen"] for rp in results_plane]
air_width = [cell_width[j]/2 - pml_width[j] - r[j] for j in range(n)]

index_to_space = lambda i, j : i/resolution[j] - cell_width[j]/2
space_to_index = lambda x, j : round(resolution[j] * (x + cell_width[j]/2))

#%% CROP SIGNALS IN FASE

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

#%% SHOW SOURCE

plt.figure()
for sfs in source_field_in_fase:
    plt.plot(np.linspace(0,10,len(sfs)), sfs)
    plt.xlabel("Tiempo ($T=\lambda/c$)")
    plt.ylabel("Campo eléctrico Ez (u.Meep)")
plt.legend(["$\lambda$ = {} nm".format(wl*10) for wl in wlen])

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

make_gif_line(file("AxisZ"))
plt.close(fig)
# del fig, ax, lims, nframes_step, nframes, call_series, label_functio