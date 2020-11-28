#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 15:46:56 2020

@author: vall
"""

import imageio as mim
import h5py as h5
import meep as mp
import matplotlib.pyplot as plt
import numpy as np
import os
from time import time
import v_save as vs

#%% PARAMETERS

cell_width = 12
resolution = 10

wlen = 1
is_integrated = True # Reccomended true if planewave. Default: False
source_width = 0 # Temporal width of smoothing. Default: 0 ==> No smoothing
source_slowness = 3 # Parameter of how slow is the smoothing.
source_length = cell_width
source_size = [0, source_length, 0]

pml_width = round(.38 * wlen, 2) 
# For 1, 0.1 source parameters, 0.4 to be sure but 0.39 doesn't look bad.
# So: 35% or 40% of larger wavelength apparently works OK.

run_time = 16

period_planes = 1/10

planes_center = [0,0,0]

xz_plane_size = [cell_width, 0, cell_width]
xy_plane_size = [cell_width, cell_width, 0]

series = "TryingEy"
folder = "CurrentResults"
home = r"/home/vall/Documents/Thesis/ThesisPython"

parameters = dict(
    wlen=wlen,
    is_integrated=is_integrated,
    source_width=source_width,
    source_slowness=source_slowness,
    source_length=source_length,
    source_size=source_size,
    pml_width=pml_width,
    cell_width=cell_width,
    resolution=resolution,
    period_planes=period_planes,
    planes_center=planes_center,
    xz_plane_size=xz_plane_size,
    xy_plane_size=xy_plane_size,
    run_time=run_time,
    series=series,
    folder=folder,
    home=home
    )

#%% LAYOUT CONFIGURATION

cell_size = mp.Vector3(cell_width, cell_width, cell_width)

boundary_layers = [mp.PML(thickness=pml_width)]

source_center = -0.5*cell_width + pml_width
sources = [mp.Source(mp.ContinuousSource(wavelength=wlen, 
                                         is_integrated=is_integrated),
                     center=mp.Vector3(source_center),
                     size=mp.Vector3(*source_size),
                     component=mp.Ey)]

symmetries = [mp.Mirror(mp.Y), mp.Mirror(mp.Z, phase=-1)]

enlapsed = []

path = os.path.join(home, folder, "{}Results".format(series))
if not os.path.isdir(path): vs.new_dir(path)
file = lambda f : os.path.join(path, f)

#%% OPTION 1: SAVE GET FUNCTIONS

def get_xz_plane(sim):
    return sim.get_array(
        center=mp.Vector3(*planes_center), 
        size=mp.Vector3(*xz_plane_size), 
        component=mp.Ez)

def get_xy_plane(sim):
    return sim.get_array(
        center=mp.Vector3(*planes_center), 
        size=mp.Vector3(*xy_plane_size), 
        component=mp.Ez)

#%% INITIALIZE

sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    boundary_layers=boundary_layers,
                    sources=sources,
                    symmetries=symmetries)

sim.init_sim()

#%% DEFINE SAVE STEP FUNCTIONS

f, save_xz_plane = vs.save_slice_generator(sim, file("XZPlane.h5"), 
                                           "Ez", get_xz_plane)
g, save_xy_plane = vs.save_slice_generator(sim, file("XYPlane.h5"), 
                                           "Ez", get_xy_plane)

to_do_while_running = [mp.at_every(period_planes, save_xz_plane, save_xy_plane)]

#%% RUN!

temp = time()
sim.run(*to_do_while_running, until=run_time)
del f, g
enlapsed.append(time() - temp)

#%% SAVE METADATA

f = h5.File(file("XZPlane.h5"), "r+")
for a in parameters: f["Ez"].attrs[a] = parameters[a]
f.close()
del f

g = h5.File(file("XYPlane.h5"), "r+")
for a in parameters: g["Ez"].attrs[a] = parameters[a]
g.close()
del g

#%% GET READY TO LOAD DATA

f = h5.File(file("XZPlane.h5"), "r")
xz_planes = f["Ez"]

g = h5.File(file("XYPlane.h5"), "r")
xy_planes = g["Ez"]

#%% MAKE XZ PLANE GIF

# What should be parameters
nframes_step = 1
nframes = int(xz_planes.shape[0]/nframes_step)
call_series = lambda i : xz_planes[i]
label_function = lambda i : 'Tiempo: {:.1f} u.a.'.format(i*period_planes)

# Animation base
fig = plt.figure()
ax = plt.subplot()
lims = (np.min(xz_planes), np.max(xz_planes))

def make_pic_plane(i):
    ax.clear()
    plt.imshow(call_series(i), interpolation='spline36', cmap='RdBu', 
               vmin=lims[0], vmax=lims[1])
    ax.text(-.1, -.105, label_function(i), transform=ax.transAxes)
    plt.show()
    plt.xlabel("Distancia en x (u.a.)")
    plt.ylabel("Distancia en z (u.a.)")
    return ax

def make_gif_plane(gif_filename):
    pics = []
    for i in range(nframes):
        ax = make_pic_plane(i*nframes_step)
        plt.savefig('temp_pic.png') 
        pics.append(mim.imread('temp_pic.png')) 
        print(str(i+1)+'/'+str(nframes))
    mim.mimsave(gif_filename, pics, fps=5)
    os.remove('temp_pic.png')
    print('Saved gif')

make_gif_plane(file("XZPlanes.gif"))
plt.close(fig)
del fig, ax, lims, nframes_step, nframes, call_series, label_function

#%% MAKE XY PLANE GIF

# What should be parameters
nframes_step = 1
nframes = int(xy_planes.shape[0]/nframes_step)
call_series = lambda i : xy_planes[i]
label_function = lambda i : 'Tiempo: {:.1f} u.a.'.format(i*period_planes)

# Animation base
fig = plt.figure()
ax = plt.subplot()
lims = (np.min(xy_planes), np.max(xy_planes))

def make_pic_plane(i):
    ax.clear()
    plt.imshow(call_series(i), interpolation='spline36', cmap='RdBu', 
               vmin=lims[0], vmax=lims[1])
    ax.text(-.1, -.105, label_function(i), transform=ax.transAxes)
    plt.show()
    plt.xlabel("Distancia en y (u.a.)")
    plt.ylabel("Distancia en x (u.a.)")
    return ax

def make_gif_plane(gif_filename):
    pics = []
    for i in range(nframes):
        ax = make_pic_plane(i*nframes_step)
        plt.savefig('temp_pic.png') 
        pics.append(mim.imread('temp_pic.png')) 
        print(str(i+1)+'/'+str(nframes))
    mim.mimsave(gif_filename, pics, fps=5)
    os.remove('temp_pic.png')
    print('Saved gif')

make_gif_plane(file("XYPlanes.gif"))
plt.close(fig)
del fig, ax, lims, nframes_step, nframes, call_series, label_function

#%%

f.close()
g.close()

sim.reset_meep()