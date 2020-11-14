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
source_width = 5 # Temporal width of smoothing. Default: 0 ==> No smoothing
source_slowness = 3 # Parameter of how slow is the smoothing.
source_size = [0, cell_width, cell_width]

pml_width = round(.38 * wlen, 2) 
# For 1, 0.1 source parameters, 0.4 to be sure but 0.39 doesn't look bad.
# So: 35% or 40% of larger wavelength apparently works OK.

period_line = 1/10
period_plane = 1

run_time = 20

plane_center = [0,0,0]
line_center = [0,0,0]

plane_size = [0, cell_width, cell_width]
line_size = [cell_width, 0, 0]

series = "RunTime{}TWidth{}".format(run_time, source_width)
folder = "PlanewaveResults"
home = r"/home/vall/Documents/Thesis/ThesisPython"

parameters = dict(
    wlen=wlen,
    is_integrated=is_integrated,
    source_width=source_width,
    source_slowness=source_slowness,
    source_size=source_size,
    pml_width=pml_width,
    cell_width=cell_width,
    resolution=resolution,
    period_line=period_line,
    period_plane=period_plane,
    plane_center=plane_center,
    line_center=line_center,
    plane_size=plane_size,
    line_size=line_size,
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
                     component=mp.Ez)]

symmetries = [mp.Mirror(mp.Y), mp.Mirror(mp.Z, phase=-1)]

enlapsed = []

path = os.path.join(home, folder, "{}Results".format(series))
if not os.path.isdir(path): vs.new_dir(path)
file = lambda f : os.path.join(path, f)

#%% OPTION 1: SAVE GET FUNCTIONS

def get_line(sim):
    return sim.get_array(
        center=mp.Vector3(*line_center), 
        size=mp.Vector3(*line_size), 
        component=mp.Ez)

def get_plane(sim):
    return sim.get_array(
        center=mp.Vector3(*plane_center), 
        size=mp.Vector3(*plane_size), 
        component=mp.Ez)


#%% INITIALIZE

sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    boundary_layers=boundary_layers,
                    sources=sources,
                    symmetries=symmetries)

sim.init_sim()

#%% DEFINE SAVE STEP FUNCTIONS

f, save_line = vs.save_slice_generator(sim, get_line, file("Lines.h5"), "Ez")
g, save_plane = vs.save_slice_generator(sim, get_plane, file("Planes.h5"), "Ez")

to_do_while_running = [mp.at_every(period_line, save_line),
                       mp.at_every(period_plane, save_plane)]

#%% RUN!

temp = time()
sim.run(*to_do_while_running, until=run_time)
del f, g
enlapsed.append(time() - temp)

#%% SAVE METADATA

f = h5.File(file("Lines.h5"), "r+")
for a in parameters: f["Ez"].attrs[a] = parameters[a]
f.close()
del f

g = h5.File(file("Planes.h5"), "r+")
for a in parameters: g["Ez"].attrs[a] = parameters[a]
g.close()
del g

#%% GET READY TO LOAD DATA

f = h5.File(file("Lines.h5"), "r")
results_line = f["Ez"]

g = h5.File(file("Planes.h5"), "r")
results_plane = g["Ez"]

#%% MAKE LINES GIF

# What should be parameters
nframes_step = 3
nframes = int(results_line.shape[0]/nframes_step)
call_series = lambda i : results_line[i]
label_function = lambda i : 'Tiempo: {:.1f} u.a.'.format(i*period_line)

# Animation base
fig = plt.figure()
ax = plt.subplot()
lims = (np.min(results_line), np.max(results_line))

def make_pic_line(i):
    ax.clear()
    plt.plot(np.linspace(-cell_width/2, cell_width/2, int(resolution*cell_width)), 
             call_series(i))
    ax.set_ylim(*lims)
    ax.text(-.12, -.1, label_function(i), transform=ax.transAxes)
    plt.xlabel("Distancia en x (u.a.)")
    plt.ylabel("Campo eléctrico Ez (u.a.)")
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

make_gif_line(file("AxisX"))
plt.close(fig)
del fig, ax, lims, nframes_step, nframes, call_series, label_function

#%% SHOW SOURCE

index_to_space = lambda i : i/resolution - cell_width/2
space_to_index = lambda x : round(resolution * (x + cell_width/2))

source_field = np.asarray(results_line[:, space_to_index(source_center)])

plt.figure()
plt.plot(np.linspace(0,run_time,len(source_field)),source_field)
plt.xlabel("Tiempo (u.a.)")
plt.ylabel("Campo eléctrico Ez (u.a.)")

plt.savefig(file("Source.png"))

#%% MAKE FOURIER FOR SOURCE

index_to_space = lambda i : i/resolution - cell_width/2
space_to_index = lambda x : round(resolution * (x + cell_width/2))

source_field = np.asarray(results_line[:, space_to_index(source_center)])

fourier = np.abs(np.fft.rfft(source_field))
fourier_freq = np.fft.rfftfreq(len(source_field), d=period_line)

plt.figure()
plt.plot(fourier_freq, fourier, 'k', linewidth=3)
plt.xlabel("Frequency (u.a.)")
plt.ylabel("Transformada del campo eléctrico Ez (u.a.)")

plt.savefig(file("SourceFFT.png"))

plt.xlim(1/wlen-.3, 1/wlen+.3)
plt.show()
plt.savefig(file("SourceFFTZoom.png"))

#%% SHOW ONE PLANE

i = 10
label_function = lambda i : 'Tiempo: {:.1f} u.a.'.format(i*period_plane)

plt.figure(dpi=150)
ax = plt.subplot()
plt.imshow(results_plane[i,:,:], interpolation='spline36', cmap='RdBu')
ax.text(-.1, -.105, label_function(i), transform=ax.transAxes)
plt.xlabel("Distancia en y (u.a.)")
plt.ylabel("Distancia en z (u.a.)")

plt.savefig(file("PlaneX=0Index{}".format(i)))

del i, label_function

#%% MAKE PLANE GIF

# What should be parameters
nframes_step = 1
nframes = int(results_plane.shape[0]/nframes_step)
call_series = lambda i : results_plane[i]
label_function = lambda i : 'Tiempo: {:.1f} u.a.'.format(i*period_plane)

# Animation base
fig = plt.figure()
ax = plt.subplot()
lims = (np.min(results_plane), np.max(results_plane))

def make_pic_plane(i):
    ax.clear()
    plt.imshow(call_series(i), interpolation='spline36', cmap='RdBu', 
               vmin=lims[0], vmax=lims[1])
    ax.text(-.1, -.105, label_function(i), transform=ax.transAxes)
    plt.show()
    plt.xlabel("Distancia en y (u.a.)")
    plt.ylabel("Distancia en z (u.a.)")
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

make_gif_plane(file("PlaneX=0"))
plt.close(fig)
del fig, ax, lims, nframes_step, nframes, call_series, label_function

#%%

f.close()
g.close()

sim.reset_meep()