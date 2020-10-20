#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 15:29:32 2020

@author: vall
"""

# Field of 120nm-diameter Au sphere given a visible monochromatic incident wave.

import imageio as mim
import h5py as h5
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time
from v_materials import import_medium
import v_save as vs

#%% PARAMETERS

# Units: 10 nm as length unit
from_um_factor = 10e-3 # Conversion of 1 μm to my length unit (=10nm/1μm)
resolution = 4 # >=8 pixels per smallest wavelength, i.e. np.floor(8/wvl_min)

# Au sphere
r = 6  # Radius of sphere: 60 nm
medium = import_medium("Au", from_um_factor) # Medium of sphere: gold (Au)

# Frequency and wavelength
wlen = 75 # 570 nm

# Space configuration
pml_width = 0.5 * wlen
air_width = 2*r

# Field Measurements
period_line = 1
period_plane = 1
after_cell_run_time = 10*wlen

# Computation time
enlapsed = []

# Saving directories
series = "2020101801"
folder = "AuSphereFieldResults"
home = "/home/vall/Documents/Thesis/ThesisPython/"

#%% GENERAL GEOMETRY SETUP

air_width = air_width - air_width%(1/resolution)

pml_width = pml_width - pml_width%(1/resolution)
pml_layers = [mp.PML(thickness=pml_width)]

symmetries = [mp.Mirror(mp.Y), 
              mp.Mirror(mp.Z, phase=-1)]
# Cause of symmetry, two mirror planes reduce cell size to 1/4

cell_width = 2 * (pml_width + air_width + r)
cell_width = cell_width - cell_width%(1/resolution)
cell_size = mp.Vector3(cell_width, cell_width, cell_width)

source_center = -0.5*cell_width + pml_width
print("Resto Source Center: {}".format(source_center%(1/resolution)))
sources = [mp.Source(mp.ContinuousSource(wavelength=wlen, 
                                         is_integrated=True),
                     center=mp.Vector3(source_center),
                     size=mp.Vector3(0, cell_width, cell_width),
                     component=mp.Ez)]
# Ez-polarized monochromatic planewave 
# (its size parameter fills the entire cell in 2d)
# >> The planewave source extends into the PML 
# ==> is_integrated=True must be specified

geometry = [mp.Sphere(material=medium,
                      center=mp.Vector3(),
                      radius=r)]
# Au sphere with frequency-dependant characteristics imported from Meep.

path = os.path.join(home, folder, "{}Results".format(series))
if not os.path.isdir(path): vs.new_dir(path)
file = lambda f : os.path.join(path, f)

#%% SAVE GET FUNCTIONS

def get_line(sim):
    return sim.get_array(
        center=mp.Vector3(), 
        size=mp.Vector3(cell_width), 
        component=mp.Ez)

def get_plane(sim):
    return sim.get_array(
        center=mp.Vector3(), 
        size=mp.Vector3(0, cell_width, cell_width), 
        component=mp.Ez)


#%% INITIALIZE

sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    boundary_layers=pml_layers,
                    sources=sources,
                    symmetries=symmetries,
                    geometry=geometry)

temp = time()
sim.init_sim()
enlapsed.append(time()-temp)

#%% DEFINE SAVE STEP FUNCTIONS

f, save_line = vs.save_slice_generator(sim, get_line, file("Lines.h5"), "Ez")
g, save_plane = vs.save_slice_generator(sim, get_plane, file("Planes.h5"), "Ez")

to_do_while_running = [mp.at_every(period_line, save_line),
                       mp.at_every(period_plane, save_plane)]

#%% RUN!

temp = time()
sim.run(*to_do_while_running, until=cell_width+after_cell_run_time)
del f, g
enlapsed.append(time() - temp)

#%% SAVE METADATA

params = dict(
    from_um_factor=from_um_factor,
    resolution=resolution,
    r=r,
    pml_width=pml_width,
    air_width=air_width,
    cell_width=cell_width,
    source_center=source_center,
    wlen=wlen,
    period_line=period_line,
    period_plane=period_plane,
    after_cell_run_time=after_cell_run_time,
    series=series,
    folder=folder, 
    home=home,
    enlapsed=enlapsed
    )

f = h5.File(file("Lines.h5"), "r+")
for a in params: f["Ez"].attrs[a] = params[a]
f.close()
del f

g = h5.File(file("Planes.h5"), "r+")
for a in params: g["Ez"].attrs[a] = params[a]
g.close()
del g

#%% GET READY TO LOAD DATA

f = h5.File(file("Lines.h5"), "r")
results_line = f["Ez"]

g = h5.File(file("Planes.h5"), "r")
results_plane = g["Ez"]

#%% SHOW SOURCE

index_to_space = lambda i : i/resolution - cell_width/2
space_to_index = lambda x : round(resolution * (x + cell_width/2))

source_field = np.asarray(results_line[:, space_to_index(source_center)])

plt.figure()
plt.plot(np.linspace(0, after_cell_run_time+cell_width, len(source_field)),
         source_field)
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

#%% MAKE PLANE GIF

# What should be parameters
nframes_step = 1
nframes = int(results_plane.shape[0]/nframes_step)
call_series = lambda i : results_plane[i].T
label_function = lambda i : 'Tiempo: {:.1f} u.a.'.format(i*period_plane)

# Animation base
fig = plt.figure()
ax = plt.subplot()
lims = (np.min(results_plane), np.max(results_plane))

def draw_pml_box():
    plt.hlines(space_to_index(-cell_width/2 + pml_width), 
               space_to_index(-cell_width/2 + pml_width), 
               space_to_index(cell_width/2 - pml_width),
               linestyle=":", color='k')
    plt.hlines(space_to_index(cell_width/2 - pml_width), 
               space_to_index(-cell_width/2 + pml_width), 
               space_to_index(cell_width/2 - pml_width),
               linestyle=":", color='k')
    plt.vlines(space_to_index(-cell_width/2 + pml_width), 
               space_to_index(-cell_width/2 + pml_width), 
               space_to_index(cell_width/2 - pml_width),
               linestyle=":", color='k')
    plt.vlines(space_to_index(cell_width/2 - pml_width), 
               space_to_index(-cell_width/2 + pml_width), 
               space_to_index(cell_width/2 - pml_width),
               linestyle=":", color='k')

def make_pic_plane(i):
    ax.clear()
    plt.imshow(call_series(i), interpolation='spline36', cmap='RdBu', 
               vmin=lims[0], vmax=lims[1])
    ax.text(-.1, -.105, label_function(i), transform=ax.transAxes)
    draw_pml_box()
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
# del fig, ax, lims, nframes_step, nframes, call_series, label_function

#%% GET Z LINE ACROSS SPHERE

index_to_space = lambda i : i/resolution - cell_width/2
space_to_index = lambda x : round(resolution * (x + cell_width/2))

z_profile = np.asarray(results_plane[:, space_to_index(0), :])

#%% MAKE LINES GIF

# What should be parameters
nframes_step = 1
nframes = int(z_profile.shape[0]/nframes_step)
call_series = lambda i : z_profile[i, :]
label_function = lambda i : 'Tiempo: {:.1f} u.a.'.format(i*period_plane)

# Animation base
fig = plt.figure()
ax = plt.subplot()
lims = (np.min(z_profile), np.max(z_profile))
shape = call_series(0).shape[0]

def make_pic_line(i):
    ax.clear()
    plt.plot(np.linspace(-cell_width/2, cell_width/2, shape), 
             call_series(i))
    ax.set_ylim(*lims)
    ax.text(-.12, -.1, label_function(i), transform=ax.transAxes)
    plt.xlabel("Distancia en z (u.a.)")
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

make_gif_line(file("AxisZ"))
plt.close(fig)
# del fig, ax, lims, nframes_step, nframes, call_series, label_function

#%% MAKE LINES GIF

# What should be parameters
nframes_step = 1
nframes = int(results_line.shape[0]/nframes_step)
call_series = lambda i : results_line[i]
label_function = lambda i : 'Tiempo: {:.1f} u.a.'.format(i*period_line)

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

def make_pic_line(i):
    ax.clear()
    plt.plot(np.linspace(-cell_width/2, cell_width/2, shape), 
             call_series(i))
    ax.set_ylim(*lims)
    ax.text(-.12, -.1, label_function(i), transform=ax.transAxes)
    draw_pml_box()
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
# del fig, ax, lims, nframes_step, nframes, call_series, label_function


#%%

f.close()
g.close()

sim.reset_meep()
