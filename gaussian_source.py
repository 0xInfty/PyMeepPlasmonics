#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 14:34:09 2020

@author: luciana, 0xInfty
"""

import imageio as mim
import meep as mp
# from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import norm
import v_save as vs

#%% PARAMETERS

freq_center = 1
freq_width = 0.1

print("Should have dpml aprox {:.2f}".format(.5 / (freq_center-freq_width/2)))
pml_width = .6

cell_width = 12
resolution = 10

period_line = 1/10
period_plane = 1

time_is_after_source = False
run_time = 100

plane_center = [0,0,0]
line_center = [0,0,0]

plane_size = [0, cell_width, cell_width]
line_size = [cell_width, 0, 0]

results_plane = []
results_line = []

def get_slice_plane(sim):
    results_plane.append(sim.get_array(
        center=mp.Vector3(*plane_center), 
        size=mp.Vector3(*plane_size), 
        component=mp.Ez))

def get_slice_line(sim):
    results_line.append(sim.get_array(
        center=mp.Vector3(*line_center), 
        size=mp.Vector3(*line_size), 
        component=mp.Ez))

to_do_while_running = [mp.at_beginning(get_slice_line),
                       # mp.at_beginning(get_slice_plane),
                       mp.at_every(period_plane, get_slice_line)] #,
                       # mp.at_every(period_line, get_slice_plane)]

series = "First"
home = r"/home/vall/Documents/Thesis/ThesisPython/GaussianSource"
path = os.path.join(home, "{}Results".format(series))
prefix = "{}Results".format(series)

#%% LAYOUT

cell_size = mp.Vector3(cell_width, cell_width, cell_width)

boundary_layers = [mp.PML(thickness=pml_width)]

source_center = -0.5*cell_width + pml_width
sources = [mp.Source(mp.GaussianSource(freq_center,
                                       fwidth=freq_width,
                                       is_integrated=True),
                     center=mp.Vector3(source_center),
                     size=mp.Vector3(0, cell_width, cell_width),
                     component=mp.Ez)]

if not os.path.isdir(path):
    vs.new_dir(path)
os.chdir(path)

sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    boundary_layers=boundary_layers,
                    sources=sources)

#%% INITIALIZE

sim.init_sim()

#%% RUN!

if time_is_after_source:
    sim.run(*to_do_while_running, until_after_source=run_time)
else:
    sim.run(*to_do_while_running, until=run_time)

results_line = np.asarray(results_line)
results_plane = np.asarray(results_plane)

#%% SAVE DATA!

footer=dict(
    freq_center=freq_center,
    freq_width=freq_width,
    pml_width=pml_width,
    cell_width=cell_width,
    resolution=resolution,
    period_line=period_line,
    period_plane=period_plane,
    plane_center=plane_center,
    line_center=line_center,
    plane_size=plane_size,
    line_size=line_size,
    time_is_after_source=time_is_after_source,
    run_time=run_time,
    series=series,
    home=home
    )

vs.savetxt(os.path.join(path, "Lines.txt"), results_line, footer=footer)
del footer

#%% SHOW ALL LINES IN COLOR MAP

plt.figure()
plt.imshow(results_line, interpolation='spline36', cmap='RdBu')
plt.xlabel("Distancia en x (int)")
plt.ylabel("Tiempo (int)")
plt.show()

plt.savefig("AxisXColorMap.png")

#%% SHOW ALL LINES

i = 10
label_function = lambda i : 'Tiempo: {:.1f} u.a.'.format(i*period_line)

plt.figure()
ax = plt.subplot()
plt.plot(np.linspace(-cell_width/2, cell_width/2, int(resolution*cell_width)), 
         results_line[i,:,:])
plt.xlabel("Distancia en x (u.a.)")
plt.ylabel("Campo eléctrico Ez (u.a.)")
ax.text(-.12, -.1, label_function(i), transform=ax.transAxes)

plt.savefig("AxisXLineIndex{}.png".format(i))

#%% SHOW ALL LINES

plt.figure()
for i in range(len(results_line)):
    plt.plot(np.linspace(-cell_width/2, cell_width/2, int(resolution*cell_width)), 
             results_line[i,:,:])
# plt.legend(["Tiempo: {} u.a.".format(i) for i in range(results_line.shape[0])],
#            ncol=2)

plt.savefig("AxisXLines.png")

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

make_gif_line("AxisX")

#%% SHOW SOURCE

index_to_space = lambda i : i/resolution - cell_width/2
space_to_index = lambda x : round(resolution * (x + cell_width/2))

source_field = np.asarray(results_line[:, space_to_index(source_center)])

plt.figure()
plt.plot(source_field)
plt.xlabel("Tiempo (u.a.)")
plt.ylabel("Campo eléctrico Ez (u.a.)")

plt.savefig("Source.png")

#%% MAKE FOURIER FOR SOURCE

fourier = np.abs(np.fft.rfft(source_field))
fourier_freq = np.fft.rfftfreq(len(source_field), d=period_line)
source_profile = norm.pdf(fourier_freq, freq_center, freq_width/(2*np.pi))

plt.figure()
plt.plot(fourier_freq, fourier, 'k', linewidth=3)
# plt.plot(fourier_freq, np.exp(-(2*np.pi**2)*(fourier_freq-freq_center)**2/(2*((freq_width)**2)))*max(fourier))
plt.plot(fourier_freq, source_profile*max(fourier)/max(source_profile), 'r', 
         linewidth=.8)
plt.xlabel("Frequency (u.a.)")
plt.ylabel("Transformada del campo eléctrico Ez (u.a.)")

plt.savefig("SourceFFT.png")

#%% SHOW ONE PLANE

i = 10
label_function = lambda i : 'Tiempo: {:.1f} u.a.'.format(i*period_plane)

plt.figure(dpi=150)
plt.imshow(results_plane[i,:,:], interpolation='spline36', cmap='RdBu')
ax.text(-.1, -.105, label_function(i), transform=ax.transAxes)
plt.xlabel("Distancia en y (u.a.)")
plt.ylabel("Distancia en z (u.a.)")

plt.savefig("PlaneX=0Index{}".format(i))

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

make_gif_plane("PlaneX=0")

#%%

sim.reset_meep()