#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 14:34:09 2020

@author: luciana, 0xInfty
"""

import h5py as h5
import imageio as mim
import meep as mp
# from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import norm
# from scipy.integrate import dblquad
from time import time
import v_save as vs

#%% PARAMETERS

### MEAN PARAMETERS

# Units: 10 nm as length unit
from_um_factor = 10e-3 # Conversion of 1 μm to my length unit (=10nm/1μm)
resolution = 2 # >=8 pixels per smallest wavelength, i.e. np.floor(8/wvl_min)

# Au sphere
r = 6  # Radius of sphere: 60 nm

# Frequency and wavelength
wlen_range = np.array([50,65]) # 500-650 nm range from lowest to highest
nfreq = 50 # Number of frequencies to discretize range
cutoff = 3.2

# Computation time
enlapsed = []
time_factor_cell = 1.2
until_after_sources = False

# Saving data
period_line = 1/10 * min(wlen_range)
period_plane = 1/50 * min(wlen_range)

# Saving directories
series = "Resolution{}NFreq{}".format(resolution, nfreq)
folder = "AuMieSphere/AuMieFieldResults"
home = "/home/vall/Documents/Thesis/ThesisPython/"

### OTHER PARAMETERS

# Frequency and wavelength
freq_range = 1/wlen_range # Hz range in Meep units from highest to lowest
freq_center = np.mean(freq_range)
freq_width = max(freq_range) - min(freq_range)

# Space configuration
pml_width = 0.38 * max(wlen_range)
air_width = r/2 # 0.5 * max(wlen_range)

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
sources = [mp.Source(mp.GaussianSource(freq_center,
                                       fwidth=freq_width,
                                       is_integrated=True,
                                       cutoff=cutoff),
                     center=mp.Vector3(source_center),
                     size=mp.Vector3(0, cell_width, cell_width),
                     component=mp.Ez)]
# Ez-polarized planewave pulse 
# (its size parameter fills the entire cell in 2d)
# >> The planewave source extends into the PML 
# ==> is_integrated=True must be specified

if time_factor_cell is not False:
    until_after_sources = time_factor_cell * cell_width
else:
    if until_after_sources is False:
        raise ValueError("Either time_factor_cell or until_after_sources must be specified")
    time_factor_cell = until_after_sources/cell_width
# Enough time for the pulse to pass through all the cell
# Originally: Aprox 3 periods of lowest frequency, using T=λ/c=λ in Meep units 
# Now: Aprox 3 periods of highest frequency, using T=λ/c=λ in Meep units 

def get_line(sim, line_center, line_size):
    return sim.get_array(
        center=mp.Vector3(*line_center), 
        size=mp.Vector3(*line_size), 
        component=mp.Ez)

def get_plane(sim, plane_center, plane_size):
    return sim.get_array(
        center=mp.Vector3(*plane_center), 
        size=mp.Vector3(*plane_size), 
        component=mp.Ez)

path = os.path.join(home, folder, "{}Results".format(series))
if not os.path.isdir(path): vs.new_dir(path)
file = lambda f : os.path.join(path, f)

#%% INITIALIZE

sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    boundary_layers=pml_layers,
                    sources=sources,
                    k_point=mp.Vector3(),
                    symmetries=symmetries)

temp = time()
sim.init_sim()
enlapsed.append( time() - temp )

#%% DEFINE SAVE STEP FUNCTIONS

f, save_line = vs.save_slice_generator(
    sim, lambda sim: get_line(sim, [0, 0, 0], [cell_width, 0, 0]), 
    file("Lines.h5"), "Ez")
gx1, save_plane_x1 = vs.save_slice_generator(
    sim, lambda sim: get_plane(sim, [-r, 0, 0], [0, cell_width, cell_width]), 
    file("Planes_X1.h5"), "Ez")
gx2, save_plane_x2 = vs.save_slice_generator(
    sim, lambda sim: get_plane(sim, [r, 0, 0], [0, cell_width, cell_width]), 
    file("Planes_X2.h5"), "Ez")
gy1, save_plane_y1 = vs.save_slice_generator(
    sim, lambda sim: get_plane(sim, [0, -r, 0], [cell_width, 0, cell_width]), 
    file("Planes_Y1.h5"), "Ez")
gy2, save_plane_y2 = vs.save_slice_generator(
    sim, lambda sim: get_plane(sim, [0, r, 0], [cell_width, 0, cell_width]), 
    file("Planes_Y2.h5"), "Ez")
gz1, save_plane_z1 = vs.save_slice_generator(
    sim, lambda sim: get_plane(sim, [0, 0, -r], [cell_width, cell_width, 0]), 
    file("Planes_Z1.h5"), "Ez")
gz2, save_plane_z2 = vs.save_slice_generator(
    sim, lambda sim: get_plane(sim, [0, 0, r], [cell_width, cell_width, 0]), 
    file("Planes_Z2.h5"), "Ez")

to_do_while_running = [mp.at_every(period_line, save_line),
                       mp.at_every(period_plane, save_plane_x1),
                       mp.at_every(period_plane, save_plane_x2),
                       mp.at_every(period_plane, save_plane_y1),
                       mp.at_every(period_plane, save_plane_y2),
                       mp.at_every(period_plane, save_plane_z1),
                       mp.at_every(period_plane, save_plane_z2)]

#%% RUN!

temp = time()
sim.run(*to_do_while_running, until_after_sources=until_after_sources)
enlapsed.append( time() - temp )

#%% SAVE METADATA

params = dict(
    from_um_factor=from_um_factor,
    resolution=resolution,
    r=r,
    wlen_range=wlen_range,
    nfreq=nfreq,
    cutoff=cutoff,
    pml_width=pml_width,
    air_width=air_width,
    source_center=source_center,
    enlapsed=enlapsed,
    period_line=period_line,
    period_plane=period_plane,
    series=series,
    folder=folder,
    home=home,
    until_after_sources=until_after_sources,
    time_factor_cell=time_factor_cell,
    )

planes_series = ["X1", "X2", "Y1", "Y2", "Z1", "Z2"]

f = h5.File(file("Lines.h5"), "r+")
for a in params: f["Ez"].attrs[a] = params[a]
f.close()
del f

for s in planes_series:
    g = h5.File(file("Planes_{}.h5".format(s)), "r+")
    for a in params: g["Ez"].attrs[a] = params[a]
    g.close()
del g

#%% GET READY TO LOAD DATA

f = h5.File(file("Lines.h5"), "r")
results_line = f["Ez"]

g = []
results_plane = []
for s in planes_series:
    gi = h5.File(file("Planes_{}.h5".format(s)), "r")
    g.append(gi)
    results_plane.append( gi["Ez"] )
del gi

#%% SHOW ALL LINES IN COLOR MAP

plt.figure()
plt.imshow(results_line, interpolation='spline36', cmap='RdBu')
plt.xlabel("Distancia en x (int)")
plt.ylabel("Tiempo (int)")
plt.show()

plt.savefig(file("AxisXColorMap.png"))

#%% SHOW ONE LINE

i = 10
label_function = lambda i : 'Tiempo: {:.1f} u.a.'.format(i*period_line)

plt.figure()
ax = plt.subplot()
plt.plot(np.linspace(-cell_width/2, cell_width/2, len(results_line[i,:])), 
         results_line[i,:])
plt.xlabel("Distancia en x (u.a.)")
plt.ylabel("Campo eléctrico Ez (u.a.)")
ax.text(-.12, -.1, label_function(i), transform=ax.transAxes)

plt.savefig(file("AxisXLineIndex{}.png".format(i)))

#%% MAKE LINES GIF

# What should be parameters
nframes_step = 3
nframes = int(results_line.shape[0]/nframes_step)
call_series = lambda i : results_line[i,:]
label_function = lambda i : 'Tiempo: {:.1f} u.a.'.format(i*period_line)

# Animation base
fig = plt.figure()
ax = plt.subplot()
lims = (np.min(results_line), np.max(results_line))

def make_pic_line(i):
    ax.clear()
    plt.plot(np.linspace(-cell_width/2, cell_width/2, len(results_line[i,:])), 
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
    fig.close()
    print('Saved gif')

make_gif_line(file("AxisX"))

#%% SHOW SOURCE

index_to_space = lambda i : i/resolution - cell_width/2
space_to_index = lambda x : round(resolution * (x + cell_width/2))

source_field = np.asarray(results_line[:, space_to_index(source_center)])

plt.figure()
plt.plot(source_field)
plt.xlabel("Tiempo (u.a.)")
plt.ylabel("Campo eléctrico Ez (u.a.)")

plt.savefig(file("Source.png"))

#%% MAKE FOURIER FOR SOURCE

index_to_space = lambda i : i/resolution - cell_width/2
space_to_index = lambda x : round(resolution * (x + cell_width/2))

source_field = np.asarray(results_line[:, space_to_index(source_center)])

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

plt.savefig(file("SourceFFT.png"))

plt.xlim(freq_center-2*freq_width, freq_center+2*freq_width)
plt.show()
plt.savefig(file("SourceFFTZoom.png"))

#%% SHOW ONE PLANE

i = 10
label_function = lambda i : 'Tiempo: {:.1f} u.a.'.format(i*period_plane)
call_series = lambda i, d : d[i,:,:]
n = results_plane[0].shape[1]

index_to_space = lambda i : i * cell_width/n - cell_width/2
space_to_index = lambda x : round(n * (x + cell_width/2) / cell_width)

zlims = (np.min(results_plane), np.max(results_plane))
planes_labels = ["Flujo {} [u.a.]".format(s) for s in planes_series]

fig, ax = plt.subplots(3, 2)
fig.subplots_adjust(hspace=0.25)
for a, h in zip(np.reshape(ax, 6), planes_labels):
    a.set_title(h.split(" ")[1].split("0")[0])

for p, a in zip(results_plane, np.reshape(ax, 6)):
    a.imshow(call_series(i, p).T, interpolation='spline36', cmap='RdBu', 
             vmin=zlims[0], vmax=zlims[1])
    a.axis("off")

plt.savefig(file("PlanesIndex{}".format(i)))

#%% FOURIER FOR ONE DOT

n = results_plane[0].shape[1]
planes_labels = ["Flujo {} [u.a]".format(l) for l in planes_series]

index_to_space = lambda i : i * cell_width/n - cell_width/2
space_to_index = lambda x : round(n * (x + cell_width/2) / cell_width)

results_plane_crop = [p[:, 
                        space_to_index(-r):space_to_index(r), 
                        space_to_index(-r):space_to_index(r)]
                      for p in results_plane]

n_crop = results_plane_crop[0].shape[1]
n_t = results_plane_crop[0].shape[0]

index_to_space_crop = lambda i : i * 2*r/n_crop - r
space_to_index_crop = lambda x : round(n_crop * (x + r) / (2*r))

i, j = [space_to_index_crop(0), space_to_index_crop(0)]
fourier_freq_planes = np.fft.rfftfreq(n_t, d=period_plane)
fourier_lines_dots = [np.abs(np.fft.rfft(p[:,i,j])) for p in results_plane_crop]

fig, ax = plt.subplots(3, 2)
fig.subplots_adjust(hspace=0.25)
for a, h in zip(np.reshape(ax, 6), planes_labels):
    a.set_title(h.split(" ")[1].split("0")[0])

for d, a in zip(fourier_lines_dots, np.reshape(ax, 6)):
    a.plot(1e3*from_um_factor/fourier_freq_planes, d)
    a.set_xlim(*wlen_range*1e3*from_um_factor)
    a.xaxis.set_visible(False)
    # a.yaxis.set_visible(False)
ax[-1,0].xaxis.set_visible(True)
ax[-1,1].xaxis.set_visible(True)
ax[-1,0].set_xlabel("Wavelength [nm]")
ax[-1,1].set_xlabel("Wavelength [nm]")

plt.savefig(file("DotsFluxFourier.png"))

#%% FOURIER FOR WHOLE PLANES

n = results_plane[0].shape[1]
planes_labels = ["Flujo {} [u.a]".format(l) for l in planes_series]

index_to_space = lambda i : i * cell_width/n - cell_width/2
space_to_index = lambda x : round(n * (x + cell_width/2) / cell_width)

results_plane_crop = [p[:, 
                        space_to_index(-r):space_to_index(r), 
                        space_to_index(-r):space_to_index(r)]
                      for p in results_plane]

n_crop = results_plane_crop[0].shape[1]
n_t = results_plane[0].shape[0]
space_index = [0,0]
    
fourier_lines_planes = [np.ndarray((n_t//2 + 1, n_crop, n_crop)) for i in range(6)]
fourier_freq_planes = np.fft.rfftfreq(n_t, d=period_plane)
for p, fp in zip(results_plane_crop, fourier_lines_planes):
    for i in range(n_crop):
        for j in range(n_crop):
            fp[:, i, j] = np.abs(np.fft.rfft(p[:, i, j]))

# fourier_planes = []
# for fp in fourier_lines_planes:
#     data = [dblquad(fpd, -r, r, -r, r) for fpd in fp]
#     fourier_planes.append(data)

fourier_planes = [np.array([np.sum(fpd)/(n_crop)**2 for fpd in fp]) for fp in fourier_lines_planes]

fig, ax = plt.subplots(3, 2)
fig.subplots_adjust(hspace=0.25)
for a, h in zip(np.reshape(ax, 6), planes_labels):
    a.set_title(h.split(" ")[1].split("0")[0])

for d, a in zip(fourier_planes, np.reshape(ax, 6)):
    a.plot(1e3*from_um_factor/fourier_freq_planes, d)
    a.set_xlim(*wlen_range*1e3*from_um_factor)
    a.xaxis.set_visible(False)
    # a.yaxis.set_visible(False)
ax[-1,0].xaxis.set_visible(True)
ax[-1,1].xaxis.set_visible(True)
ax[-1,0].set_xlabel("Wavelength [nm]")
ax[-1,1].set_xlabel("Wavelength [nm]")

plt.savefig(file("PlanesFluxFourier.png"))

#%% FOURIER FOR ONE DOT IN TIME

k = 1200

n = results_plane[0].shape[1]
planes_labels = ["Flujo {} [u.a]".format(l) for l in planes_series]

index_to_space = lambda i : i * cell_width/n - cell_width/2
space_to_index = lambda x : round(n * (x + cell_width/2) / cell_width)

results_plane_crop = [p[0:k, 
                        space_to_index(-r):space_to_index(r), 
                        space_to_index(-r):space_to_index(r)]
                      for p in results_plane]

n_crop = results_plane_crop[0].shape[1]
n_t = results_plane_crop[0].shape[0]

index_to_space_crop = lambda i : i * 2*r/n_crop - r
space_to_index_crop = lambda x : round(n_crop * (x + r) / (2*r))

i, j = [space_to_index_crop(0), space_to_index_crop(0)]
fourier_freq_planes = np.fft.rfftfreq(k, d=period_plane)
fourier_lines_dots = [np.abs(np.fft.rfft(p[:,i,j])) for p in results_plane_crop]

fig, ax = plt.subplots(3, 2)
fig.subplots_adjust(hspace=0.25)
for a, h in zip(np.reshape(ax, 6), planes_labels):
    a.set_title(h.split(" ")[1].split("0")[0])

for d, a in zip(fourier_lines_dots, np.reshape(ax, 6)):
    a.plot(1e3*from_um_factor/fourier_freq_planes, d)
    a.set_xlim(*wlen_range*1e3*from_um_factor)
    a.xaxis.set_visible(False)
    a.yaxis.set_visible(False)
ax[-1,0].xaxis.set_visible(True)
ax[-1,1].xaxis.set_visible(True)
ax[-1,0].set_xlabel("Wavelength [nm]")
ax[-1,1].set_xlabel("Wavelength [nm]")

plt.savefig(file("DotsFluxFourier-{}-{}.png".format(k, n_t)))

#%% FOURIER FOR WHOLE PLANES IN TIME

k = 1200

n = results_plane[0].shape[1]
planes_labels = ["Flujo {} [u.a]".format(l) for l in planes_series]

index_to_space = lambda i : i * cell_width/n - cell_width/2
space_to_index = lambda x : round(n * (x + cell_width/2) / cell_width)

results_plane_crop = [p[0:k, 
                        space_to_index(-r):space_to_index(r), 
                        space_to_index(-r):space_to_index(r)]
                      for p in results_plane]

n_crop = results_plane_crop[0].shape[1]
n_t = results_plane[0].shape[0]
space_index = [0,0]
    
fourier_lines_planes = [np.ndarray((k//2 + 1, n_crop, n_crop)) for i in range(6)]
fourier_freq_planes = np.fft.rfftfreq(k, d=period_plane)
for p, fp in zip(results_plane_crop, fourier_lines_planes):
    for i in range(n_crop):
        for j in range(n_crop):
            fp[:, i, j] = np.abs(np.fft.rfft(p[:, i, j]))

# fourier_planes = []
# for fp in fourier_lines_planes:
#     data = [dblquad(fpd, -r, r, -r, r) for fpd in fp]
#     fourier_planes.append(data)

fourier_planes = [np.array([np.sum(fpd)/(n_crop)**2 for fpd in fp]) for fp in fourier_lines_planes]

fig, ax = plt.subplots(3, 2)
fig.subplots_adjust(hspace=0.25)
for a, h in zip(np.reshape(ax, 6), planes_labels):
    a.set_title(h.split(" ")[1].split("0")[0])

for d, a in zip(fourier_planes, np.reshape(ax, 6)):
    a.plot(1e3*from_um_factor/fourier_freq_planes, d)
    a.set_xlim(*wlen_range*1e3*from_um_factor)
    a.xaxis.set_visible(False)
    a.yaxis.set_visible(False)
ax[-1,0].xaxis.set_visible(True)
ax[-1,1].xaxis.set_visible(True)
ax[-1,0].set_xlabel("Wavelength [nm]")
ax[-1,1].set_xlabel("Wavelength [nm]")

plt.savefig(file("PlanesFluxFourier-{}-{}.png".format(k, n_t)))


#%% MAKE FOURIER FOR ONE DOT IN TIME GIF

# What should be parameters
nframes_zero = 1000
nframes_step = 10
nframes = int((results_plane[0].shape[0] - nframes_zero)/nframes_step)

# Generate data to plot
n = results_plane[0].shape[1]
planes_labels = ["Flujo {} [u.a]".format(l) for l in planes_series]

index_to_space = lambda i : i * cell_width/n - cell_width/2
space_to_index = lambda x : round(n * (x + cell_width/2) / cell_width)

results_plane_crop = [p[:, 
                        space_to_index(-r):space_to_index(r), 
                        space_to_index(-r):space_to_index(r)]
                      for p in results_plane]

n_crop = results_plane_crop[0].shape[1]
n_t = results_plane_crop[0].shape[0]

index_to_space_crop = lambda i : i * 2*r/n_crop - r
space_to_index_crop = lambda x : round(n_crop * (x + r) / (2*r))

i, j = [space_to_index_crop(0), space_to_index_crop(0)]
call_x_series = lambda k : 1e3*from_um_factor/np.fft.rfftfreq(k, d=period_plane)
call_y_series = lambda k : [np.abs(np.fft.rfft(p[0:k,i,j])) for p in results_plane_crop]
label_function = lambda k : 'Tiempo: {:.1f} u.a.'.format(k*period_plane)

# Animation base
fig, ax = plt.subplots(3, 2)
fig.subplots_adjust(hspace=0.25)
# lims = (np.min(results_plane), np.max(results_plane))

def make_pic_fourier(k):    
    
    for d, a, ps in zip(call_y_series(k), np.reshape(ax, 6), planes_series):
        a.clear()
        a.set_title(ps)
        a.plot(call_x_series(k), d)
        # a.set_xlim(*wlen_range*1e3*from_um_factor)
        a.xaxis.set_visible(False)
        # a.yaxis.set_visible(False)
    ax[-1,0].xaxis.set_visible(True)
    ax[-1,1].xaxis.set_visible(True)
    ax[-1,0].set_xlabel("Wavelength [nm]")
    ax[-1,1].set_xlabel("Wavelength [nm]")
    ax[0,1].text(-.37, 1.2, label_function(k), transform=ax[0,1].transAxes)
    plt.show()
    
    return ax

def make_gif_fourier(gif_filename):
    pics = []
    for k in range(nframes_zero, nframes*nframes_step+nframes_zero, nframes_step):
        ax = make_pic_fourier(k)
        plt.savefig('temp_pic.png') 
        pics.append(mim.imread('temp_pic.png')) 
        print(str(k)+'/'+str(nframes+nframes_zero))
    mim.mimsave(gif_filename+'.gif', pics, fps=5)
    os.remove('temp_pic.png')
    print('Saved gif')

make_gif_fourier(file("DotsFluxFourier2"))

#%%

sim.reset_meep()