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
nfreq = 100 # Number of frequencies to discretize range
cutoff = 3.2

# Computation time
enlapsed = []
time_factor_cell = 1.2
until_after_sources = False

# Saving data
period_line = 1/10 * min(wlen_range)
period_plane = 1/50 * min(wlen_range)
meep_flux = True
H_field = True

# Saving directories
series = "TimeFactorCell{}".format(time_factor_cell)
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

def get_line(sim, line_center, line_size, component):
    return sim.get_array(
        center=mp.Vector3(*line_center), 
        size=mp.Vector3(*line_size), 
        component=component) # mp.Ez, mp.Hy

def get_plane(sim, plane_center, plane_size, component):
    return sim.get_array(
        center=mp.Vector3(*plane_center), 
        size=mp.Vector3(*plane_size), 
        component=component)

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

if meep_flux:
    # Scattered power --> Computed by surrounding it with closed DFT flux box 
    # (its size and orientation are irrelevant because of Poynting's theorem) 
    box_x1 = sim.add_flux(freq_center, freq_width, nfreq, 
                          mp.FluxRegion(center=mp.Vector3(x=-r),
                                        size=mp.Vector3(0,2*r,2*r)))
    box_x2 = sim.add_flux(freq_center, freq_width, nfreq, 
                          mp.FluxRegion(center=mp.Vector3(x=+r),
                                        size=mp.Vector3(0,2*r,2*r)))
    box_y1 = sim.add_flux(freq_center, freq_width, nfreq,
                          mp.FluxRegion(center=mp.Vector3(y=-r),
                                        size=mp.Vector3(2*r,0,2*r)))
    box_y2 = sim.add_flux(freq_center, freq_width, nfreq,
                          mp.FluxRegion(center=mp.Vector3(y=+r),
                                        size=mp.Vector3(2*r,0,2*r)))
    box_z1 = sim.add_flux(freq_center, freq_width, nfreq,
                          mp.FluxRegion(center=mp.Vector3(z=-r),
                                        size=mp.Vector3(2*r,2*r,0)))
    box_z2 = sim.add_flux(freq_center, freq_width, nfreq,
                          mp.FluxRegion(center=mp.Vector3(z=+r),
                                        size=mp.Vector3(2*r,2*r,0)))

temp = time()
sim.init_sim()
enlapsed.append( time() - temp )

#%% DEFINE SAVE STEP FUNCTIONS

def slice_generator_params(get_slice, center, size):
    if H_field: 
        datasets = ["Ez", "Hy"]
        get_slices = [lambda sim: get_slice(sim, center, size, mp.Ez),
                      lambda sim: get_slice(sim, center, size, mp.Hy)]
    else: 
        datasets = "Ez"
        get_slices = lambda sim: get_slice(sim, center, size, mp.Ez)
    return datasets, get_slices

f, save_line = vs.save_slice_generator(sim, file("Lines.h5"), 
    *slice_generator_params(get_line, [0, 0, 0], [cell_width, 0, 0]))
gx1, save_plane_x1 = vs.save_slice_generator(sim, file("Planes_X1.h5"), 
    *slice_generator_params(get_line, [-r, 0, 0], [0, cell_width, cell_width]))
gx2, save_plane_x2 = vs.save_slice_generator(sim, file("Planes_X2.h5"),
    *slice_generator_params(get_plane, [r, 0, 0], [0, cell_width, cell_width]))
gy1, save_plane_y1 = vs.save_slice_generator(sim, file("Planes_Y1.h5"), 
    *slice_generator_params(get_plane, [0, -r, 0], [cell_width, 0, cell_width]))
gy2, save_plane_y2 = vs.save_slice_generator(sim, file("Planes_Y2.h5"),
    *slice_generator_params(get_plane, [0, r, 0], [cell_width, 0, cell_width]))
gz1, save_plane_z1 = vs.save_slice_generator(sim, file("Planes_Z1.h5"), 
    *slice_generator_params(get_plane, [0, 0, -r], [cell_width, cell_width, 0]))
gz2, save_plane_z2 = vs.save_slice_generator(sim, file("Planes_Z2.h5"),
    *slice_generator_params(get_plane, [0, 0, r], [cell_width, cell_width, 0]))

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

if meep_flux:
    freqs = np.asarray(mp.get_flux_freqs(box_x1))
    box_x1_data = sim.get_flux_data(box_x1)
    box_x2_data = sim.get_flux_data(box_x2)
    box_y1_data = sim.get_flux_data(box_y1)
    box_y2_data = sim.get_flux_data(box_y2)
    box_z1_data = sim.get_flux_data(box_z1)
    box_z2_data = sim.get_flux_data(box_z2)
    
    box_x1_flux0 = np.asarray(mp.get_fluxes(box_x1))
    box_x2_flux0 = np.asarray(mp.get_fluxes(box_x2))
    box_y1_flux0 = np.asarray(mp.get_fluxes(box_y1))
    box_y2_flux0 = np.asarray(mp.get_fluxes(box_y2))
    box_z1_flux0 = np.asarray(mp.get_fluxes(box_z1))
    box_z2_flux0 = np.asarray(mp.get_fluxes(box_z2))

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
if H_field:
    fields_series = ["Ez", "Hy"]
else:
    fields_series = "Ez"

f = h5.File(file("Lines.h5"), "r+")
for fs in fields_series:
    for a in params: f[fs].attrs[a] = params[a]
f.close()
del f

for s in planes_series:
    g = h5.File(file("Planes_{}.h5".format(s)), "r+")
    for f in fields_series: 
        for a in params: g[fs].attrs[a] = params[a]
    g.close()
del g

if meep_flux:
    data_mid = np.array([1e3*from_um_factor/freqs, box_x1_flux0, box_x2_flux0, 
                         box_y1_flux0, box_y2_flux0, box_z1_flux0, box_z2_flux0]).T
    
    header_mid = ["Longitud de onda [nm]", 
                  "Flujo X10 [u.a.]",
                  "Flujo X20 [u.a]",
                  "Flujo Y10 [u.a]",
                  "Flujo Y20 [u.a]",
                  "Flujo Z10 [u.a]",
                  "Flujo Z20 [u.a]"]
    
    vs.savetxt(file("MidFlux.txt"), data_mid, header=header_mid, footer=params)

#%% GET READY TO LOAD DATA

f = h5.File(file("Lines.h5"), "r")
results_line_E = f["Ez"]
results_line_H = f["Hy"]

g = []
results_plane_E = []
results_plane_H = []
for s in planes_series:
    gi = h5.File(file("Planes_{}.h5".format(s)), "r")
    g.append(gi)
    results_plane_E.append( gi["Ez"] )
    results_plane_H.append( gi["Ez"] )
del gi

#%% PLOT FLUX FOURIER MID DATA

if meep_flux:
    ylims = (np.min(data_mid[:,1:]), np.max(data_mid[:,1:]))
    ylims = (ylims[0]-.1*(ylims[1]-ylims[0]),
             ylims[1]+.1*(ylims[1]-ylims[0]))
    
    fig, ax = plt.subplots(3, 2, sharex=True)
    fig.subplots_adjust(hspace=0, wspace=.05)
    for a in ax[:,1]:
        a.yaxis.tick_right()
        a.yaxis.set_label_position("right")
    for a, h in zip(np.reshape(ax, 6), header_mid[1:]):
        a.set_ylabel(h)
    
    for d, a in zip(data_mid[:,1:].T, np.reshape(ax, 6)):
        a.plot(1e3*from_um_factor/freqs, d)
        a.set_ylim(*ylims)
    ax[-1,0].set_xlabel("Wavelength [nm]")
    ax[-1,1].set_xlabel("Wavelength [nm]")
    
    plt.savefig(file("MidFlux.png"))

#%% SHOW ALL LINES IN COLOR MAP

for fs, rl in zip(fields_series, [results_line_E, results_line_H]):
    plt.figure()
    plt.imshow(rl, interpolation='spline36', cmap='RdBu')
    plt.xlabel("Distancia en x (int)")
    plt.ylabel("Tiempo (int)")
    plt.show()

    plt.savefig(file("AxisXColorMap{}.png".format(fs)))

#%% MAKE LINES GIF

# What should be parameters
nframes_step = 3
nframes = int(results_line_E.shape[0]/nframes_step)
call_series = lambda i : results_line_E[i,:]
call_series_x = lambda i : np.linspace(-cell_width/2, cell_width/2, 
                                       len(results_line_E[i,:]))
label_function = lambda i : 'Tiempo: {:.1f} u.a.'.format(i*period_line)
label_y = "Campo eléctrico Ez (u.a.)"
lims = (np.min(results_line_E), np.max(results_line_E))
color = 'C0'

# Animation base
fig = plt.figure()
ax = plt.subplot()

def make_pic_line(i):
    ax.clear()
    plt.plot(call_series_x(i), call_series(i), color)
    ax.set_ylim(*lims)
    ax.text(-.12, -.1, label_function(i), transform=ax.transAxes)
    plt.xlabel("Distancia en x (u.a.)")
    plt.ylabel(label_y)
    plt.show()
    return ax

def make_gif_line(gif_filename):
    pics = []
    for i in range(nframes):
        ax = make_pic_line(i*nframes_step)
        plt.savefig('temp_pic.png') 
        pics.append(mim.imread('temp_pic.png')) 
        print(str(i+1)+'/'+str(nframes))
    mim.mimsave(gif_filename, pics, fps=5)
    os.remove('temp_pic.png')
    plt.close(fig)
    print('Saved gif')

make_gif_line(file("AxisXEzCrop.gif"))

#%% MAKE LINES GIF FOR HY

# What should be parameters
nframes_step = 3
nframes = int(results_line_H.shape[0]/nframes_step)
call_series = lambda i : results_line_H[i,:]
call_series_x = lambda i : np.linspace(-cell_width/2, cell_width/2, 
                                       len(results_line_H[i,:]))
label_function = lambda i : 'Tiempo: {:.1f} u.a.'.format(i*period_line)
label_y = "Campo magnético Hy (u.a.)"
lims = (np.min(results_line_H), np.max(results_line_H))
color = 'C1'

# Animation base
fig = plt.figure()
ax = plt.subplot()

make_gif_line(file("AxisXHy.gif"))

#%% SHOW SOURCE

index_to_space = lambda i : i/resolution - cell_width/2
space_to_index = lambda x : int(round(resolution * (x + cell_width/2)))

source_field_E = np.asarray(results_line_E[:, space_to_index(source_center)])
source_field_H = np.asarray(results_line_H[:, space_to_index(source_center)])

plt.figure()
plt.plot(np.arange(len(source_field_E))*period_line, source_field_E)
plt.xlabel("Tiempo (u.a.)")
plt.ylabel("Campo eléctrico Ez (u.a.)")

plt.savefig(file("SourceEz.png"))

plt.figure()
plt.plot(np.arange(len(source_field_H))*period_line, source_field_H, 'C1')
plt.xlabel("Tiempo (u.a.)")
plt.ylabel("Campo magnético Hy (u.a.)")

plt.savefig(file("SourceHy.png"))

#%% MAKE FOURIER FOR SOURCE

index_to_space = lambda i : i/resolution - cell_width/2
space_to_index = lambda x : int(round(resolution * (x + cell_width/2)))

source_field = np.asarray(results_line_E[:, space_to_index(source_center)])

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
n = results_plane_E[0].shape[1]

index_to_space = lambda i : i * cell_width/n - cell_width/2
space_to_index = lambda x : int(round(n * (x + cell_width/2) / cell_width))

zlims = (np.min(results_plane_E), np.max(results_plane_E))
planes_labels = ["Flujo {} [u.a.]".format(s) for s in planes_series]

fig, ax = plt.subplots(3, 2)
fig.subplots_adjust(hspace=0.25)
for a, h in zip(np.reshape(ax, 6), planes_labels):
    a.set_title(h.split(" ")[1].split("0")[0])

for p, a in zip(results_plane_E, np.reshape(ax, 6)):
    a.imshow(call_series(i, p).T, interpolation='spline36', cmap='RdBu', 
             vmin=zlims[0], vmax=zlims[1])
    a.axis("off")

plt.savefig(file("PlanesIndex{}".format(i)))

#%% FOURIER FOR ONE DOT

n = results_plane_E[0].shape[1]
planes_labels = ["Flujo {} [u.a]".format(l) for l in planes_series]

index_to_space = lambda i : i * cell_width/n - cell_width/2
space_to_index = lambda x : int(round(n * (x + cell_width/2) / cell_width))

results_plane_E_crop = [p[:, 
                        space_to_index(-r):space_to_index(r), 
                        space_to_index(-r):space_to_index(r)]
                      for p in results_plane_E]

n_crop = results_plane_E_crop[0].shape[1]
n_t = results_plane_E_crop[0].shape[0]

index_to_space_crop = lambda i : i * 2*r/n_crop - r
space_to_index_crop = lambda x : int(round(n_crop * (x + r) / (2*r)))

i, j = [space_to_index_crop(0), space_to_index_crop(0)]
fourier_freq_planes = np.fft.rfftfreq(n_t, d=period_plane)
fourier_lines_dots = [np.abs(np.fft.rfft(p[:,i,j])) for p in results_plane_E_crop]

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

n = results_plane_E[0].shape[1]
planes_labels = ["Flujo {} [u.a]".format(l) for l in planes_series]

index_to_space = lambda i : i * cell_width/n - cell_width/2
space_to_index = lambda x : int(round(n * (x + cell_width/2) / cell_width))

results_plane_E_crop = [p[:, 
                        space_to_index(-r):space_to_index(r), 
                        space_to_index(-r):space_to_index(r)]
                      for p in results_plane_E]

n_crop = results_plane_E_crop[0].shape[1]
n_t = results_plane_E[0].shape[0]
space_index = [0,0]
    
fourier_lines_planes = [np.ndarray((n_t//2 + 1, n_crop, n_crop)) for i in range(6)]
fourier_freq_planes = np.fft.rfftfreq(n_t, d=period_plane)
for p, fp in zip(results_plane_E_crop, fourier_lines_planes):
    for i in range(n_crop):
        for j in range(n_crop):
            fp[:, i, j] = np.abs(np.fft.rfft(p[:, i, j]))**2

# fourier_planes = []
# for fp in fourier_lines_planes:
#     data = [dblquad(fpd, -r, r, -r, r) for fpd in fp]
#     fourier_planes.append(data)

fourier_planes = [np.array([(1e3*from_um_factor)**2*np.sum(fpd)/(n_crop)**2 
                            for fpd in fp]) for fp in fourier_lines_planes]

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

#%% MAKE FOURIER FOR ONE DOT IN TIME GIF

for zoom in [False, True]:
    
    # What should be parameters
    frames_zero = 10
    frames_end = results_plane_E[0].shape[0]
    nframes_step = 15
    if frames_end>results_plane_E[0].shape[0]: raise ValueError("Too large!")
    nframes = int((frames_end - frames_zero)/nframes_step)

    
    # Generate data to plot
    n = results_plane_E[0].shape[1]
    planes_labels = ["Flujo {} [u.a]".format(l) for l in planes_series]
    
    index_to_space = lambda i : i * cell_width/n - cell_width/2
    space_to_index = lambda x : int(round(n * (x + cell_width/2) / cell_width))
    
    results_plane_E_crop = np.array(
        [p[:, space_to_index(-r):space_to_index(r), 
              space_to_index(-r):space_to_index(r)]
                          for p in results_plane_E])
    
    n_crop = results_plane_E_crop[0].shape[1]
    n_t = results_plane_E_crop[0].shape[0]
    
    index_to_space_crop = lambda i : i * 2*r/n_crop - r
    space_to_index_crop = lambda x : int(round(n_crop * (x + r) / (2*r)))
    
    i, j = [space_to_index_crop(0), space_to_index_crop(0)]
    call_x_series = lambda k : 1e3*from_um_factor/np.fft.rfftfreq(k, d=period_plane)
    call_y_series = lambda k : [np.abs(np.fft.rfft(p[0:k,i,j])) for p in results_plane_E_crop]
    label_function = lambda k : 'Tiempo: {:.1f} u.a.'.format(k*period_plane)
    
    # Animation base
    fig, ax = plt.subplots(3, 2)
    fig.subplots_adjust(hspace=0.25)
    # lims = (np.min(results_plane_E), np.max(results_plane_E))
    
    def make_pic_fourier(k):    
        
        for d, a, ps in zip(call_y_series(k), np.reshape(ax, 6), planes_series):
            a.clear()
            a.set_title(ps)
            a.plot(call_x_series(k), d)
            if zoom: a.set_xlim(*wlen_range*1e3*from_um_factor)
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
        for k in range(frames_zero, frames_end, nframes_step):
            ax = make_pic_fourier(k)
            plt.savefig('temp_pic.png') 
            pics.append(mim.imread('temp_pic.png')) 
            print(str(k)+'/'+str(nframes_step*nframes+frames_zero))
        mim.mimsave(gif_filename, pics, fps=5)
        os.remove('temp_pic.png')
        print('Saved gif')
        plt.close(fig)
    
    if zoom: make_gif_fourier(file("DotsFluxFourierZoom.gif"))
    else: make_gif_fourier(file("DotsFluxFourier.gif"))

#%%

sim.reset_meep()