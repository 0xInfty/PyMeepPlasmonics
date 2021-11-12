#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# From Meep tutorial: 2d ring-resonator modes

import h5py as h5
import matplotlib.pyplot as plt
import meep as mp
import numpy as np
from time import time
import os
import v_save as sav
from imageio import imread, mimsave

#%% GEOMETRY LAYOUT

# Because of cylindrical symmetry, we could simulate it much more 
# efficiently using cylindrical coordinates, but we're using 2D cartesians.

path = "/home/vall/Documents/Thesis/PyMeepPlasmonics/MeepTutorial/RingResults"
prefix = "Ring"

n = 3.4                 # index of waveguide
w = 1                   # width of waveguide
r = 1                   # inner radius of ring
pad = 4                 # padding between waveguide and edge of PML
dpml = 2                # thickness of PML
sxy = 2*(r+w+pad+dpml)  # cell size

# Create a ring waveguide by two overlapping cylinders
# 1st) Outer Cylinder
# Then) Inner (air) cylinder.

c1 = mp.Cylinder(radius=r+w, material=mp.Medium(index=n))
c2 = mp.Cylinder(radius=r)

# If we don't want to excite a specific mode symmetry, we can just
# put a single point source at some arbitrary place, pointing in some
# arbitrary direction.  We will only look for Ez-polarized modes.

fcen = 0.15             # pulse center frequency
df = 0.1                # pulse width (in frequency)

# Basic strategy to find modes (frequency and decay times)?
# 1st) Short pulse directly inside the cavity
# 2nd) Once the source is off, analyze bouncing fields
# 3rd) We do NOT compute the Fourier transformation cause high frequency would
# need long running time. Instead we apply filter diagonalization.
# 4th) If you want the field patterns, you run the simulation again
# with a narrow-bandwidth long pulse to excite only the mode you want.

# Beware! Computing modes in time domain is somewhat subtle and requires care.
# - You could miss a mode if the source happens to be nearly orthogonal to it.
# - You could also miss it if it's too close to another mode.
# - You will sometimes also accidentally identify spurious peak frequencies.

src = mp.Source(mp.GaussianSource(fcen, fwidth=df), mp.Ez, mp.Vector3(r+0.1))

sim = mp.Simulation(cell_size=mp.Vector3(sxy, sxy),
                    geometry=[c1, c2],
                    sources=[src],
                    resolution=10,
                    symmetries=[mp.Mirror(mp.Y)],
                    boundary_layers=[mp.PML(dpml)],
                    filename_prefix=prefix)

h = mp.Harminv(mp.Ez, mp.Vector3(r+0.1), fcen, df)

if not os.path.isdir(path):
    sav.new_dir(path)
os.chdir(path)

#%% SIMULATION: FREQUENCY

start = time()
sim.run(mp.at_beginning(mp.output_epsilon),
        mp.after_sources(h),
        until_after_sources=300)
end = time()
print("elapsed time: {:.2f}".format(end-start))

#%% SIMULATION: FIELDS

freqs = [h.modes[i][0] for i in range(len(h.modes))]

label = lambda w : "Freq-{:.3f}".format(w)

path_freqs = []
start = [start, time()]
for i, w in enumerate(freqs):
    
    sim.reset_meep()
    
    fcen = round(w,3)
    df = .01
    
    src = mp.Source(mp.GaussianSource(fcen, fwidth=df), 
                    mp.Ez, mp.Vector3(r+0.1))
    
    sim = mp.Simulation(cell_size=mp.Vector3(sxy, sxy),
                    geometry=[c1, c2],
                    sources=[src],
                    resolution=10,
                    symmetries=[mp.Mirror(mp.Y)],
                    boundary_layers=[mp.PML(dpml)],
                    filename_prefix=prefix+"-"+label(w))
    
    path_i = os.path.join(path, label(w))
    if not os.path.isdir(path_i):
        sav.new_dir(path_i)
    os.chdir(path_i)
    path_freqs.append(path_i)
    
    sim.run(mp.at_beginning(mp.output_epsilon),
            mp.after_sources(mp.to_appended("Ez", 
                                            mp.at_every(
                                                1/fcen/20, 
                                                mp.output_efield_z))), 
            until_after_sources=1/fcen)
    
end = [end, time()]
del i, w, path_i
# Output fields for one period at the end.  
# (If we output at a single time, we might accidentally catch 
# the Ez field when it is almost zero and get a distorted view.)

print("elapsed time 2: {:.2f}".format(end[1]-start[1]))

#%% PLOTS :P

# General parameters
nframes_step = 1
frame_label = lambda i : '{:.0f}'.format(i)
gif_filename = lambda w : "Ring-" + label(w) + "-ez"

file_freqs = []
for i, w, p in zip(range(len(freqs)), freqs, path_freqs):
    
    # Extract data
    os.chdir(p)
    file = os.path.join(p, prefix + "-" + label(w) + "-Ez.h5")
    file_freqs.append(file)
    f = h5.File(file,"r")
    ez = np.array(f['ez'])
    f.close()
    nframes = len(ez[0,0,:])
    call_Z_series = lambda i : ez[:,:,i]
    x, y, *more = sim.get_array_metadata() # (x,y,z,w) = sim.get_array_metadata()
    del more
    
    # Single plot configuration
    fig = plt.figure(tight_layout=True)
    ax = plt.subplot()
    im = ax.pcolormesh(x, y, ez[:,:,0].T, cmap='bwr', shading='gouraud',
                       vmin=np.min(ez), vmax=np.max(ez))
    ax.set_aspect('equal')
    plt.show()
    
    def make_pic(i):
        ax.clear()
        im = ax.pcolormesh(x, y, call_Z_series(i).T, cmap='bwr', shading='gouraud',
                           vmin=np.min(ez), vmax=np.max(ez))
        ax.set_aspect('equal')
        ax.text(0, -2, frame_label(i), transform=ax.transAxes)
        plt.show()
        return ax
        
    def make_gif():
        #plt.figure()
        pics = []
        for i in range(0, nframes*nframes_step, nframes_step):
            ax = make_pic(i)
            plt.savefig('temp_pic.png') 
            pics.append(imread('temp_pic.png')) 
            print(str(i+1)+'/'+str(nframes_step*nframes))
        mimsave(gif_filename(w)+'.gif', pics, fps=5)
        os.remove('temp_pic.png')
        print('Saved gif')
    
    make_gif()
    plt.close()
