# -*- coding: utf-8 -*-

# From the Meep tutorial: plotting permittivity and fields of a bent waveguide

import os as os
import meep as mp
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import imageio as mim

#%% SIMULATION SETUP

path = "/home/vall/Documents/Thesis/NanoMeepPlasmonics/MeepTutorial/BentWaveguideResults"
prefix = "BentWaveguide"

cell = mp.Vector3(16,16,0)
geometry = [mp.Block(mp.Vector3(12,1,mp.inf),
                     center=mp.Vector3(-2.5,-3.5),
                     material=mp.Medium(epsilon=12)),
            mp.Block(mp.Vector3(1,12,mp.inf),
                     center=mp.Vector3(3.5,2),
                     material=mp.Medium(epsilon=12))]
pml_layers = [mp.PML(1.0)]
resolution = 10

sources = [mp.Source(mp.ContinuousSource(wavelength=2*(11**0.5), width=20),
                     component=mp.Ez,
                     center=mp.Vector3(-7,-3.5),
                     size=mp.Vector3(0,1))]
# Instead of freq, you can choose wavelength
# You can have a line-source instead of a punctual source

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution,
                    filename_prefix=prefix)
os.chdir(path)

#%% CHECK LAYOUT

sim.init_sim() # Initialize the structures as if you were to run
eps_array = sim.get_epsilon()
x, y, *more = sim.get_array_metadata() # (x,y,z,w) = sim.get_array_metadata()
# Returns coordinates and interpolation weights of the fields :)
del more

plt.figure()
ax = plt.subplot(111)
plt.pcolormesh(x,y,np.transpose(eps_array),shading='gouraud')
ax.set_aspect('equal')
plt.show()

#%% FIRST SIMULATION --> HDF FILE

sim.run(mp.at_beginning(mp.output_epsilon),
        mp.to_appended("ez", mp.at_every(0.6, mp.output_efield_z)),
        until=200)

#%% TOYING AROUND WITH HDF FILE :P

#filename = os.path.join(path, prefix + "-ez.h5")
#f = h5.File(filename,"r")

#list(f.keys()) # group ~ dictionary
#f["ez"].shape # datasheet ~ Numpy array
#f["ez"][0,0,:].shape

#f.create_dataset("larala", data=np.ones(100))
#f.create_group("folder")
#f.create_dataset("folder2/larala", data=np.ones(100))

#f["larala"].attrs["date"] = "20200908"
#f["larala"].attrs["comments"] = "No hay cambios"
#myattr = dict(f["larala"].attrs)

#f.close()

#%% BASIC PLOTS

# Single plot configuration
filename = os.path.join(path, prefix + "-ez.h5")
f = h5.File(filename,"r")
ez100 = f['ez'][:,:,100]
f.close()
    
# Make single 3D surface plot
x, y = np.meshgrid(x, y)
fig = plt.figure(tight_layout=True)  
ax = fig.gca(projection='3d')
ax.plot_surface(x, y, ez100, cmap='bwr')
fig.show()

# Make single 2D color map plot - right spatial scale, doesn't show Z value of cursor
fig = plt.figure(tight_layout=True)
ax = plt.subplot()
im = ax.pcolormesh(x, y, ez100.T, cmap='bwr', shading='gouraud')
ax.set_aspect('equal')

#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.05)
#plt.colorbar(im, cax=cax)

# Make single 2D color map plot - no spatial scale, shows Z value of cursor
fig = plt.figure(tight_layout=True)
ax = plt.subplot()
im = ax.imshow(ez100.T, cmap='bwr', interpolation='spline36', origin='lower')

# Add references' colorbar - not the original scale; it's already normalized
#plt.colorbar(im, use_gridspec=True)
#plt.show()

#%% 3D ANIMATION - this might take longer, but it's interactive

# What should be parameters
nframes = 111
nframes_step = 3
call_Z_series = lambda i : f['ez'][:,:,nframes_step*i]
label_function = lambda i : '{:.0f}'.format(nframes_step*i)
gif_filename="ez"

# Try 3D animation
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(x, y, f['ez'][:,:,0].T, cmap='bwr',
                vmin=np.min(f['ez']), vmax=np.max(f['ez']))
lims = (np.min(f['ez']), np.max(f['ez']))
ax.set_zlim(*lims)
plt.show()

def update(i):
    ax.clear()
    ax.plot_surface(x, y, call_Z_series(i), cmap='bwr',
                    vmin=np.min(f['ez']), vmax=np.max(f['ez']))
    ax.set_zlim(*lims)
    ax.text(0, -2, 0.40, label_function(i), transform=ax.transAxes)
    plt.show()
    return

anim = animation.FuncAnimation(fig, update, frames=nframes, 
                               interval=210, cache_frame_data=False)

#%% 3D ANIMATION - might be faster, although not interactive

# What should be parameters
nframes = 111
nframes_step = 3
call_Z_series = lambda i : f['ez'][:,:,i] # This is different from above
label_function = lambda i : '{:.0f}'.format(i) # This is also different
gif_filename="ez"

# Try different 3D animation
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(x, y, f['ez'][:,:,0].T, cmap='bwr',
                vmin=np.min(f['ez']), vmax=np.max(f['ez']))
lims = (np.min(f['ez']), np.max(f['ez']))
ax.set_zlim(*lims)
plt.show()

def make_pic(i):
    ax.clear()
    ax.plot_surface(x, y, call_Z_series(i), cmap='bwr',
                    vmin=lims[0], vmax=lims[-1])
    ax.set_zlim(*lims)
    ax.text(0, -2, 0.40, label_function(i), transform=ax.transAxes)
    plt.show()
    return ax # Here lies another difference with 'update'
    
def make_gif(gif_filename):
    #plt.figure()
    pics = []
    for i in range(0, nframes*nframes_step, nframes_step):
        ax = make_pic(i)
        plt.savefig('temp_pic.png') 
        pics.append(mim.imread('temp_pic.png')) 
        print(str(i)+'/'+str(nframes_step*nframes))
    mim.mimsave(gif_filename+'.gif', pics, fps=5)
    os.remove('temp_pic.png')
    print('Saved gif')

#%% SECOND SIMULATION --> SLICE

slice_results = []
def get_slice(sim):
    slice_results.append(sim.get_array(
        center=mp.Vector3(0,-3.5), 
        size=mp.Vector3(16,0), 
        component=mp.Ez))

sim.run(mp.at_beginning(mp.output_epsilon),
        mp.at_every(0.6, get_slice),
        until=200)
# This doesn't create an HDF file because it stores at list vals

plt.figure()
plt.imshow(slice_results, 
           interpolation='spline36', 
           cmap='RdBu')
#plt.axis('off')
plt.show()

#%% THIRD SIMULATION --> PNGs

#sim.run(mp.at_every(0.6 , mp.output_png(mp.Ez, "-Zc dkbluered")), until=200) 
# This took longer and made many files that filled the same total space :'(