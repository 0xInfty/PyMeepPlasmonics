# -*- coding: utf-8 -*-

# From Meep Tutorial: Mie Scattering of a Lossless Dielectric Sphere

# Scattering efficiency of an homogeneous sphere given an incident planewave.

import h5py as h5
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from time import time
import PyMieScatt as ps
import v_save as vs

#%%

### MEAN PARAMETERS

# Units: 1μm as length unit
resolution = 25
# >=8 pixels per smallest wavelength, i.e. np.floor(8/wvl_min)

# Dielectric Sphere
r = 1.0  # radius of sphere
n_sphere = 2.0 # refrac

# Frequency and wavelength
wlen_range = np.array([2*np.pi*r/10, 2*np.pi*r/2])
# From 10% to 50% of the circumference
nfreq = 100

# Computation time
enlapsed = []

# Saving directories
series = "2020102602"
folder = "MieResults"
home = "/home/vall/Documents/Thesis/ThesisPython/MeepTutorial/"

### OTHER PARAMETERS

# Frequency and wavelength
freq_range = 1/wlen_range # Hz range in Meep units from highest to lowest
freq_center = np.mean(freq_range)
freq_width = max(freq_range) - min(freq_range)

# Space configuration
pml_width = 0.5 * max(wlen_range)
air_width = 0.5 * max(wlen_range)

#%% GENERAL GEOMETRY SETUP

pml_layers = [mp.PML(thickness=pml_width)]

symmetries = [mp.Mirror(mp.Y),
              mp.Mirror(mp.Z, phase=-1)]
# Cause of symmetry, two mirror planes reduce cell size to 1/4

cell_width = 2 * (pml_width + air_width + r)
cell_size = mp.Vector3(cell_width, cell_width, cell_width)

source_center = -0.5*cell_width + pml_width
sources = [mp.Source(mp.GaussianSource(freq_center,
                                       fwidth=freq_width,
                                       is_integrated=True),
                                       # cutoff=3.2),
                     center=mp.Vector3(source_center),
                     size=mp.Vector3(0, cell_width, cell_width),
                     component=mp.Ez)]
# Ez-polarized planewave pulse 
# (its size parameter fills the entire cell in 2d)
# >> The planewave source extends into the PML 
# ==> is_integrated=True must be specified

geometry = [mp.Sphere(material=mp.Medium(index=n_sphere),
                      center=mp.Vector3(),
                      radius=r)]
# Lossless dielectric sphere 
# Wavelength-independent refractive index of 2.0

path = os.path.join(home, folder, "{}Results".format(series))
if not os.path.isdir(path): vs.new_dir(path)
file = lambda f : os.path.join(path, f)


#%% FIRST RUN: SET UP

sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    boundary_layers=pml_layers,
                    sources=sources,
                    k_point=mp.Vector3(),
                    symmetries=symmetries)
# >> k_point zero specifies boundary conditions needed
# for the source to be infinitely extended

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
# Funny you can encase the sphere (r radius) so closely (2r-sided box)

#%% FIRST RUN: INITIALIZE

temp = time()
sim.init_sim()
enlapsed = time() - temp

"""
8x8x8 with resolution 25
(25 cells inside diameter)
==> Build 3.43 s
"""

#%% FIRST RUN: SIMULATION NEEDED TO NORMALIZE

temp = time()
sim.run(until_after_sources=10)
enlapsed = [enlapsed, time()-temp]

freqs = np.asarray(mp.get_flux_freqs(box_x1))
box_x1_data = sim.get_flux_data(box_x1)
box_x2_data = sim.get_flux_data(box_x2)
box_y1_data = sim.get_flux_data(box_y1)
box_y2_data = sim.get_flux_data(box_y2)
box_z1_data = sim.get_flux_data(box_z1)
box_z2_data = sim.get_flux_data(box_z2)

# box_x1_flux0 = mp.get_fluxes(box_x1)
box_x1_flux0 = np.asarray(mp.get_fluxes(box_x1))
box_x2_flux0 = np.asarray(mp.get_fluxes(box_x2))
box_y1_flux0 = np.asarray(mp.get_fluxes(box_y1))
box_y2_flux0 = np.asarray(mp.get_fluxes(box_y2))
box_z1_flux0 = np.asarray(mp.get_fluxes(box_z1))
box_z2_flux0 = np.asarray(mp.get_fluxes(box_z2))

field = sim.get_array(center=mp.Vector3(), 
                      size=(cell_width, cell_width, cell_width), 
                      component=mp.Ez)

sim.reset_meep()

#%% SAVE MID DATA

params = dict(
    resolution=resolution,
    r=r,
    pml_width=pml_width,
    air_width=air_width,
    cell_width=cell_width,
    source_center=source_center,
    wlen_range=wlen_range,
    nfreq=nfreq,
    series=series,
    folder=folder, 
    home=home,
    enlapsed=enlapsed
    )

f = h5.File(file("MidField.h5"), "w")
f.create_dataset("Ez", data=field)
for a in params: f["Ez"].attrs[a] = params[a]
f.close()
del f

data_mid = np.array([1000/freqs, box_x1_flux0, box_x2_flux0, box_y1_flux0, 
                     box_y2_flux0, box_z1_flux0, box_z2_flux0]).T

header_mid = ["Longitud de onda [nm]", 
              "Flujo X10 [u.a.]",
              "Flujo X20 [u.a]",
              "Flujo Y10 [u.a]",
              "Flujo Y20 [u.a]",
              "Flujo Z10 [u.a]",
              "Flujo Z20 [u.a]"]

vs.savetxt(file("MidFlux.txt"), data_mid, header=header_mid, footer=params)

#%% PLOT FLUX FOURIER MID DATA

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
    a.plot(1000/freqs, d)
    a.set_ylim(*ylims)

plt.savefig(file("MidFlux.png"))

#%% PLOT FLUX WALLS FIELD

index_to_space = lambda i : i/resolution - cell_width/2
space_to_index = lambda x : round(resolution * (x + cell_width/2))

field_walls = [field[space_to_index(-r),:,:],
               field[space_to_index(r),:,:],
               field[:,space_to_index(-r),:],
               field[:,space_to_index(r),:],
               field[:,:,space_to_index(-r)],
               field[:,:,space_to_index(r)]]

zlims = (np.min([np.min(f) for f in field_walls]), 
         np.max([np.max(f) for f in field_walls]))

fig, ax = plt.subplots(3, 2)
fig.subplots_adjust(hspace=0.25)
for a, h in zip(np.reshape(ax, 6), header_mid[1:]):
    a.set_title(h.split(" ")[1].split("0")[0])

for f, a in zip(field_walls, np.reshape(ax, 6)):
    a.imshow(f.T, interpolation='spline36', cmap='RdBu', 
             vmin=zlims[0], vmax=zlims[1])
    a.axis("off")

plt.savefig(file("MidField.png"))

# Why do I seem to have constant field at planes not parallel to the source?

#%% SECOND RUN: SETUP

sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    boundary_layers=pml_layers,
                    sources=sources,
                    k_point=mp.Vector3(),
                    symmetries=symmetries,
                    geometry=geometry)

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

temp = time()
sim.load_minus_flux_data(box_x1, box_x1_data)
sim.load_minus_flux_data(box_x2, box_x2_data)
sim.load_minus_flux_data(box_y1, box_y1_data)
sim.load_minus_flux_data(box_y2, box_y2_data)
sim.load_minus_flux_data(box_z1, box_z1_data)
sim.load_minus_flux_data(box_z2, box_z2_data)
enlapsed.append( time() - temp )
# del box_x1_data, box_x2_data, box_y1_data, box_y2_data
# del box_z1_data, box_z2_data

#%% SECOND RUN: SIMULATION :D

temp = time()
sim.run(until_after_sources=100)
enlapsed.append( time() - temp )
del temp

# box_x1_flux = mp.get_fluxes(box_x1)
# box_x2_flux = mp.get_fluxes(box_x2)
# box_y1_flux = mp.get_fluxes(box_y1)
# box_y2_flux = mp.get_fluxes(box_y2)
# box_z1_flux = mp.get_fluxes(box_z1)
# box_z2_flux = mp.get_fluxes(box_z2)
box_x1_flux = np.asarray(mp.get_fluxes(box_x1))
box_x2_flux = np.asarray(mp.get_fluxes(box_x2))
box_y1_flux = np.asarray(mp.get_fluxes(box_y1))
box_y2_flux = np.asarray(mp.get_fluxes(box_y2))
box_z1_flux = np.asarray(mp.get_fluxes(box_z1))
box_z2_flux = np.asarray(mp.get_fluxes(box_z2))

#%% ANALYSIS

# scatt_flux = np.asarray(box_x1_flux) - np.asarray(box_x2_flux)
# scatt_flux = scatt_flux + np.asarray(box_y1_flux) - np.asarray(box_y2_flux)
# scatt_flux = scatt_flux + np.asarray(box_z1_flux) - np.asarray(box_z2_flux)
scatt_flux = box_x1_flux - box_x2_flux
scatt_flux = scatt_flux + box_y1_flux - box_y2_flux
scatt_flux = scatt_flux + box_z1_flux - box_z2_flux

# intensity = np.asarray(box_x1_flux0)/(2*r)**2
intensity = box_x1_flux0/(2*r)**2
# Flux of one of the six monitor planes / Área
# (the closest one, facing the planewave source) 
# This is why the six sides of the flux box are separated
# (Otherwise, the box could've been one flux object with weights ±1 per side)

scatt_cross_section = np.divide(scatt_flux, intensity)
# Scattering cross section σ = 
# = scattered power in all directions / incident intensity.

scatt_eff_meep = -1 * scatt_cross_section / (np.pi*r**2)
# Scattering efficiency =
# = scattering cross section / cross sectional area of the sphere

freqs = np.array(freqs)
scatt_eff_theory = [ps.MieQ(n_sphere, 
                            1000/f,
                            2*r*1000,
                            asDict=True)['Qsca'] 
                    for f in freqs]
# The simulation results are validated by comparing with 
# analytic theory of PyMieScatt module

#%% PLOT

plt.figure(dpi=150)
plt.loglog(2*np.pi*r*np.asarray(freqs),
           scatt_eff_meep,'bo-',label='Meep')
plt.loglog(2*np.pi*r*np.asarray(freqs),
           scatt_eff_theory,'ro-',label='Theory')
plt.grid(True,which="both",ls="-")
plt.xlabel('Circumference/Wavelength [2πr/λ]')
plt.ylabel('Scattering efficiency [σ/πr$^{2}$]')
plt.legend()
plt.title('Mie Scattering of a Lossless Dielectric Sphere')
plt.tight_layout()

plt.savefig(file("MeepTutPlot.png"))

#%% SAVE DATA

data = np.array([10/freqs, scatt_eff_meep, scatt_eff_theory]).T

header = ["Longitud de onda [nm]", 
          "Sección eficaz efectiva (Meep) [u.a.]", 
          "Sección eficaz efectiva (Theory) [u.a.]"]

data_base = np.array([10/freqs, box_x1_flux0, box_x1_flux, box_x2_flux, 
                      box_y1_flux, box_y2_flux, box_z1_flux, box_z2_flux,
                      intensity, scatt_flux, scatt_cross_section]).T

header_base = ["Longitud de onda [nm]", 
               "Flujo X10 [u.a.]",
               "Flujo X1 [u.a]",
               "Flujo X2 [u.a]",
               "Flujo Y1 [u.a]",
               "Flujo Y2 [u.a]",
               "Flujo Z1 [u.a]",
               "Flujo Z2 [u.a]",
               "Intensidad incidente [u.a.]", 
               "Flujo scattereado [u.a.]",
               "Sección eficaz de scattering [u.a.]"]

vs.savetxt(file("Results.txt"), data, header=header, footer=params)
vs.savetxt(file("BaseResults.txt"), data_base, header=header_base, footer=params)