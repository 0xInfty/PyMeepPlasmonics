# -*- coding: utf-8 -*-

# From Meep Tutorial: Mie Scattering of a Lossless Dielectric Sphere

# Scattering efficiency of an homogeneous sphere given an incident planewave.

import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from time import time
import PyMieScatt as ps
from v_materials import import_medium
from v_units import MeepUnitsManager

#%%

### MEAN PARAMETERS

# Units: 10 nm as length unit
from_um_factor = 10e-3 # Conversion of 1 μm to my length unit (=10nm/1μm)
resolution = 2 # >=8 pixels per smallest wavelength, i.e. np.floor(8/wvl_min)

# Dielectric sphere
r = 6  # Radius of sphere: 60 nm
medium = import_medium("Au", from_um_factor) # Medium of sphere: gold (Au)

# Frequency and wavelength
wlen_range = np.array([20,80]) # 200-800 nm range from lowest to highest
nfreq = 100 # Number of frequencies to discretize range

### OTHER PARAMETERS

# Units
uman = MeepUnitsManager(from_um_factor=from_um_factor)

# Frequency and wavelength
freq_range = 1/wlen_range # Hz range in Meep units from highest to lowest
freq_center = np.mean(freq_range)
freq_width = max(freq_range) - min(freq_range)

# Space configuration
pml_width = 0.5 * max(wlen_range)
air_width = 2 * r # 0.5 * max(wlen_range)

#%% FIRST RUN: GEOMETRY SETUP

pml_layers = [mp.PML(thickness=pml_width)]

symmetries = [mp.Mirror(mp.Y),
              mp.Mirror(mp.Z, phase=-1)]
# Cause of symmetry, two mirror planes reduce cell size to 1/4

cell_width = 2 * (pml_width + air_width + r)
cell_size = mp.Vector3(cell_width, cell_width, cell_width)

sources = [mp.Source(mp.GaussianSource(freq_center,
                                       fwidth=freq_width,
                                       is_integrated=True),
                     center=mp.Vector3(-0.5*cell_width + pml_width),
                     size=mp.Vector3(0, cell_width, cell_width),
                     component=mp.Ez)]
# Ez-polarized planewave pulse 
# (its size parameter fills the entire cell in 2d)
# >> The planewave source extends into the PML 
# ==> is_integrated=True must be specified

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

#%% FIRST RUN: SIMULATION NEEDED TO NORMALIZE

temp = time()
sim.init_sim()
enlapsed = time() - temp

"""
112x112x112 with resolution 4
(48 cells inside diameter)
==> 30 s to build

112x112x112 with resolution 2
(24 cells inside diameter)
==> 4.26 s to build

116x116x116 with resolution 2
(24 cells inside diameter)
==> 5.63 s

172x172x172 with resolution 2
(24 cells inside diameter)
==> 17.47 s
"""

temp = time()
sim.run(until_after_sources=1.1*cell_width) 
# Enough time for the pulse to pass through all the cell
enlapsed = [enlapsed, time()-temp]
# Originally: Aprox 3 periods of lowest frequency, using T=λ/c=λ in Meep units 
# Now: Aprox 3 periods of highest frequency, using T=λ/c=λ in Meep units 

"""
112x112x112 with resolution 2
(24 cells inside diameter)
==> 135.95 s to complete 1st run

116x116x116 with resolution 2
(24 cells inside diameter)
==> 208 s to complete 1st run
"""

freqs = mp.get_flux_freqs(box_x1)
box_x1_data = sim.get_flux_data(box_x1)
box_x2_data = sim.get_flux_data(box_x2)
box_y1_data = sim.get_flux_data(box_y1)
box_y2_data = sim.get_flux_data(box_y2)
box_z1_data = sim.get_flux_data(box_z1)
box_z2_data = sim.get_flux_data(box_z2)

box_x1_flux0 = mp.get_fluxes(box_x1)

sim.reset_meep()

#%% SECOND RUN: GEOMETRY SETUP

geometry = [mp.Sphere(material=medium,
                      center=mp.Vector3(),
                      radius=r)]
# Lossless dielectric sphere 
# Wavelength-independent refractive index of 2.0

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

"""
112x112x112 with resolution 2
(24 cells in diameter)
==> 5.16 s to build with sphere

116x116x116 with resolution 2
(24 cells inside diameter)
==> 9.71 s to build with sphere
"""

temp = time()
sim.load_minus_flux_data(box_x1, box_x1_data)
sim.load_minus_flux_data(box_x2, box_x2_data)
sim.load_minus_flux_data(box_y1, box_y1_data)
sim.load_minus_flux_data(box_y2, box_y2_data)
sim.load_minus_flux_data(box_z1, box_z1_data)
sim.load_minus_flux_data(box_z2, box_z2_data)
enlapsed.append( time() - temp )

"""
112x112x112 with resolution 2
(24 cells in diameter)
==> 0.016 s to add flux

116x116x116 with resolution 2
(24 cells inside diameter)
==> 0.021 s to add flux
"""

#%% SECOND RUN: SIMULATION :D

temp = time()
sim.run(until_after_sources=10*1.1*cell_width) 
enlapsed.append( time() - temp )
# Aprox 30 periods of lowest frequency, using T=λ/c=λ in Meep units 

box_x1_flux = mp.get_fluxes(box_x1)
box_x2_flux = mp.get_fluxes(box_x2)
box_y1_flux = mp.get_fluxes(box_y1)
box_y2_flux = mp.get_fluxes(box_y2)
box_z1_flux = mp.get_fluxes(box_z1)
box_z2_flux = mp.get_fluxes(box_z2)

#%% ANALYSIS

scatt_flux = np.asarray(box_x1_flux) - np.asarray(box_x2_flux)
scatt_flux = scatt_flux + np.asarray(box_y1_flux) - np.asarray(box_y2_flux)
scatt_flux = scatt_flux + np.asarray(box_z1_flux) - np.asarray(box_z2_flux)

intensity = np.asarray(box_x1_flux0)/(2*r)**2
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
freqs_trim = freqs[freqs<0.0403274589668105]

scatt_eff_theory = [ps.MieQ(np.sqrt(medium.epsilon(f)[0,0]*medium.mu(f)[0,0]), 
                            10/f,
                            2*r*10,
                            asDict=True)['Qsca'] 
                    for f in freqs_trim]
# The simulation results are validated by comparing with 
# analytic theory of PyMieScatt module

plt.figure(dpi=150)
plt.plot(10/freqs, scatt_eff_meep,'bo-',label='Meep')
plt.plot(10/freqs_trim, scatt_eff_theory,'bo-',label='Theory')
# plt.loglog(2*np.pi*r*np.asarray(freqs),
#            scatt_eff_meep,'ro-',label='Meep')
# plt.loglog(2*np.pi*r*np.asarray(freqs),
#            scatt_eff_theory,'ro-',label='Theory')
# plt.grid(True,which="both",ls="-")
plt.xlabel('Wavelength [nm]')
plt.ylabel('Scattering efficiency [σ/πr$^{2}$]')
plt.legend()
plt.title('Mie Scattering of a Lossless Dielectric Sphere')
plt.tight_layout()