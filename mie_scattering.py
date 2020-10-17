# -*- coding: utf-8 -*-

# From Meep Tutorial: Mie Scattering of a Lossless Dielectric Sphere

# Scattering efficiency of an homogeneous sphere given an incident planewave.

import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import os
# from mayavi import mlab
from time import time
import PyMieScatt as ps
from v_materials import import_medium
import v_save as vs
# from v_units import MeepUnitsManager

#%% PARAMETERS

### MEAN PARAMETERS

# Units: 10 nm as length unit
from_um_factor = 10e-3 # Conversion of 1 μm to my length unit (=10nm/1μm)
resolution = 4 # >=8 pixels per smallest wavelength, i.e. np.floor(8/wvl_min)

# Dielectric sphere
r = 6  # Radius of sphere: 60 nm
medium = import_medium("Au", from_um_factor) # Medium of sphere: gold (Au)

# Frequency and wavelength
wlen_range = np.array([50,65]) # 500-650 nm range from lowest to highest
nfreq = 100 # Number of frequencies to discretize range

# Computation time
enlapsed = []

### OTHER PARAMETERS

# Units
# uman = MeepUnitsManager(from_um_factor=from_um_factor)

# Frequency and wavelength
freq_range = 1/wlen_range # Hz range in Meep units from highest to lowest
freq_center = np.mean(freq_range)
freq_width = max(freq_range) - min(freq_range)

# Space configuration
pml_width = 0.38 * max(wlen_range)
air_width = r/2 # 0.5 * max(wlen_range)

# Saving directories
series = "2020101701"
folder = "MieResults"
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
sources = [mp.Source(mp.GaussianSource(freq_center,
                                       fwidth=freq_width,
                                       is_integrated=True,
                                       cutoff=3.2),
                     center=mp.Vector3(-0.5*cell_width + pml_width),
                     size=mp.Vector3(0, cell_width, cell_width),
                     component=mp.Ez)]
# Ez-polarized planewave pulse 
# (its size parameter fills the entire cell in 2d)
# >> The planewave source extends into the PML 
# ==> is_integrated=True must be specified

geometry = [mp.Sphere(material=medium,
                      center=mp.Vector3(),
                      radius=r)]
# Lossless dielectric sphere 
# Wavelength-independent refractive index of 2.0

# Check structure
# sim = mp.Simulation(resolution=resolution,
#                     cell_size=cell_size,
#                     geometry=geometry,
#                     k_point=mp.Vector3(),
#                     symmetries=symmetries)

# enlapsed = []
# temp = time()
# sim.init_sim()
# enlapsed.append( time() - temp )

# epsilon = sim.get_epsilon(freq_center)
# sim.reset_meep()

# s = mlab.contour3d(np.abs(epsilon), colormap="YlGnBu")
# mlab.show()

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
enlapsed.append( time() - temp )

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

98 x 98 x 98 with resolution 3
(36 cells inside diameter)
==> 9.57 s

172x172x172 with resolution 2
(24 cells inside diameter)
==> 17.47 s

67 x 67 x 67 with resolution 4
(48 cells inside diameter)
==> 7.06 s

67,375 x 67,375 x 67,375 with resolution 8
(100 cells inside diameter)
==> 56.14 s
"""

#%% FIRST RUN: SIMULATION NEEDED TO NORMALIZE

temp = time()
sim.run(until_after_sources=1.2*cell_width)  #1.1
# Enough time for the pulse to pass through all the cell
    #     mp.stop_when_fields_decayed(
    # np.mean(wlen_range), # dT = mean period of source
    # mp.Ez, # Component of field to check
    # mp.Vector3(0.5*cell_width - pml_width, 0, 0), # Where to check
    # 1e-3)) # Factor to decay
enlapsed.append( time() - temp )
# Originally: Aprox 3 periods of lowest frequency, using T=λ/c=λ in Meep units 
# Now: Aprox 3 periods of highest frequency, using T=λ/c=λ in Meep units 

"""
112x112x112 with resolution 2
(24 cells inside diameter)
==> 135.95 s to complete 1st run

116x116x116 with resolution 2
(24 cells inside diameter)
==> 208 s to complete 1st run

67 x 67 x 67 with resolution 4
(48 cells inside diameter)
==> 2000 s = 33 min to complete 1st run
"""

freqs = mp.get_flux_freqs(box_x1)
box_x1_data = sim.get_flux_data(box_x1)
box_x2_data = sim.get_flux_data(box_x2)
box_y1_data = sim.get_flux_data(box_y1)
box_y2_data = sim.get_flux_data(box_y2)
box_z1_data = sim.get_flux_data(box_z1)
box_z2_data = sim.get_flux_data(box_z2)

box_x1_flux0 = np.asarray(mp.get_fluxes(box_x1))

sim.reset_meep()

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

#%% SECOND RUN: INITIALIZE

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

67 x 67 x 67 with resolution 4
(48 cells inside diameter)
==> 14.47 s to build with sphere
"""

temp = time()
sim.load_minus_flux_data(box_x1, box_x1_data)
sim.load_minus_flux_data(box_x2, box_x2_data)
sim.load_minus_flux_data(box_y1, box_y1_data)
sim.load_minus_flux_data(box_y2, box_y2_data)
sim.load_minus_flux_data(box_z1, box_z1_data)
sim.load_minus_flux_data(box_z2, box_z2_data)
enlapsed.append( time() - temp )
del box_x1_data, box_x2_data, box_y1_data, box_y2_data
del box_z1_data, box_z2_data

"""
112x112x112 with resolution 2
(24 cells in diameter)
==> 0.016 s to add flux

116x116x116 with resolution 2
(24 cells inside diameter)
==> 0.021 s to add flux

67 x 67 x 67 with resolution 4
(48 cells inside diameter)
==> 0.043 s to add flux
"""

#%% SECOND RUN: SIMULATION :D

temp = time()
sim.run(until_after_sources=10*1.2*cell_width) 
    #     mp.stop_when_fields_decayed(
    # np.mean(wlen_range), # dT = mean period of source
    # mp.Ez, # Component of field to check
    # mp.Vector3(0.5*cell_width - pml_width, 0, 0), # Where to check
    # 1e-3)) # Factor to decay
enlapsed.append( time() - temp )
del temp
# Aprox 30 periods of lowest frequency, using T=λ/c=λ in Meep units 

box_x1_flux = np.asarray(mp.get_fluxes(box_x1))
box_x2_flux = np.asarray(mp.get_fluxes(box_x2))
box_y1_flux = np.asarray(mp.get_fluxes(box_y1))
box_y2_flux = np.asarray(mp.get_fluxes(box_y2))
box_z1_flux = np.asarray(mp.get_fluxes(box_z1))
box_z2_flux = np.asarray(mp.get_fluxes(box_z2))

#%% ANALYSIS

scatt_flux = box_x1_flux - box_x2_flux
scatt_flux = scatt_flux + box_y1_flux - box_y2_flux
scatt_flux = scatt_flux + box_z1_flux - box_z2_flux

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
scatt_eff_theory = [ps.MieQ(np.sqrt(medium.epsilon(f)[0,0]*medium.mu(f)[0,0]), 
                            10/f,
                            2*r*10,
                            asDict=True)['Qsca'] 
                    for f in freqs]
# The simulation results are validated by comparing with 
# analytic theory of PyMieScatt module

#%% PLOT ALL TOGETHER

plt.figure()
plt.plot(10/freqs, scatt_eff_meep,'bo-',label='Meep')
plt.plot(10/freqs, scatt_eff_theory,'bo-',label='Theory')
plt.xlabel('Wavelength [nm]')
plt.ylabel('Scattering efficiency [σ/πr$^{2}$]')
plt.legend()
plt.title('Mie Scattering of Au Sphere With {} nm Radius'.format(r*10))
plt.tight_layout()
plt.savefig(file("Comparison.png"))

#%% PLOT SEPARATE

plt.figure()
plt.plot(10/freqs, scatt_eff_meep,'bo-',label='Meep')
plt.xlabel('Wavelength [nm]')
plt.ylabel('Scattering efficiency [σ/πr$^{2}$]')
plt.legend()
plt.title('Mie Scattering of Au Sphere With {} nm Radius'.format(r*10))
plt.tight_layout()
plt.savefig(file("Meep.png"))

plt.figure()
plt.plot(10/freqs, scatt_eff_theory,'bo-',label='Theory')
plt.xlabel('Wavelength [nm]')
plt.ylabel('Scattering efficiency [σ/πr$^{2}$]')
plt.legend()
plt.title('Mie Scattering of Au Sphere With {} nm Radius'.format(r*10))
plt.tight_layout()
plt.savefig(file("Theory.png"))

#%% PLOT ONE ABOVE THE OTHER

fig, axes = plt.subplots(nrows=2, sharex=True)
fig.subplots_adjust(hspace=0)

axes[0].plot(10/freqs, scatt_eff_meep,'bo-',label='Meep')
axes[0].yaxis.tick_right()
axes[0].set_ylabel('Scattering efficiency [σ/πr$^{2}$]')
axes[0].legend()

axes[1].plot(10/freqs, scatt_eff_theory,'bo-',label='Theory')
axes[1].set_xlabel('Wavelength [nm]')
axes[1].set_ylabel('Scattering efficiency [σ/πr$^{2}$]')
axes[1].legend()

plt.savefig(file("Comparison.png"))


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

params = dict(
    from_um_factor=from_um_factor,
    resolution=resolution,
    r=r,
    wlen_range=wlen_range,
    nfreq=nfreq,
    pml_width=pml_width,
    air_width=air_width,
    source_center=source_center,
    enlapsed=enlapsed,
    series=series,
    folder=folder,
    home=home
    )

vs.savetxt(file("Results.txt"), data, header=header, footer=params)
vs.savetxt(file("BaseResults.txt"), data_base, header=header_base, footer=params)