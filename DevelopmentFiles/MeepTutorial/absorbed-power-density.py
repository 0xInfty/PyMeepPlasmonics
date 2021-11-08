# -*- coding: utf-8 -*-

# From Meep Tutorial: Absorbed Power Density Map of Lossy Cylinder

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import meep as mp
from meep.materials import SiO2

# Absorbed power density: Absp = Re[E* . dP/dt]
# Total polarization field: P = D - E so that D = E + P
# Equivalent harmonic domain: Absp = ω Im[E* . D]

# Calculating this quantity involves two steps: 
# (1) compute Fourier-transformed E and D fields in a region ("dft_fields") 
# (2) compute ω Im[E∗⋅D]
# Watch it! This only works when the permittivity is specified using 
# Drude-Lorentzian susceptibility. Conductivity is not supported.

#%% PARAMETERS

resolution = 100  # pixels/um

r = 1.0     # radius of cylinder

dpml = 1.0
dair = 2.0  # air padding thickness

s = 2*(dpml+dair+r) # cell size

wvl = 1.0 # source parameters
fcen = 1/wvl

#%% BASIC CONFIGURATION

cell_size = mp.Vector3(s,s)

pml_layers = [mp.PML(thickness=dpml)]


sources = [mp.Source(mp.GaussianSource(fcen,
                                       fwidth=0.1*fcen,
                                       is_integrated=True),
                     center=mp.Vector3(-0.5*s+dpml),
                     size=mp.Vector3(0, s),
                     component=mp.Ez)]
# Beware! is_integrated=True needed for any planewave source extending into PML

symmetries = [mp.Mirror(mp.Y)]

geometry = [mp.Cylinder(material=SiO2, # look at that! fancy :3
                        center=mp.Vector3(),
                        radius=r,
                        height=mp.inf)]
# We work in 2D because of the infinite translation symmetry in z

sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    boundary_layers=pml_layers,
                    sources=sources,
                    k_point=mp.Vector3(),
                    symmetries=symmetries,
                    geometry=geometry)

#%% FLUX AND FIELDS CONFIGURATIONS

dft_fields = sim.add_dft_fields([mp.Dz, mp.Ez],
                                fcen,0,1,
                                center=mp.Vector3(),
                                size=mp.Vector3(2*r,2*r),
                                yee_grid=True)
# Watch it! To eliminate discretization artifacts when computing E∗⋅D product, 
# yee_grid=True ensures that Ez and Dz are computed on the Yee grid rather than
# interpolated to the centered grid.
# We cannot use get_array_metadata for dft_fields region or its interpolation weights because this involves the centered grid.

# Closed box surrounding cylinder for computing total incoming flux
flux_box = sim.add_flux(fcen, 0, 1,
                        mp.FluxRegion(center=mp.Vector3(x=-r),
                                      size=mp.Vector3(0,2*r),
                                      weight=+1),
                        mp.FluxRegion(center=mp.Vector3(x=+r),
                                      size=mp.Vector3(0,2*r),
                                      weight=-1),
                        mp.FluxRegion(center=mp.Vector3(y=+r),
                                      size=mp.Vector3(2*r,0),
                                      weight=-1),
                        mp.FluxRegion(center=mp.Vector3(y=-r),
                                      size=mp.Vector3(2*r,0),
                                      weight=+1))

#%% SIMULATION!

sim.run(until_after_sources=100)

Dz = sim.get_dft_array(dft_fields, mp.Dz, 0)
Ez = sim.get_dft_array(dft_fields, mp.Ez, 0)
absorbed_power_density = 2*np.pi*fcen * np.imag(np.conj(Ez)*Dz)

dxy = 1/resolution**2
absorbed_power = np.sum(absorbed_power_density)*dxy
absorbed_flux = mp.get_fluxes(flux_box)[0]
err = abs(absorbed_power-absorbed_flux)/absorbed_flux
print("flux:, {} (dft_fields), {} (dft_flux), {} (error)".format(absorbed_power,absorbed_flux,err))

plt.figure()
sim.plot2D()
plt.savefig('power_density_cell.png',dpi=150,bbox_inches='tight')

plt.figure()
x = np.linspace(-r,r,Dz.shape[0])
y = np.linspace(-r,r,Dz.shape[1])
plt.pcolormesh(x,
               y,
               np.transpose(absorbed_power_density),
               cmap='inferno_r',
               shading='gouraud',
               vmin=0,
               vmax=np.amax(absorbed_power_density))
plt.xlabel("x (μm)")
plt.xticks(np.linspace(-r,r,5))
plt.ylabel("y (μm)")
plt.yticks(np.linspace(-r,r,5))
plt.gca().set_aspect('equal')
plt.title("absorbed power density" + "\n" +"SiO2 Labs(λ={} μm) = {:.2f} μm".format(wvl,wvl/np.imag(np.sqrt(SiO2.epsilon(fcen)[0][0]))))
plt.colorbar()
plt.savefig('power_density_map.png',dpi=150,bbox_inches='tight')

# Note on units: (Meep power)/(unit length)^2 for Absp
# where (unit length) is 1 μm