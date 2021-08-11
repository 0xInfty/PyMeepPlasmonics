# -*- coding: utf-8 -*-

# From Meep Tutorial: Mie Scattering of a Lossless Dielectric Sphere

# Scattering efficiency of an homogeneous sphere given an incident planewave.

import sys
sys.path.append("/home/vall/Documents/Thesis/ThesisPython")

import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from time import time
import os
import PyMieScatt as ps
import v_save as vs

#%%

r = 1.0  # radius of sphere
n_sphere = 2.0 # refrac

wvl_min = 2*np.pi*r/10
wvl_max = 2*np.pi*r/2
# From 10% to 50% of the circumference

frq_min = 1/wvl_max
frq_max = 1/wvl_min
frq_cen = 0.5*(frq_min+frq_max)
dfrq = frq_max-frq_min
nfrq = 100

from_um_factor = 1 # Conversion of 1 μm to my length unit (=10nm/1μm)
resolution = 25
# >=8 pixels per smallest wavelength, i.e. np.floor(8/wvl_min)

dpml = 0.5*wvl_max
dair = 0.5*wvl_max

# Saving directories
series = "pmpyParallel7"
folder = "MeepTutorial/TestPS"
home = "/home/vall/Documents/Thesis/ThesisResults"

#%% FIRST RUN: GEOMETRY SETUP

pml_layers = [mp.PML(thickness=dpml)]

symmetries = [mp.Mirror(mp.Y),
              mp.Mirror(mp.Z, phase=-1)]
# Cause of symmetry, two mirror planes reduce cell size to 1/4

s = 2*(dpml+dair+r)
cell_size = mp.Vector3(s, s, s)


sources = [mp.Source(mp.GaussianSource(frq_cen,fwidth=dfrq,is_integrated=True),
                     center=mp.Vector3(-0.5*s+dpml),
                     size=mp.Vector3(0, s, s), # s = cell size
                     component=mp.Ez)]
# Ez-polarized planewave pulse 
# (its size parameter fills the entire cell in 2d)
# >> The planewave source extends into the PML 
# ==> is_integrated=True must be specified

path = os.path.join(home, folder, series)
if not os.path.isdir(path): vs.new_dir(path)
file = lambda f : os.path.join(path, f)

temp = time()
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
box_x1 = sim.add_flux(frq_cen, dfrq, nfrq, 
                      mp.FluxRegion(center=mp.Vector3(x=-r),
                                    size=mp.Vector3(0,2*r,2*r)))
box_x2 = sim.add_flux(frq_cen, dfrq, nfrq, 
                      mp.FluxRegion(center=mp.Vector3(x=+r),
                                    size=mp.Vector3(0,2*r,2*r)))
box_y1 = sim.add_flux(frq_cen, dfrq, nfrq, 
                      mp.FluxRegion(center=mp.Vector3(y=-r),
                                    size=mp.Vector3(2*r,0,2*r)))
box_y2 = sim.add_flux(frq_cen, dfrq, nfrq, 
                      mp.FluxRegion(center=mp.Vector3(y=+r),
                                    size=mp.Vector3(2*r,0,2*r)))
box_z1 = sim.add_flux(frq_cen, dfrq, nfrq, 
                      mp.FluxRegion(center=mp.Vector3(z=-r),
                                    size=mp.Vector3(2*r,2*r,0)))
box_z2 = sim.add_flux(frq_cen, dfrq, nfrq, 
                      mp.FluxRegion(center=mp.Vector3(z=+r),
                                    size=mp.Vector3(2*r,2*r,0)))
# Funny you can encase the sphere (r radius) so closely (2r-sided box)

temp = time()
sim.init_sim()
elapsed = [time() - temp]
print("elapsed on 1st Building: {} s".format(elapsed[-1]))

"""
8x8x8 with resolution 25
(25 cells inside diameter)
==> Build 3.43 s
"""

#%% FIRST RUN: SIMULATION NEEDED TO NORMALIZE

temp = time()
sim.run(until_after_sources=10)
elapsed = [*elapsed, time()-temp]
print("elapsed on 1st Simulation: {} s".format(elapsed[-1]))

freqs = mp.get_flux_freqs(box_x1)
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

freqs = np.array(freqs)

sim.reset_meep()

#%% FIST RUN: SAVE MID DATA

params = dict(
    from_um_factor=from_um_factor,
    resolution=resolution,
    elapsed=elapsed,
    )

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

#%% SECOND RUN: GEOMETRY SETUP

geometry = [mp.Sphere(material=mp.Medium(index=n_sphere),
                      center=mp.Vector3(),
                      radius=r)]
# Lossless dielectric sphere 
# Wavelength-independent refractive index of 2.0

temp = time()
sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    boundary_layers=pml_layers,
                    sources=sources,
                    k_point=mp.Vector3(),
                    symmetries=symmetries,
                    geometry=geometry)

box_x1 = sim.add_flux(frq_cen, dfrq, nfrq, 
                      mp.FluxRegion(center=mp.Vector3(x=-r),
                                    size=mp.Vector3(0,2*r,2*r)))
box_x2 = sim.add_flux(frq_cen, dfrq, nfrq, 
                      mp.FluxRegion(center=mp.Vector3(x=+r),
                                    size=mp.Vector3(0,2*r,2*r)))
box_y1 = sim.add_flux(frq_cen, dfrq, nfrq, 
                      mp.FluxRegion(center=mp.Vector3(y=-r),
                                    size=mp.Vector3(2*r,0,2*r)))
box_y2 = sim.add_flux(frq_cen, dfrq, nfrq, 
                      mp.FluxRegion(center=mp.Vector3(y=+r),
                                    size=mp.Vector3(2*r,0,2*r)))
box_z1 = sim.add_flux(frq_cen, dfrq, nfrq, 
                      mp.FluxRegion(center=mp.Vector3(z=-r),
                                    size=mp.Vector3(2*r,2*r,0)))
box_z2 = sim.add_flux(frq_cen, dfrq, nfrq, 
                      mp.FluxRegion(center=mp.Vector3(z=+r),
                                    size=mp.Vector3(2*r,2*r,0)))

sim.load_minus_flux_data(box_x1, box_x1_data)
sim.load_minus_flux_data(box_x2, box_x2_data)
sim.load_minus_flux_data(box_y1, box_y1_data)
sim.load_minus_flux_data(box_y2, box_y2_data)
sim.load_minus_flux_data(box_z1, box_z1_data)
sim.load_minus_flux_data(box_z2, box_z2_data)
elapsed = [*elapsed, time()-temp]
print("elapsed on 2nd Building: {} s".format(elapsed[-1]))

#%% SECOND RUN: SIMULATION :D

temp = time()
sim.run(until_after_sources=100)

box_x1_flux = np.asarray(mp.get_fluxes(box_x1))
box_x2_flux = np.asarray(mp.get_fluxes(box_x2))
box_y1_flux = np.asarray(mp.get_fluxes(box_y1))
box_y2_flux = np.asarray(mp.get_fluxes(box_y2))
box_z1_flux = np.asarray(mp.get_fluxes(box_z1))
box_z2_flux = np.asarray(mp.get_fluxes(box_z2))
elapsed = [*elapsed, time()-temp]
print("elapsed on 2nd Simulation: {} s".format(elapsed[-1]))

#%% ANALYSIS

scatt_flux = box_x1_flux - box_x2_flux
scatt_flux = scatt_flux + box_y1_flux - box_y2_flux
scatt_flux = scatt_flux + box_z1_flux - box_z2_flux

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

scatt_eff_theory = [ps.MieQ(n_sphere, 
                            1000/f,
                            2*r*1000,
                            asDict=True)['Qsca'] 
                    for f in freqs]
# The simulation results are validated by comparing with 
# analytic theory of PyMieScatt module

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
plt.savefig(file("Plot.png"))

#%% SAVE FINAL DATA

data = np.array([1e3*from_um_factor/freqs, scatt_eff_meep, scatt_eff_theory]).T

params["elapsed"] = elapsed

header = ["Longitud de onda [nm]", 
          "Sección eficaz efectiva (Meep) [u.a.]", 
          "Sección eficaz efectiva (Theory) [u.a.]"]

data_base = np.array([1e3*from_um_factor/freqs, box_x1_flux0, box_x1_flux,
                      box_x2_flux, box_y1_flux, box_y2_flux, 
                      box_z1_flux, box_z2_flux, 
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
