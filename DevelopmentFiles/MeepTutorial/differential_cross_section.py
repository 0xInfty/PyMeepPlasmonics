#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# From Meep Tutorial: Differential Cross Section of Mie Sphere

import meep as mp
import numpy as np
import PyMieScatt as ps
import time as tm

start_time = tm.time()

#%% PARAMETERS :D

n_sphere = 2.0
r = 1.0  # radius of sphere

frq_cen = 1.0

dpml = 0.5
dair = 1.5 # at least 0.5/frq_cen padding between source and near-field monitor

resolution = 25 # pixels/um
s = 2*(dpml+dair+r) # full cell characteristic length

npts = 100     # number of points in [0,pi) range of polar angles 
               # to sample far fields along semi-circle
ff_r = 10000*r # radius of far-field semi-circle


#%% FIRST RUN: BASE CONFIGURATION

cell_size = mp.Vector3(s,s,s)

pml_layers = [mp.PML(thickness=dpml)]

# Circularly-polarized source with propagation axis along x
# ==> σdiff is invariant with rotation angle ϕ around x axis 
# ==> Far fields need only be sampled with the polar angle θ
sources = [mp.Source(mp.GaussianSource(frq_cen,
                                       fwidth=0.2*frq_cen,
                                       is_integrated=True),
                     center=mp.Vector3(-0.5*s+dpml),
                     size=mp.Vector3(0,s,s),
                     component=mp.Ez),
           mp.Source(mp.GaussianSource(frq_cen,
                                       fwidth=0.2*frq_cen,
                                       is_integrated=True),
                     center=mp.Vector3(-0.5*s+dpml),
                     size=mp.Vector3(0,s,s),
                     component=mp.Ey,
                     amplitude=1j)] 
# Important! amplitude=1j creates 90º phase difference between them
# Careful! is_integrated=True needed for planewave source that extends into PML
# Circular polarization breaks mirror simmetry ==> Increases simulation size :(

sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    boundary_layers=pml_layers,
                    sources=sources,
                    k_point=mp.Vector3())
# Despite 1j, we save only real fields by default (force_complex_fields=False)
# (force_...=True would double the floating-point memory consumption)

#%% FIRST RUN: NORMALIZATION DATA

box_flux = sim.add_flux(frq_cen, 0, 1,
                        mp.FluxRegion(center=mp.Vector3(x=-2*r),
                                      size=mp.Vector3(0,4*r,4*r)))
# 'near2far' requires the box to lie entirely within the homogeneous air region
# ==> Each side has 2 diameter length

# Watch it! Now we're using 'add_near2far' instead of 'add_flux'
nearfield_box = sim.add_near2far(frq_cen, 0, 1,
                                 mp.Near2FarRegion(center=mp.Vector3(x=-2*r),
                                                   size=mp.Vector3(0,4*r,4*r),
                                                   weight=+1),
                                 mp.Near2FarRegion(center=mp.Vector3(x=+2*r),
                                                   size=mp.Vector3(0,4*r,4*r),
                                                   weight=-1),
                                 mp.Near2FarRegion(center=mp.Vector3(y=-2*r),
                                                   size=mp.Vector3(4*r,0,4*r),
                                                   weight=+1),
                                 mp.Near2FarRegion(center=mp.Vector3(y=+2*r),
                                                   size=mp.Vector3(4*r,0,4*r),
                                                   weight=-1),
                                 mp.Near2FarRegion(center=mp.Vector3(z=-2*r),
                                                   size=mp.Vector3(4*r,4*r,0),
                                                   weight=+1),
                                 mp.Near2FarRegion(center=mp.Vector3(z=+2*r),
                                                   size=mp.Vector3(4*r,4*r,0),
                                                   weight=-1))

sim.run(until_after_sources=10)

input_flux = mp.get_fluxes(box_flux)[0]
nearfield_box_data = sim.get_near2far_data(nearfield_box)
# Watch it! Now we're also using 'get_near2far_data'

sim.reset_meep()

#%% SECOND RUN: FULL CONFIGURATION :)

geometry = [mp.Sphere(material=mp.Medium(index=n_sphere),
                      center=mp.Vector3(),
                      radius=r)]

sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    boundary_layers=pml_layers,
                    sources=sources,
                    k_point=mp.Vector3(),
                    geometry=geometry)

#%% SECOND RUN: FULL SIMULATION :D

nearfield_box = sim.add_near2far(frq_cen, 0, 1,
                                 mp.Near2FarRegion(center=mp.Vector3(x=-2*r),
                                                   size=mp.Vector3(0,4*r,4*r),
                                                   weight=+1),
                                 mp.Near2FarRegion(center=mp.Vector3(x=+2*r),
                                                   size=mp.Vector3(0,4*r,4*r),
                                                   weight=-1),
                                 mp.Near2FarRegion(center=mp.Vector3(y=-2*r),
                                                   size=mp.Vector3(4*r,0,4*r),
                                                   weight=+1),
                                 mp.Near2FarRegion(center=mp.Vector3(y=+2*r),
                                                   size=mp.Vector3(4*r,0,4*r),
                                                   weight=-1),
                                 mp.Near2FarRegion(center=mp.Vector3(z=-2*r),
                                                   size=mp.Vector3(4*r,4*r,0),
                                                   weight=+1),
                                 mp.Near2FarRegion(center=mp.Vector3(z=+2*r),
                                                   size=mp.Vector3(4*r,4*r,0),
                                                   weight=-1))

sim.load_minus_near2far_data(nearfield_box, nearfield_box_data)

sim.run(until_after_sources=100)

# Computing σdiff in a given direction involves three base steps: 
# (1) Near fields on a closed box surrounding the object
# (2) From near fields, compute far fields at single point far away (R >> 2r)
# (3) Calculate Poynting flux of far fields in outward direction: Real[E∗ × H]

# Then two computations based on those three steps:
# (A) Differential cross section = scattered power per solid angle
# ==> σdiff in outward direction = R².F / incident intensity
# (B) Scattering cross section = total scattered power / incident intensity
# ==> σscatt = integral of σdif over all spherical angles

angles = np.pi/npts*np.arange(npts)
E = np.zeros((npts,3), dtype=np.complex128)
H = np.zeros((npts,3), dtype=np.complex128)
for n in range(npts):
    ff = sim.get_farfield(nearfield_box, 
                          ff_r*mp.Vector3(np.cos(angles[n]),
                                          0,
                                          np.sin(angles[n])))
    E[n,:] = [np.conj(ff[j]) for j in range(3)] # conj: complexe conjugate
    H[n,:] = [ff[j+3] for j in range(3)]

Px = np.real(np.multiply(E[:,1],H[:,2]) - np.multiply(E[:,2],H[:,1]))
Py = np.real(np.multiply(E[:,2],H[:,0]) - np.multiply(E[:,0],H[:,2]))
Pz = np.real(np.multiply(E[:,0],H[:,1]) - np.multiply(E[:,1],H[:,0]))
Pr = np.sqrt(np.square(Px) + np.square(Py) + np.square(Pz))

intensity = input_flux / (4*r)**2
diff_cross_section = ff_r**2 * Pr / intensity
scatt_cross_section_meep = 2*np.pi * np.sum(np.multiply(
    diff_cross_section,
    np.sin(angles))) * np.pi/npts
scatt_cross_section_theory = ps.MieQ(n_sphere,
                                     1000/frq_cen,
                                     2*r*1000,
                                     asDict=True,
                                     asCrossSection=True)['Csca']*1e-6 # um^2

stop_time = tm.time()

print(">> Results x resolution={:.0f} <<".format(resolution))
print("- σscatt x meep: {:.16f} \n- σscatt x theory: {:.16f}".format(
    scatt_cross_section_meep, scatt_cross_section_theory))
print("- absolute error: {:.3e}".format(
    abs(scatt_cross_section_meep - scatt_cross_section_theory)))
print("- relative error: {:.2f}%".format(
    100*abs(scatt_cross_section_meep - scatt_cross_section_theory)/scatt_cross_section_theory))
print("- elapsed time: {:.2f} s".format(stop_time - start_time))

#%%


"""
>> Results x resolution=20 <<
- σscatt x meep: 8.1554468215885638 
- σscatt x theory: 8.3429545590438750
- absolute error: 1.875e-01
- relative error: 2.25%
- elapsed time: 410.43 s

>> Results x resolution=25 <<
- σscatt x meep: 8.2215435272741395 
- σscatt x theory: 8.3429545590438750
- absolute error: 1.214e-01
- relative error: 1.46%
- elapsed time: 934.35 s
"""