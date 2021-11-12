# -*- coding: utf-8 -*-

# From the Meep tutorial: plotting permittivity and fields of a straight waveguide

#%% SIMULATION

#from __future__ import division

import meep as mp

cell = mp.Vector3(16, 8, 0) # length 16 μm, width 8 μm, depth 0 μm

geometry = [mp.Block(mp.Vector3(mp.inf,1,mp.inf),
                     center=mp.Vector3(), # centered at 0,0,0
                     material=mp.Medium(epsilon=12))]
# Waveguide --> Block (parallelepiped infty length, 1 μm width, infty height)
# Medium --> Dielectric relative constant ε=12
# Default --> Any place where there are no objects there is air (=1)

sources = [mp.Source(mp.ContinuousSource(frequency=0.15),
                     component=mp.Ez, # Electric current Jz
                     center=mp.Vector3(-7,0))]
# Fixed-frequency sinusoid exp(−iωt) that by default is turned on at t=0
# Frequency --> 0.15 * (2πc) --> 2πc = inverse of the vacuum wavelength
# ==> Vacuum wavelength of about 1/0.15=6.67 μm
# ==> Wavelength of about 2 μm in the ε=12 material 
# ==> Waveguide half a wavelength wide
# ==> Hopefully single mode anallytically = to roughly 0.16076
# Center at (-7,0), which is 1 μm to the right of the left edge of the cell 
# It's better if some space between sources and cell boundaries, 
# to keep the boundary conditions from interfering with them.

pml_layers = [mp.PML(1)]
# Absorbing boundaries simulate vacuum (pure transmition, no reflections)
# 1 mum thickness ==> It's inside the cell!
# It overlaps our waveguide, so that it absorbs waveguide modes.

resolution = 10
# Number of pixels per unit of length (1 μm)
# ==> 160×80 cell
# In gral, at least 8 pixels/wavelength in the highest ε medium

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution)

sim.run(until=200)

#%% ANALYSIS

#import numpy as np
import matplotlib.pyplot as plt

eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
plt.figure()
plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
plt.axis('off') # piola
plt.show()
# There is a borderline with ε neither 1 nor 12 
# I guess it's the interpolation for the waveguide
# plt.savefig("/nfs/home/vpais/PyMeepResults")

ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
plt.figure()
plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
plt.imshow(ez_data.transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9)
plt.plot((cell.x/2-7)*resolution,
         (cell.y/2-0)*resolution,
         "wX")
plt.axis('off')
plt.show()
if mp.am_master():
    plt.savefig(f"/nfs/home/vpais/PyMeepResults/Test/MeepTutTUPAC/NP{mp.count_processors():.0f}.png")