# -*- coding: utf-8 -*-

# From Meep tutorial: transmission around a 90-degree waveguide bend in 2d

import numpy as np
import meep as mp
import matplotlib.pyplot as plt
#import vsClasses as vcl
import vsPlot as vpl

#%% GEOMETRY LAYOUT: STRAIGHT WAVEGUIDE

resolution = 10 # pixels/um

sx = 16  # amount of cells in X direction
sy = 32  # amount of cells in Y direction

cell = mp.Vector3(sx, sy, 0)

dpml = 1.0 # depth of PML layer
pml_layers = [mp.PML(dpml)]

pad = 4  # padding distance between waveguide and cell edge
w = 1    # width of waveguide

wvg_xcen =  0.5*(sx - w - 2*pad)  # x center of vert. wvg
wvg_ycen = -0.5*(sy - w - 2*pad)  # y center of horiz. wvg

geometry = [mp.Block(size=mp.Vector3(mp.inf, w, mp.inf),
                     center=mp.Vector3(0, wvg_ycen, 0),
                     material=mp.Medium(epsilon=12))]

fcen = 0.15  # pulse center frequency
df = 0.1     # pulse width (in frequency)
sou_xcen = -0.5*sx + dpml # x center of source
sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df),
                     component=mp.Ez,
                     center=mp.Vector3(sou_xcen, wvg_ycen, 0),
                     size=mp.Vector3(0, w, 0))]
# Gaussian source instead of Continuous
# Has a center frequency and a frequency width

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution)

#%% MEASUREMENT CONFIGURATION

# This must be done AFTER specifying the Simulation object 
# because all of the field parameters are initialized 
# when flux planes are created

nfreq = 100  # number of frequencies at which to compute flux
# Sort of resolution in frequency space?

# Reflected flux
refl_fr = mp.FluxRegion(center=mp.Vector3(sou_xcen + 0.5, wvg_ycen, 0),
                        size=mp.Vector3(0, 2*w, 0)) # Geometry 2D => Flux 1D
refl = sim.add_flux(fcen, df, nfreq, refl_fr)
# >> Here the FluxRegion is a line, so the direction is its normal by default: 
# it does not need to be explicitly defined.
# >> Note that flux lines are separated by dpml from the boundary of the cell, 
# so that they do not lie within the absorbing PML regions. 
# >> We only compute fluxes for frequencies within our pulse bandwidth. 
# because far outside the pulse bandwidth, the spectral power is so low 
# that numerical errors make the computed fluxes useless.

# Transmitted flux
tran_fr = mp.FluxRegion(center=mp.Vector3(0.5*sx - dpml, wvg_ycen, 0),
                        size=mp.Vector3(0, 2*w, 0))
tran = sim.add_flux(fcen, df, nfreq, tran_fr)

#%% FIRST SIMULATION: USED TO NORMALIZE LATER :P

pt = mp.Vector3(0.5*sx - dpml - 0.5, wvg_ycen)
sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, pt, 1e-3))
# This is made in order to separate incident and reflected fields
# >> The stop... condition examines the component c at the point pt 
# and keeps running until its absolute value squared has decayed 
# by at least decay_by from its peak value. Between checks, 
# it leaves a dt gap of time.
# >> We need to keep running after the source has turned off 
# because we must give the pulse time to propagate completely across the cell.

# Save flux fields data for reflection plane
straight_refl_data = sim.get_flux_data(refl)
# get... Gets Fourier-transformed fields as NAMED NUMPY ARRAYS
# "This object is only useful for passing to load... and should be considered opaque"

# Save incident power for transmission plane
straight_tran_flux = np.array(mp.get_fluxes(tran))
# get... Gets flux spectrum accumulated as LIST

# Clear simulation
sim.reset_meep()

#%% GEOMETRY LAYOUT: BENT WAVEGUIDE

geometry = [mp.Block(mp.Vector3(sx-pad,w,mp.inf),
                     center=mp.Vector3(-0.5*pad,wvg_ycen),
                     material=mp.Medium(epsilon=12)),
            mp.Block(mp.Vector3(w,sy-pad,mp.inf),
                     center=mp.Vector3(wvg_xcen,0.5*pad),
                     material=mp.Medium(epsilon=12))]

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution)

#%% MEASUREMENT CONFIGURATION

# Reflected flux
refl = sim.add_flux(fcen, df, nfreq, refl_fr)

# Transmitted flux
tran_fr = mp.FluxRegion(center=mp.Vector3(wvg_xcen,0.5*sy-dpml-0.5,0),
                        size=mp.Vector3(2*w,0,0))
tran = sim.add_flux(fcen, df, nfreq, tran_fr)

# Before we start the second run, we load these fields, negated
sim.load_minus_flux_data(refl, straight_refl_data)
# This subtracts the Fourier-transformed incident fields from the Fourier transforms of the scattered fields.
# Turns out it is more convenient to do this before the simulation and then accumulate the Fourier transform.

#%% SECOND SIMULATION: FULL NORMALIZED SIMULATION :D

pt = mp.Vector3(wvg_xcen, 0.5*sy - dpml - 0.5)
sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, pt, 1e-3))

# Get final data :D
bend_refl_flux = np.array(mp.get_fluxes(refl))
bend_tran_flux = np.array(mp.get_fluxes(tran))

flux_freqs = np.array(mp.get_flux_freqs(refl))
# Get what would be on the x axis

#%%

# Reflectance = reflected flux / incident flux
reflectance = -bend_refl_flux/straight_tran_flux
# We also multiply by -1 because Meep fluxes are computed in +coord direction
# and we want the flux in the −x direction

# Transmitance = transmitted flux / incident flux
transmitance = bend_tran_flux/straight_tran_flux

# Scattered loss
scattered_loss = 1 - transmitance - reflectance

# Wavelength
wavelength = 1/flux_freqs
# Again: to have something to plot against

# wl = []
# Rs = []
# Ts = []
# for i in range(nfreq):
#     wl = np.append(wl, 1/flux_freqs[i])
#     Rs = np.append(Rs,-bend_refl_flux[i]/straight_tran_flux[i])
#     Ts = np.append(Ts,bend_tran_flux[i]/straight_tran_flux[i])

fig = plt.figure()
vpl.add_style()
plt.title("Initial results")
plt.plot(wavelength, reflectance,'bo-',label='Reflectance')
plt.plot(wavelength, transmitance,'ro-',label='Transmittance')
plt.plot(wavelength, scattered_loss,'go-',label='Scattered loss')
plt.xlabel("Wavelength (μm)")
plt.legend()
plt.show()
    
# We should also check whether our data is converged. 
# We can do this by increasing the resolution and cell size 
# and seeing by how much the numbers change. In this case.
# Try doubling the cell size!

#%% GEOMETRY LAYOUT: STRAIGHT WAVEGUIDE AGAIN

# Clear simulation
sim.reset_meep()

sx = 32  # amount of cells in X direction
sy = 64  # amount of cells in Y direction
cell = mp.Vector3(sx, sy, 0)

wvg_xcen =  0.5*(sx - w - 2*pad)  # x center of vert. wvg
wvg_ycen = -0.5*(sy - w - 2*pad)  # y center of horiz. wvg

geometry = [mp.Block(size=mp.Vector3(mp.inf, w, mp.inf),
                     center=mp.Vector3(0, wvg_ycen, 0),
                     material=mp.Medium(epsilon=12))]

sou_xcen = -0.5*sx + dpml # x center of source
sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df),
                     component=mp.Ez,
                     center=mp.Vector3(sou_xcen, wvg_ycen, 0),
                     size=mp.Vector3(0, w, 0))]

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution)

#%% MEASUREMENT CONFIGURATION AGAIN

# Reflected flux
refl_fr = mp.FluxRegion(center=mp.Vector3(sou_xcen + 0.5, wvg_ycen, 0),
                        size=mp.Vector3(0, 2*w, 0)) # Geometry 2D => Flux 1D
refl = sim.add_flux(fcen, df, nfreq, refl_fr)

# Transmitted flux
tran_fr = mp.FluxRegion(center=mp.Vector3(0.5*sx - dpml, wvg_ycen, 0),
                        size=mp.Vector3(0, 2*w, 0))
tran = sim.add_flux(fcen, df, nfreq, tran_fr)

#%% FIRST SIMULATION: USED TO NORMALIZE LATER :P

pt = mp.Vector3(0.5*sx - dpml - 0.5, wvg_ycen)
sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, pt, 1e-3))

# Save flux fields data for reflection plane
straight_refl_data = sim.get_flux_data(refl)
# get... Gets Fourier-transformed fields as NAMED NUMPY ARRAYS
# "This object is only useful for passing to load... and should be considered opaque"

# Save incident power for transmission plane
straight_tran_flux_2 = np.array(mp.get_fluxes(tran))
# get... Gets flux spectrum accumulated as LIST

# Clear simulation
sim.reset_meep()

#%% GEOMETRY LAYOUT: BENT WAVEGUIDE

geometry = [mp.Block(mp.Vector3(sx-pad,w,mp.inf),
                     center=mp.Vector3(-0.5*pad,wvg_ycen),
                     material=mp.Medium(epsilon=12)),
            mp.Block(mp.Vector3(w,sy-pad,mp.inf),
                     center=mp.Vector3(wvg_xcen,0.5*pad),
                     material=mp.Medium(epsilon=12))]

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution)

#%% MEASUREMENT CONFIGURATION

# Reflected flux
refl = sim.add_flux(fcen, df, nfreq, refl_fr)

# Transmitted flux
tran_fr = mp.FluxRegion(center=mp.Vector3(wvg_xcen,0.5*sy-dpml-0.5,0),
                        size=mp.Vector3(2*w,0,0))
tran = sim.add_flux(fcen, df, nfreq, tran_fr)

# Before we start the second run, we load these fields, negated
sim.load_minus_flux_data(refl, straight_refl_data)
# This subtracts the Fourier-transformed incident fields from the Fourier transforms of the scattered fields.
# Turns out it is more convenient to do this before the simulation and then accumulate the Fourier transform.

#%% SECOND SIMULATION: FULL NORMALIZED SIMULATION :D

pt = mp.Vector3(wvg_xcen, 0.5*sy - dpml - 0.5)
sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, pt, 1e-3))

# Get final data :D
bend_refl_flux_2 = np.array(mp.get_fluxes(refl))
bend_tran_flux_2 = np.array(mp.get_fluxes(tran))

flux_freqs_2 = np.array(mp.get_flux_freqs(refl))
# Get what would be on the x axis

#%%

# Reflectance = reflected flux / incident flux
reflectance_2 = -bend_refl_flux_2/straight_tran_flux_2
# We also multiply by -1 because Meep fluxes are computed in +coord direction
# and we want the flux in the −x direction

# Transmitance = transmitted flux / incident flux
transmitance_2 = bend_tran_flux_2/straight_tran_flux_2

# Scattered loss
scattered_loss_2 = 1 - transmitance_2 - reflectance_2

# Wavelength
wavelength_2 = 1/flux_freqs_2
# Again: to have something to plot against

# wl = []
# Rs = []
# Ts = []
# for i in range(nfreq):
#     wl = np.append(wl, 1/flux_freqs[i])
#     Rs = np.append(Rs,-bend_refl_flux[i]/straight_tran_flux[i])
#     Ts = np.append(Ts,bend_tran_flux[i]/straight_tran_flux[i])

plt.figure()
vpl.add_style()
plt.title("Final results")
plt.plot(wavelength, reflectance,'bo-',label='Reflectance')
plt.plot(wavelength, transmitance,'ro-',label='Transmittance')
plt.plot(wavelength, scattered_loss,'go-',label='Scattered loss')
plt.plot(wavelength_2, reflectance_2,'k--',label='Double size')
plt.plot(wavelength_2, transmitance_2,'k--')
plt.plot(wavelength_2, scattered_loss_2,'k--')
plt.xlabel("Wavelength (μm)")
plt.legend()
plt.show()