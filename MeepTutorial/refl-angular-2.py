# -*- coding: utf-8 -*-

# Modified from Meep Tutorial: Planar interface :)

import numpy as np
import meep as mp
import math
# import h5py as h5
# import os
import matplotlib.pyplot as plt

#%%

def Fresnel_planar_interface(theta, resolution=200):
    """Reflectance and transmittance on planar interface.
    
    Parameters
    ----------
    theta : int, float
        Angle of incidence CCW from the normal, expressed in degrees.
    resolution=200 : int, optional
        Cells per μm.
    
    Returns
    -------
    
    """

    dpml = 1.0                      # PML thickness                
    sz = 10 + 2*dpml                # 10 is size of cell without PMLs
    # A 1d cell must be used since a higher-dimensional cell 
    # will introduce artificial modes due to band folding.
    
    cell_size = mp.Vector3(0, 0, sz)
    pml_layers = [mp.PML(dpml)]

    # Gaussian w/visible spectrum
    wvl_min = 0.4                   # min wavelength
    wvl_max = 0.8                   # max wavelength
    fmin = 1/wvl_max                # min frequency
    fmax = 1/wvl_min                # max frequency
    fcen = 0.5*(fmin+fmax)          # center frequency
    df = fmax-fmin                  # frequency width
    nfreq = 50                      # number of frequency bins
    # Unlike a continuous-wave (CW) source, a pulsed source turns off. 
    
    # Rotation angle of planewave
    # (input in degrees and transformed to radians)
    theta_r = math.radians(theta)
    # Counterclockwise (CCW) around Y axis
    # 0 degrees along +Z axis

    # Plane of incidence is XZ
    k = mp.Vector3(math.sin(theta_r), 0, math.cos(theta_r)).scale(fmin)
    # Where k is given by dispersion relation formula 
    # ω = c|k⃗|/n (planewave in homogeneous media of index n)
    # As the source here is incident from air, 
    # |k⃗| = ω

    # >> Note that a fixed wavevector only applies to a single frequency. 
    # Any broadband source is incident at a given angle 
    # for only a single frequency. 
    # >> In order to model the S-polarization, we must use an Ey source. 
    # This example involves just the P-polarization.
    
    # If normal incidence, force number of dimensions to be 1
    if theta_r == 0:
        dimensions = 1
    else:
        dimensions = 3
    # In Meep, a 1d cell is defined along the z direction. 
    # When k⃗  is not set, only the Ex and Hy field components are permitted.
    
    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df), 
                         component=mp.Ex, 
                         center=mp.Vector3(0,0,-0.5*sz+dpml))]

    sim = mp.Simulation(cell_size=cell_size,
                        boundary_layers=pml_layers,
                        sources=sources,
                        k_point=k,
                        dimensions=dimensions,
                        resolution=resolution)

    refl_fr = mp.FluxRegion(center=mp.Vector3(0, 0, -0.25*sz))
    refl = sim.add_flux(fcen, df, nfreq, refl_fr)
    
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ex, mp.Vector3(0, 0, -0.5*sz+dpml), 1e-9))

    empty_flux = mp.get_fluxes(refl)
    empty_data = sim.get_flux_data(refl)
    sim.reset_meep()

    # add a block with n=3.5 for the air-dielectric interface
    geometry = [mp.Block(mp.Vector3(mp.inf, mp.inf, 0.5*sz), 
                         center=mp.Vector3(0, 0, 0.25*sz), 
                         material=mp.Medium(index=3.5))]

    sim = mp.Simulation(cell_size=cell_size,
                        geometry=geometry,
                        boundary_layers=pml_layers,
                        sources=sources,
                        k_point=k,
                        dimensions=dimensions,
                        resolution=resolution)

    refl = sim.add_flux(fcen, df, nfreq, refl_fr)
    sim.load_minus_flux_data(refl, empty_data)

    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ex, mp.Vector3(0, 0, -0.5*sz+dpml), 1e-9))

    refl_flux = np.array(mp.get_fluxes(refl))
    freqs = np.array(mp.get_flux_freqs(refl))
    wavelengths = 1/freqs
    angles = np.degrees( np.arcsin(k.x/freqs) ) # 90 if freq = freqmin?
    # Angle for the (kx,ω) pair.
    reflectances = -refl_flux / np.array(empty_flux)
       
    # f = h5.File("MeepTutorial/FresnelResults/FresnelResults.h5", "w")
    # f.create_dataset("")
    
    return k.x, wavelengths, angles, reflectances

#%%

kx, wavelengths, angles, reflectances = Fresnel_planar_interface(0)

plt.plot(wavelengths, reflectances)