#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 17:05:09 2021

Script to work with surface temperature and Gaussian beam power on spherical NP.

It treats surface temperature increasements due to a focused Gaussian beam 
power on a spherical nanoparticle submerged in an isotropic medium.

- Calculates surface temperature increasement for a given Gaussian beam power 
  with known central wavelength and beam waist.
- Computes Gaussian beam power for a given surface temperature increasement 
  with known central wavelength and beam waist
- Returns other laser's required power to produce the same surface temperature 
  increasement as a reference Gaussian beam when all central wavelengths and 
  beam waists are known.

@author: vall
"""

import numpy as np
import v_theory as vt
import vmp_materials as vmt

#%% PARAMETERS

# Spherical NP
material = "Au"
paper = "JC"
reference = "RIinfo"
r = 30 # Radius in nm

# Surrounding medium
surrounding_N = 1.33 # Water complex refractive index, dimensionless
surrounding_kappa = 0.58 # Water thermal conductivity in W/Km

# Incident Laser
wlen = 532 # Wavelength in nm
P = 0.46 # Power in mW
w0 = 266 # w0 is the Gaussian beam's waist in nm
lasers = [dict(color="Azul", wlen=405, w0=266),
          dict(color="Verde", wlen=532, w0=266),
          dict(color="Rojo", wlen=642, w0=266)]

# Surface Temperature
delta_T = 214 # Surface temperature initial increasement caused by the beam in K.

# Mode
mode = "RefT" # Either 'RefT' or 'RefP'

#%% 

inner_epsilon_function = vmt.epsilon_function(material, paper, reference)
inner_N = np.sqrt(inner_epsilon_function(wlen))

sigma_abs = vt.sigma_abs_Mie(r, wlen, inner_N, surrounding_N)

estimated_delta_T = vt.delta_T(P, sigma_abs, w0, r, surrounding_kappa)

estimated_P = vt.P(delta_T, sigma_abs, w0, r, surrounding_kappa)

#%%

ref_laser = lasers[[l["wlen"] for l in lasers].index(wlen)]

for l in lasers:
    inner_N = np.sqrt(inner_epsilon_function(wlen))
    l["sigma_abs"] = vt.sigma_abs_Mie(r, l["wlen"], inner_N, surrounding_N)

ref_laser["P"] = P
for l in lasers:
    if l!=ref_laser:
        l["P"] = (l["w0"]/ref_laser["w0"])**2 
        l["P"] = l["P"] * (ref_laser["sigma_abs"]/l["sigma_abs"])
        l["P"] = l["P"] * ref_laser["P"]

print(f"Para referencia {ref_laser['color'].lower()} con potencia {ref_laser['P']:.3f} mW...")
for l in lasers:
    if l!=ref_laser:
        print(f">> {l['color']} requiere potencia {round(l['P'],3)} mW")