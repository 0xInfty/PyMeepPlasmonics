#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 16:18:00 2021

@author: vall
"""

from socket import gethostname
if "Nano" in gethostname():
    syshome = "/home/nanofisica/Documents/Vale/ThesisPython"
elif "vall" in gethostname():
    syshome = "/home/vall/Documents/Thesis/ThesisPython"
else:
    raise ValueError("Your PC must be registered at the top of this code")

import sys
sys.path.append(syshome)

import numpy as np
import matplotlib.pyplot as plt
import PyMieScatt as ps
from v_meep import import_medium

#%% PARAMETERS

# Units: 10 nm as length unit
from_um_factor = 10e-3 # Conversion of 1 μm to my length unit (=10nm/1μm)
resolution = 3 # >=8 pixels per smallest wavelength, i.e. np.floor(8/wvl_min)

# Au sphere
r = [2.4, 3.2, 4, 5.15]  # Radius of sphere: diameters 40, 64, 80 and 103 nm
medium = import_medium("Au", from_um_factor) # Medium of sphere: gold (Au)
wlen_range = np.array([45, 60]) # 450 nm to 600 nm

#%% MORE PARAMETERS

freq_range = 1/wlen_range # Hz range in Meep units from highest to lowest
freq_center = np.mean(freq_range)
freq_width = max(freq_range) - min(freq_range)
freqs = np.linspace(min(freq_range), max(freq_range), 100)

#%% THEORY
scatt_eff_theory = [[ps.MieQ(np.sqrt(medium.epsilon(f)[0,0]*medium.mu(f)[0,0]), 
                            1e3*from_um_factor/f,
                            2*r*1e3*from_um_factor,
                            asDict=True)['Qsca'] 
                    for f in freqs] for r in [2, 3.2, 4, 5.15]]
scatt_eff_theory = np.array(scatt_eff_theory)
scatt_eff_theory = scatt_eff_theory.T
scatt_eff_theory = np.array([(s - min(s))/(max(s) - min(s)) for s in scatt_eff_theory.T]).T
plt.plot(1e3*from_um_factor/freqs, scatt_eff_theory)
plt.xlabel("Longitud de onda (nm)")
plt.ylabel("Sección eficaz de scattering")

plt.legend([f"{ int(1e3*2*ri*from_um_factor) } nm" for ri in r])