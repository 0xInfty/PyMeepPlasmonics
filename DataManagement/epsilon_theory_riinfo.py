#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 14:19:04 2021

@author: vall
"""

import numpy as np
import v_save as vs
import os

syshome = vs.get_sys_home()

#%%

name = "Ag_R_ComplexN_RIinfo"
paper = "A. D. Rakić, A. B. Djurišic, J. M. Elazar, and M. L. Majewski. Optical properties of metallic films for vertical-cavity optoelectronic devices, Appl. Opt. 37, 5271-5283 (1998)"
reference = "https://refractiveindex.info/?shelf=main&book=Ag&page=Rakic-LD"

#%%

file = lambda name : os.path.join(syshome, "MaterialsData", name)

data = np.loadtxt(file(name + ".csv"),
                  skiprows=1,
                  delimiter=",")

data[:,0] = data[:,0] * 1000 # from um to nm

#%%

header = [r"Wavelength $\lambda$ [nm]", "n", "k"]

footer = {"paper": paper,
          "wlen_range": [min(data[:,0]), max(data[:,0])],
          "reference": reference}

vs.savetxt(file(name + ".txt"), data, footer=footer, header=header)
