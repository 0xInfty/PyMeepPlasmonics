#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 12:51:23 2020

@author: vall
"""

import h5py as h5
import os
import v_save as vs

#%%

# Saving directories
series = ["2020102601","2020102708","2020102709",
          *["202010271"+str(i) for i in range(3)]]
folder = "MieResults"
home = "/home/vall/Documents/Thesis/ThesisPython/MeepTutorial/"

#%%

params = []
for s in series:
    
    path = os.path.join(home, folder, "{}Results".format(s))
    file = lambda f : os.path.join(path, f)
    
    p = vs.retrieve_footer(file("MidFlux.txt"))
    
    f = h5.File(file("MidField.h5"), "r+")
    for a in p.keys(): f["Ez"].attrs[a] = p[a]
    f.close()
    
    params.append(p)

#%%

resolution = [p["resolution"] for p in params]
pml_width = [p["pml_width"] for p in params]
air_width = [p["air_width"] for p in params]
cutoff = [p["cutoff"] for p in params]
until_after_sources = [p["until_after_sources"] for p in params]