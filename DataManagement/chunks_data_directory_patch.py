#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 14:34:26 2021

@author: vall
"""

import v_meep as vm
import v_save as vs
import h5py as h5
import numpy as np
import os

sysname = vs.get_sys_name()
syshome = vs.get_sys_home()
home = vs.get_home()

#%%

chunks_file = os.path.join(home, "ChunksData/ChunksDataDirectory.txt")

chunks_dir = vs.retrieve_footer(chunks_file)

#%%

data_files = chunks_dir["path"]

found_files = []
found_params = []
for df in data_files:
    try:
        found_params.append(vs.retrieve_footer(os.path.join(df, "Results.txt")))
        found_files.append(df)
    except:
        print("No file found at ", df)
print(f"Only found {100*len(found_files)/len(data_files):.2f}% of the files")

#%%

resolution = np.array([chunks_dir["resolution"]])
from_um_factor = np.array(chunks_dir["from_um_factor"])
cell_width = np.array(chunks_dir["cell_width"])
pml_width = np.array(chunks_dir["pml_width"])
air_r_width = cell_width - 2*pml_width

r = air_r_width/3
r_nm = r * from_um_factor * 1e3

typical_d_nm = np.array([48, 64, 80, 103, 120])
typical_r_nm = typical_d_nm / 2

index = []
r_nm_reconstructed = []
for r_nm_i in r_nm:
    index.append( np.argmin(abs(typical_r_nm - r_nm_i)) )
    r_nm_reconstructed.append( typical_r_nm[index[-1]] )

r_reconstructed = r_nm_reconstructed / (from_um_factor*1e3)
r_reconstructed[-8:] = np.array([5.15]*8)

flux_box_size = 2*r_reconstructed
flux_box_size[-8:] = np.array(chunks_dir["flux_box_size"][-8:])

chunks_dir["flux_box_size"] = list(flux_box_size)

#%%

keys = list(chunks_dir.keys())
n = len(chunks_dir[keys[0]])

#%%

chunks_dir["n_cores"] = []
for i in range(n):
    chunks_dir["n_cores"].append( min([4, chunks_dir["n_processes"][i]]) )

#%%

submerged_index = []
surface_index = []
for nsubm, nsurf in zip(chunks_dir["submerged_index"], chunks_dir["surface_index"]):
    if nsurf == 1 and nsubm != 1:
        nsurf = nsubm
    submerged_index.append(nsubm)
    surface_index.append(nsurf)

chunks_dir["submerged_index"] = submerged_index
chunks_dir["surface_index"] = surface_index

#%%

index = []
for i in range(len(chunks_dir["path"])-1):
    if chunks_dir["path"][i+1] != chunks_dir["path"][i]:
        index.append(i)

#%%

new_chunks_dir = {}
for key, values_list in chunks_dir.items():
    chunks_dir[key] = [values_list[i] for i in index]
    
#%%

differences = []
for k, v_list in chunks_dir.items():
    if v_list[-1] != v_list[-2]:
        differences.append(k)

f0 = h5.File(os.path.join(home, "ChunksData", chunks_dir["chunks_path"][-2], "Layout.h5"), "r")
ff = h5.File(os.path.join(home, "ChunksData", chunks_dir["chunks_path"][-1], "Layout.h5"), "r")

#%%

for k in keys:
    chunks_dir[k] = chunks_dir[k][:-1]

#%%

vs.savetxt(chunks_file, np.array([]), footer=chunks_dir, overwrite=True)

#%%

blank_chunks_dir = {}
for k in keys: blank_chunks_dir[k] = []

vs.savetxt(os.path.join(home, "ChunksData/ChunksDataDirBlank.txt"), 
           np.array([]), footer=blank_chunks_dir, overwrite=True)