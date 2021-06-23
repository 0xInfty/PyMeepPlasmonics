#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 14:34:26 2021

@author: vall
"""

import v_meep as vm
import v_save as vs
import numpy as np
import os

sysname = vs.get_sys_name()
syshome = vs.get_sys_home()
home = vs.get_home()

#%%

flux_file = os.path.join(home, "FluxData/FluxDataDirectory.txt")

flux_dir = vs.retrieve_footer(flux_file)

data_files = flux_dir["path"]

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

resolution = np.array([flux_dir["resolution"]])
from_um_factor = np.array(flux_dir["from_um_factor"])
cell_width = np.array(flux_dir["cell_width"])
pml_width = np.array(flux_dir["pml_width"])
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
flux_box_size[-8:] = np.array(flux_dir["flux_box_size"][-8:])

flux_dir["flux_box_size"] = list(flux_box_size)

#%%

vs.savetxt(flux_file, np.array([]), footer=flux_dir, overwrite=True)
