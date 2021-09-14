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

norm_file = os.path.join(home, "FieldData/FieldDataDirectory.txt")

norm_dir = vs.retrieve_footer(norm_file)

#%%

keys = list(norm_dir.keys())
n = len(norm_dir[keys[0]])

#%%

for k in keys:
    norm_dir[k] = norm_dir[k][-1:]

#%%

vs.savetxt(norm_file, np.array([]), footer=norm_dir, overwrite=True)

#%%

blank_norm_file = {}
blank_norm_file["norm_path"] = []
for k in vm.normfield_key_params: blank_norm_file[k] = []
blank_norm_file["path"] = []

vs.savetxt(os.path.join(home, "FieldData/FieldDataDirBlank.txt"),
           np.array([]), footer=blank_norm_file, overwrite=True)