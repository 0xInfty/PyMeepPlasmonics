#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 15:46:56 2020

@author: vall
"""

import h5py as h5
import meep as mp
# import numpy as np
import os
from time import time
import v_save as vs

#%% PARAMETERS

wlen = 1
is_integrated = True # Default: False

pml_width = round(.38 * wlen, 2) 
# For 1, 0.1 source parameters, 0.4 to be sure but 0.39 doesn't look bad.
# So: 35% or 40% of larger wavelength apparently works OK.

cell_width = 12
resolution = 10

period_line = 1/10
period_plane = 1

time_is_after_source = False
run_time = 20

plane_center = [0,0,0]
line_center = [0,0,0]

plane_size = [0, cell_width, cell_width]
line_size = [cell_width, 0, 0]

series = "1st"
folder = "DataManagement"
home = r"/home/vall/Documents/Thesis/ThesisPython"

parameters = dict(
    wlen=wlen,
    is_integrated=is_integrated,
    pml_width=pml_width,
    cell_width=cell_width,
    resolution=resolution,
    period_line=period_line,
    period_plane=period_plane,
    plane_center=plane_center,
    line_center=line_center,
    plane_size=plane_size,
    line_size=line_size,
    time_is_after_source=time_is_after_source,
    run_time=run_time,
    series=series,
    home=home)

#%% LAYOUT CONFIGURATION

cell_size = mp.Vector3(cell_width, cell_width, cell_width)

boundary_layers = [mp.PML(thickness=pml_width)]

source_center = -0.5*cell_width + pml_width
sources = [mp.Source(mp.ContinuousSource(wavelength=wlen, 
                                         is_integrated=is_integrated),
                     center=mp.Vector3(source_center),
                     size=mp.Vector3(0, cell_width, cell_width),
                     component=mp.Ez)]

symmetries = [mp.Mirror(mp.Y), mp.Mirror(mp.Z, phase=-1)]

elapsed = []

path = os.path.join(home, folder, "{}Results".format(series))
if not os.path.isdir(path): vs.new_dir(path)
file = lambda f : os.path.join(path, f)

#%% OPTION 1: SAVE GET FUNCTIONS ##############################################

def get_line(sim):
    return sim.get_array(
        center=mp.Vector3(*line_center), 
        size=mp.Vector3(*line_size), 
        component=mp.Ez)

def get_plane(sim):
    return sim.get_array(
        center=mp.Vector3(*plane_center), 
        size=mp.Vector3(*plane_size), 
        component=mp.Ez)

#%% INITIALIZE

sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    boundary_layers=boundary_layers,
                    sources=sources,
                    symmetries=symmetries)

sim.init_sim()

#%% DEFINE SAVE STEP FUNCTIONS

f, save_line = vs.save_slice_generator(sim, file("Lines.h5"), "Ez", get_line)
g, save_plane = vs.save_slice_generator(sim, file("Planes.h5"), "Ez", get_plane)

to_do_while_running = [mp.at_every(period_line, save_line),
                       mp.at_every(period_plane, save_plane)]

#%% RUN!

temp = time()
if time_is_after_source:
    sim.run(*to_do_while_running, until_after_source=run_time)
else:
    sim.run(*to_do_while_running, until=run_time)
del f, g
elapsed.append(time() - temp)

#%% SAVE METADATA

f = h5.File(file("Lines.h5"), "r+")
for a in parameters: f["Ez"].attrs[a] = parameters[a]
f.close()
del f

g = h5.File(file("Planes.h5"), "r+")
for a in parameters: g["Ez"].attrs[a] = parameters[a]
g.close()
del g

#%% GET READY TO LOAD DATA

# f = h5.File(file("Lines.h5"), "r")
# results_line = f["Ez"]

# g = h5.File(file("Planes.h5"), "r")
# results_plane = g["Ez"]

#%% OPTION 2: JUST APPEND FUNCTIONS ###########################################

results_plane = []
results_line = []

def get_slice_plane(sim):
    results_plane.append(sim.get_array(
        center=mp.Vector3(*plane_center), 
        size=mp.Vector3(*plane_size), 
        component=mp.Ez))

def get_slice_line(sim):
    results_line.append(sim.get_array(
        center=mp.Vector3(*line_center), 
        size=mp.Vector3(*line_size), 
        component=mp.Ez))

to_do_while_running = [mp.at_beginning(get_slice_line),
                       mp.at_beginning(get_slice_plane),
                       mp.at_every(period_line, get_slice_line),
                       mp.at_every(period_plane, get_slice_plane)]

#%% INITIALIZE

sim.reset_meep()

sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    boundary_layers=boundary_layers,
                    sources=sources,
                    symmetries=symmetries)

sim.init_sim()

#%% RUN!

temp = time()
if time_is_after_source:
    sim.run(*to_do_while_running, until_after_source=run_time)
else:
    sim.run(*to_do_while_running, until=run_time)
elapsed.append(time() - temp)

#%% SAVE DATA!

vs.savetxt(file("Lines.txt"), results_line, footer=parameters, overwrite=True)
del results_line, results_plane

#%% OPTION 3: SAVE FULL FIELD ################################################

to_do_while_running = [mp.to_appended("Ez", 
                                      mp.at_beginning(mp.output_efield_z),
                                      mp.at_every(period_line, 
                                                  mp.output_efield_z))]

#%% INITIALIZE

sim.reset_meep()

sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    boundary_layers=boundary_layers,
                    sources=sources,
                    symmetries=symmetries,
                    filename_prefix="")
sim.use_output_directory(path)

sim.init_sim()

#%% RUN!

temp = time()
if time_is_after_source:
    sim.run(*to_do_while_running, until_after_source=run_time)
else:
    sim.run(*to_do_while_running, until=run_time)
elapsed.append(time() - temp)

#%% SAVE METADATA

h = h5.File(file('Ez.h5'), "r+")
for a in parameters: h["ez"].attrs[a] = parameters[a]
h.close()
del h

#%% GET READY TO LOAD DATA

# h = h5.File(file("Ez.h5"), "r")
# results_field = h["Ez"]

#%% OPTION 4: USE MY FUNCTIONS TO SAVE WHOLE FIELD ############################

def get_field(sim):
    return sim.get_efield_z()

#%% INITIALIZE

sim.reset_meep()

sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    boundary_layers=boundary_layers,
                    sources=sources,
                    symmetries=symmetries)

sim.init_sim()

#%% DEFINE SAVE STEP FUNCTIONS

h, save_field = vs.save_slice_generator(sim, file("Field.h5"), "Ez", get_field)

to_do_while_running = [mp.at_every(period_line, save_field)]

#%% RUN!

temp = time()
if time_is_after_source:
    sim.run(*to_do_while_running, until_after_source=run_time)
else:
    sim.run(*to_do_while_running, until=run_time)
del h
elapsed.append(time() - temp)

#%% SAVE METADATA

h = h5.File(file("Field.h5"), "r+")
for a in parameters: h["Ez"].attrs[a] = parameters[a]
h.close()
del h

#%% GET READY TO LOAD DATA

# h = h5.File(file("Field.h5"), "r")
# results_field = h["Ez"]
