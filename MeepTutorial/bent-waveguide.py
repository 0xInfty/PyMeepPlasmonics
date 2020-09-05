# -*- coding: utf-8 -*-

# From the Meep tutorial: plotting permittivity and fields of a bent waveguide

#%% SIMULATION

#from __future__ import division

import os as os
import meep as mp

path = "/home/vall/Documents/Thesis/Python/MeepTutorial/BentWaveguideResults"
prefix = "BentWaveguide"

cell = mp.Vector3(16,16,0)
geometry = [mp.Block(mp.Vector3(12,1,mp.inf),
                     center=mp.Vector3(-2.5,-3.5),
                     material=mp.Medium(epsilon=12)),
            mp.Block(mp.Vector3(1,12,mp.inf),
                     center=mp.Vector3(3.5,2),
                     material=mp.Medium(epsilon=12))]
pml_layers = [mp.PML(1.0)]
resolution = 10

sources = [mp.Source(mp.ContinuousSource(wavelength=2*(11**0.5), width=20),
                     component=mp.Ez,
                     center=mp.Vector3(-7,-3.5),
                     size=mp.Vector3(0,1))]
# Instead of freq, you can choose wavelength
# You can have a line-source instead of a punctual source

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution,
                    filename_prefix=prefix)
os.chdir(path)

sim.run(mp.at_beginning(mp.output_epsilon),
        mp.to_appended("ez", mp.at_every(0.6, mp.output_efield_z)),
        until=200)

#%% ANALYSIS

import h5py as h5
import numpy as np

filename = os.path.join(path, prefix + "-ez.h5")
f = h5.File(filename,"r")

#list(f.keys()) # group ~ dictionary
#f["ez"].shape # datasheet ~ Numpy array
#f["ez"][0,0,:].shape

#f.create_dataset("larala", data=np.ones(100))
#f.create_group("folder")
#f.create_dataset("folder2/larala", data=np.ones(100))

#f["larala"].attrs["date"] = "20200908"
#f["larala"].attrs["comments"] = "No hay cambios"
#myattr = dict(f["larala"].attrs)

f.close()
