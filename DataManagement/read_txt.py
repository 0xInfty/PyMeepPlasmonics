#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 12:24:04 2020

@author: vall
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import v_save as vs

#%% PARAMETERS

# Saving directories
series = ["2020111101",
          "2020111103"]
folder = "AuMieResults"
home = "/home/vall/Documents/Thesis/ThesisPython/"

#%% LOAD DATA

file = lambda f, s : os.path.join(home, folder, "{}Results".format(s), f)

data = []
params = []
for s in series:
    data.append(np.loadtxt(file("MidFlux.txt", s)))
    params.append(vs.retrieve_footer(file("MidFlux.txt", s)))
header = vs.retrieve_header(file("MidFlux.txt", s))

r = [p["r"] for p in params]
resolution = [p["resolution"] for p in params]
from_um_factor  = [p["from_um_factor"] for p in params]
until_after_sources = [p["until_after_sources"] for p in params]

#%% PLOT

new_series = ["F8"]#["", "F1", "F2"]
def proposed_factor(series, r, resolution, from_um_factor, until_after_sources):
    if series=="":
        return 1
    elif series=="F1":
        return from_um_factor**4 * r**2
    elif series=="F2":
        return from_um_factor**2 / ( (r**2)*(resolution**2) )
    elif series=="F3":
        return from_um_factor**4 * r**2 * until_after_sources
    elif series=="F4":
        return from_um_factor**5 * r**2 * until_after_sources
    elif series=="F5":
        return from_um_factor**4 * r**2 / until_after_sources
    elif series=="F6":
        return from_um_factor**3 / until_after_sources
    elif series=="F7":
        return from_um_factor**4 * (until_after_sources*(r**2))
    elif series=="F8":
        return from_um_factor**4 * (2*r)**2
    else:
        raise ValueError("Series factor not defined")

for s, di, ri, resi, fromi, unti in zip(series, data, r, resolution, 
                                        from_um_factor, until_after_sources):
    
    for ns in new_series:

        ylims = (np.min(di[:,1:]), np.max(di[:,1:]))
        ylims = (ylims[0]-.1*(ylims[1]-ylims[0]),
                 ylims[1]+.1*(ylims[1]-ylims[0]))
        ylims = np.array(ylims) * proposed_factor(ns, ri, resi, fromi, unti)
        
        fig, ax = plt.subplots(3, 2, sharex=True)
        fig.subplots_adjust(hspace=0, wspace=.05)
        for a in ax[:,1]:
            a.yaxis.tick_right()
            a.yaxis.set_label_position("right")
        for a, h in zip(np.reshape(ax, 6), header[1:]):
            a.set_ylabel(h)
        
        for d, a in zip(di[:,1:].T, np.reshape(ax, 6)):
            a.plot(1e3*fromi/di[:,0], d * proposed_factor(ns, ri, resi, fromi, unti))
            a.set_ylim(*ylims)
        ax[-1,0].set_xlabel("Wavelength [nm]")
        ax[-1,1].set_xlabel("Wavelength [nm]")
        
        plt.savefig(file("MidFlux{}.png".format(ns), s))
        
        plt.close()

#%% PLOT TOGETHER

ylims = (np.min([ds[:,1:] for ds in data]), 
         np.max([ds[:,1:] for ds in data]))
ylims = (ylims[0]-.1*(ylims[1]-ylims[0]),
         ylims[1]+.1*(ylims[1]-ylims[0]))

fig, ax = plt.subplots(3, 2, sharex=True)
fig.subplots_adjust(hspace=0, wspace=.05)
for a in ax[:,1]:
    a.yaxis.tick_right()
    a.yaxis.set_label_position("right")
for a, h in zip(np.reshape(ax, 6), header[1:]):
    a.set_ylabel(h)

for ds, fromi in zip(data, from_um_factor):
    for d, a in zip(ds[:,1:].T, np.reshape(ax, 6)):
        a.plot(1e3*fromi/ds[:,0], d)
        a.set_ylim(*ylims)
ax[-1,0].set_xlabel("Wavelength [nm]")
ax[-1,1].set_xlabel("Wavelength [nm]")

plt.savefig(file("MidFlux.png"))