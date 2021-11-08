#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 12:24:04 2020

@author: vall
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plab
import os
import PyMieScatt as ps
from vmp_materials import import_medium
import v_save as vs
import v_utilities as vu

#%% PARAMETERS

# Saving directories
folder = "AuMieMediums/AllWaterTest/7)TestFluxTime"
home = vs.get_home()

# Sorting and labelling data series
sorting_function = lambda l:l
series_must = "" # leave "" per default
series_mustnt = "" # leave "" per default
series_column = 1

#%% LOAD DATA
# for f, sf, sm, smn in zip(folder, sorting_function, series_must, series_mustnt):

path = os.path.join(home, folder)
file = lambda f, s : os.path.join(path, f, s)

series = os.listdir(path)
series = vu.filter_to_only_directories(series)
series = vu.filter_by_string_must(series, series_must)
if series_mustnt!="": series = vu.filter_by_string_must(series, series_mustnt, False)
series = sorting_function(series)

data = []
params = []
for s in series:
    data.append(np.loadtxt(file(s, "Results.txt")))
    params.append(vs.retrieve_footer(file(s, "Results.txt")))
data_header = vs.retrieve_header(file(s, "Results.txt")) 

for i, p in enumerate(params):
    if not isinstance(p, dict): 
        params[i] = vu.fix_params_dict(p)

midflux = []
for s in series:
    midflux.append(np.loadtxt(file(s, "MidFlux.txt")))
midflux_header = vs.retrieve_header(file(s, "MidFlux.txt"))

endflux = []
for s in series:
    endflux.append(np.loadtxt(file(s, "BaseResults.txt")))
endflux_header = vs.retrieve_header(file(s, "BaseResults.txt"))

for i, p in enumerate(params):
    if not isinstance(p, dict): 
        params[i] = vu.fix_params_dict(p)

r = []
from_um_factor = []
resolution = []
index = []
second_time_factor = []
until_after_sources = []
time_factor_cell = []
for p in params:
    r.append( p["r"] )
    from_um_factor.append( p["from_um_factor"] )
    resolution.append( p["resolution"] )
    until_after_sources.append( p["until_after_sources"] )
    second_time_factor.append( p["second_time_factor"] )
    time_factor_cell.append( p["time_factor_cell"] )
    try:
        index.append( p["submerged_index"] )
    except KeyError:
        index.append( 1.33 )
        
# print(until_after_sources[0] == until_after_sources[1] / (2*1.33))

#%% PLOT MIDFLUX

freqs = [mf[:,0] for mf in midflux]

ylims = []
for mflux in midflux:
    yl = (np.min(mflux[:,1:]), np.max(mflux[:,1:]))
    # print(yl)
    yl = (yl[0] - .1*(yl[1]-yl[0]),
          yl[1]+.1*(yl[1]-yl[0]))
    # print(yl)
    ylims.append(yl)
ylims = (np.min(ylims), np.max(ylims))

linestyles = ["-c", ":k"]

fig, ax = plt.subplots(3, 2, sharex=True)
fig.subplots_adjust(hspace=0, wspace=.05)
for a in ax[:,1]:
    a.yaxis.tick_right()
    a.yaxis.set_label_position("right")
for a, h in zip(np.reshape(ax, 6), midflux_header[1:]):
    a.set_ylabel(h)

for mflux, frqs, fum, ls in zip(midflux, freqs, from_um_factor, linestyles):
    for mf, a in zip(mflux[:,1:].T, np.reshape(ax, 6)):
        a.plot(1e3*fum/frqs, mf, ls)
        a.set_ylim(*ylims)
ax[-1,0].set_xlabel(r"Wavelength $\lambda$ [nm]")
ax[-1,1].set_xlabel(r"Wavelength $\lambda$ [nm]")

a.legend(["Old Time", "New Time"])

# plt.savefig(file("MidFlux.png"))
        
#%% PLOT FINAL FLUX

freqs = [ef[:,0] for ef in endflux]

ylims = []
for eflux in endflux:
    yl = (np.min(eflux[:,2:8]), np.max(eflux[:,2:8]))
    # print(yl)
    yl = (yl[0] - .05*(yl[1]-yl[0]),
          yl[1]+.05*(yl[1]-yl[0]))
    # print(yl)
    ylims.append(yl)
ylims = (np.min(ylims), np.max(ylims))

linestyles = ["-c", ":k"]

fig, ax = plt.subplots(3, 2, sharex=True)
fig.subplots_adjust(hspace=0, wspace=.05)
for a in ax[:,1]:
    a.yaxis.tick_right()
    a.yaxis.set_label_position("right")
for a, h in zip(np.reshape(ax, 6), endflux_header[2:8]):
    a.set_ylabel(h)

for eflux, frqs, fum, ls in zip(endflux, freqs, from_um_factor, linestyles):
    for ef, a in zip(eflux[:,2:8].T, np.reshape(ax, 6)):
        a.plot(1e3*fum/frqs, ef, ls)
        a.set_ylim(*ylims)
ax[-1,0].set_xlabel(r"Wavelength $\lambda$ [nm]")
ax[-1,1].set_xlabel(r"Wavelength $\lambda$ [nm]")

a.legend(["Old Time", "New Time"])

# plt.savefig(file("EndFlux.png"))

#%% PLOT SCATTERING

wlens = [d[:,0] for d in data]

ylims = []
for d, fum, rd in zip(data, from_um_factor, r):
    yl = (np.min(d[:,1]), np.max(d[:,1]))
    # print(yl)
    yl = (yl[0] - .05*(yl[1]-yl[0]),
          yl[1]+.05*(yl[1]-yl[0]))
    # print(yl)
    ylims.append(np.array(yl) * np.pi * (fum * rd * 1e3)**2 )
ylims = (np.min(ylims), np.max(ylims))

linestyles = ["-c", ":k"]

plt.figure()
for wl, d, rd, fum, ls in zip(wlens, data, r, from_um_factor, linestyles):
    plt.plot(wl, d[:,1] * np.pi * (fum * rd * 1e3)**2, ls)
plt.xlabel(r"Wavelength $\lambda$ [nm]")
plt.ylabel(r"Scattering Cross Section [nm$^2$]")
plt.ylim(*ylims)
plt.legend(["Old Time", "New Time"])

# plt.savefig(file("Scattering.png"))