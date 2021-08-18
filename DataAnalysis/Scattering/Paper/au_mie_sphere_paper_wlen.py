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
import v_analysis as va
import v_materials as vmt
import v_save as vs
import v_utilities as vu

#%% PARAMETERS

# Saving directories
folder = ["AuMieSphere/AuMie/13)TestPaper/4)PaperJCFit/TestWLenJC/NewTime"]
home = vs.get_home()

# Sorting and labelling data series
sorting_function = [lambda l : vu.sort_by_number(l, -2)]
# def special_label(s):
#     if "5" in s:
#         return "Mie"
#     else:
#         return ""
series_label = [lambda s : f"Meep $\lambda$ Range Max {vu.find_numbers(s)[-2]} nm"]
series_must = [""] # leave "" per default
series_mustnt = [""] # leave "" per default
series_column = [1]

# Scattering plot options
plot_title = "Scattering for JC Au spheres in vacuum with 103 nm diameter"
series_colors = [plab.cm.Reds]
series_linestyles = ["solid"]
plot_make_big = False
plot_file = lambda n : os.path.join(home, "DataAnalysis/PaperWLenRange" + n)

#%% LOAD DATA

path = []
file = []
series = []
data = []
params = []
header = []

for f, sf, sm, smn in zip(folder, sorting_function, series_must, series_mustnt):

    path.append( os.path.join(home, f) )
    file.append( lambda f, s : os.path.join(path[-1], f, s) )
    
    series.append( os.listdir(path[-1]) )
    series[-1] = vu.filter_by_string_must(series[-1], sm)
    if smn!="": series[-1] = vu.filter_by_string_must(series[-1], smn, False)
    series[-1] = sf(series[-1])
    
    data.append( [] )
    params.append( [] )
    for s in series[-1]:
        data[-1].append(np.loadtxt(file[-1](s, "Results.txt")))
        params[-1].append(vs.retrieve_footer(file[-1](s, "Results.txt")))
    header.append( vs.retrieve_header(file[-1](s, "Results.txt")) )
    
    for i in range(len(params[-1])):
        if not isinstance(params[-1][i], dict): 
            params[-1][i] = vu.fix_params_dict(params[-1][i])
    
r = []
from_um_factor = []
for par in params:
    r.append([p["r"] for p in par])
    from_um_factor.append([p["from_um_factor"] for p in par])

#%% LOAD MIE DATA

from_um_factor = params[0][0]["from_um_factor"]
wlen_range = params[0][0]["wlen_range"]
r = params[0][0]["r"]
index = params[0][0]["submerged_index"]

medium = vmt.import_medium("Au", from_um_factor, paper="JC")

wlens = data[0][-1][:,0]
freqs = 1e3*from_um_factor/wlens
scatt_eff_theory = [ps.MieQ(np.sqrt(medium.epsilon(f)[0,0]*medium.mu(f)[0,0]), 
                            1e3*from_um_factor/f,
                            2*r*1e3*from_um_factor,
                            nMedium=index,
                            asDict=True)['Qsca'] 
                    for f in freqs]
scatt_eff_theory = np.array(scatt_eff_theory)

#%% GET MAX WAVELENGTH

max_wlen = []
for d, sc in zip(data, series_column):
    max_wlen.append( [d[i][np.argmax(d[i][:,sc]), 0] for i in range(len(d))] )
max_wlen_theory = wlens[np.argmax(scatt_eff_theory)]

dif_max_wlen = [ml - max_wlen_theory for ml in max_wlen[0]]

#%% WAVELENGTH MAXIMUM DIFFERENCE VS WAVELENGTH RANGE MAXIMUM

wlen_maximum = [vu.find_numbers(s)[-2] for s in series[0]]

plt.title("Difference in scattering maximum for Au 103 nm sphere in water")
plt.plot(wlen_maximum, dif_max_wlen, '.k', markersize=12)
plt.grid(True)
plt.legend(["Data", r"Fit $f(r)=a_0 e^{-a_1 r} + a_2$"])
plt.xlabel("Wavelength Range Maximum [nm]")
plt.ylabel("Difference in wavelength $\lambda_{max}^{MEEP}-\lambda_{max}^{MIE}$ [nm]")
vs.saveplot(plot_file("WLenDiff.png"), overwrite=True)

#%% AVERAGE RESIDUUM VS WAVELENGTH RANGE MAXIMUM

residuums = [[scatt_eff_theory - d[:,1] for d in dat] for dat in data]

plt.figure()
for dat, res, ser in zip(data, residuums, series):
    for d, rs, s in zip(dat, res, ser):
        plt.plot(d[:,0], rs, '.', label=f"Range Max {vu.find_numbers(s)[-2]:.0f} nm")
plt.legend()
plt.xlabel("Wavelength [nm]")
plt.ylabel("Scattering Effienciency Residuum")

accumulated_residuums = [sum(r) for r in res for res in residuums]
mean_residuums = [np.mean(r) for r in res for res in residuums]
# ¿Cómo comparo estas curvas si tienen amplitudes distintas y recorren rangos distintos de longitud de onda?

#%%

norm_residuums = [[scatt_eff_theory/max(scatt_eff_theory) - d[:,1]/max(d[:,1]) for d in dat] for dat in data]

plt.figure()
for dat, res, ser in zip(data, norm_residuums, series):
    for d, r, s in zip(dat, res, ser):
        plt.plot(d[:,0], r, '.', label=f"Range Max {vu.find_numbers(s)[-2]:.0f} nm")
plt.legend()
plt.xlabel("Wavelength [nm]")
plt.ylabel("Normalized Scattering Residuum")

#%% GET elapsed TIME

elapsed_time = [params[0][i]["elapsed"] for i in range(len(data[0]))]
total_elapsed_time = [sum(et) for et in elapsed_time]

rsq, m, b = va.linear_fit(np.array(wlen_maximum), 
                          np.array(total_elapsed_time), 
                          mb_units=["nm/s","s"])

# plt.figure()
plt.title("elapsed total time for simulation of Au 103 nm sphere in water")
# plt.plot(wlen_maximum, total_elapsed_time, '.k', markersize=12)
plt.legend(["Data", r"Fit $f(r)=a_0 \lambda + a_1$"], loc="lower right")
plt.xlabel("Wavelength Range Maximum [nm]")
plt.ylabel("elapsed time [s]")
vs.saveplot(plot_file("TotTime.png"), overwrite=True)
        
plt.figure()
plt.title("elapsed time for simulations of Au 103 nm sphere in water")
plt.plot(wlen_maximum, [et[1] for et in elapsed_time], 'D-b', label="Sim I")
plt.plot(wlen_maximum, [et[-1] for et in elapsed_time], 's-b', label="Sim II")
plt.xlabel("Wavelength Range Maximum [nm]")
plt.ylabel("elapsed time in simulations [s]")
plt.legend()
plt.savefig(plot_file("SimTime.png"), bbox_inches='tight')

plt.figure()
plt.title("elapsed time for building of Au 103 nm sphere in water")
plt.plot(wlen_maximum, [et[0] for et in elapsed_time], 'D-r', label="Sim I")
plt.plot(wlen_maximum, [et[2] for et in elapsed_time], 's-r', label="Sim II")
plt.xlabel("Wavelength Range Maximum [nm]")
plt.ylabel("elapsed time in building [s]")
plt.legend()
plt.savefig(plot_file("BuildTime.png"), bbox_inches='tight')

plt.figure()
plt.title("elapsed time for loading flux of Au 103 nm sphere in water")
plt.plot(wlen_maximum, [et[3] for et in elapsed_time], 's-m')
plt.xlabel("Wavelength Range Maximum [nm]")
plt.ylabel("elapsed time in loading flux [s]")
plt.savefig(plot_file("LoadTime.png"), bbox_inches='tight')

#%% PLOT NORMALIZED

colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

plt.figure()
plt.title(plot_title)
for s, d, p, sc, psl, pc, pls in zip(series, data, params, series_column, 
                                     series_label, colors, series_linestyles):

    for ss, sd, sp, spc in zip(s, d, p, pc):
        if ss!=series[0][0]:
            plt.plot(sd[:,0], sd[:,sc] / max(sd[:,sc]), 
                     linestyle=pls, color=spc, label=psl(ss))

plt.plot(wlens, scatt_eff_theory / max(scatt_eff_theory), 
         linestyle="dashed", color='red', label="Mie Theory")
plt.xlabel("Wavelength [nm]")
plt.ylabel("Normalized Scattering Cross Section")
plt.legend()
if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    del mng
vs.saveplot(plot_file("AllScattNorm.png"), overwrite=True)

#%% PLOT IN UNITS

colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

plt.figure()
plt.title(plot_title)
for s, d, p, sc, psl, pc, pls in zip(series, data, params, series_column, 
                                     series_label, colors, series_linestyles):

    for ss, sd, sp, spc in zip(s, d, p, pc):
        if ss!=series[0][0]:
            plt.plot(sd[:,0], sd[:,sc] * np.pi * (sp["r"] * sp["from_um_factor"] * 1e3)**2,
                     linestyle=pls, color=spc, label=psl(ss))
            
plt.plot(wlens, scatt_eff_theory  * np.pi * (sp["r"] * sp["from_um_factor"] * 1e3)**2,
         linestyle="dashed", color='red', label="Mie Theory")
plt.xlabel("Wavelength [nm]")
plt.ylabel(r"Scattering Cross Section [nm$^2$]")
plt.legend()
if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    del mng
vs.saveplot(plot_file("AllScatt.png"), overwrite=True)
