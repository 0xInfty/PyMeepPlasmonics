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
from v_materials import import_medium
import v_save as vs
import v_utilities as vu

#%% PARAMETERS

# Saving directories
folder = ["AuMieSphere/AuMie/10)MaxRes/Max103FUMixRes",
          "AuMieSphere/AuMie/13)TestPaper/4)PaperJCFit/PaperJCFit103MaxRes/AllVac450600"]
# folder = ["AuMieMediums/AllWaterMaxRes/AllWaterMax103Res",
#           "AuMieSphere/AuMie/13)TestPaper/4)PaperJCFit/PaperJCFit103MaxRes/AllWat500650"]
home = vs.get_home()

# Sorting and labelling data series
sorting_function = [lambda l : vu.sort_by_number(l, -1),
                    lambda l : vu.sort_by_number(l, -1)]
# def special_label(s):
#     if "5" in s:
#         return "Mie"
#     else:
#         return ""
series_label = [lambda s : f"Meep R Resolution {vu.find_numbers(s)[-1]}",
                lambda s : f"Meep JC Resolution {vu.find_numbers(s)[-1]}"]
series_must = ["", ""] # leave "" per default
series_mustnt = ["Failed", "Failed"] # leave "" per default
series_column = [1, 1]
series_colors = [plab.cm.Reds, plab.cm.Blues]
series_linestyles = ["solid", "solid"]

plot_title = "JC/R Au 103 nm sphere in vacuum"
# plot_title = "JC/R Au 103 nm sphere in water"
theory_label = [lambda s : f"R Theory {vu.find_numbers(s)[0]} nm",
                lambda s : f"JC Theory {vu.find_numbers(s)[0]} nm"]
theory_linestyle = ["dashed", "dashed"]

# Scattering plot options
plot_make_big = True
plot_file = lambda n : os.path.join(home, "DataAnalysis/PaperMaxRes" + n)

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
    
base_from_um_factor = 20e-3    

r = []
from_um_factor = []
resolution = []
joint_from_um_factor = []
joint_resolution = []
paper = []
index = []
for p in params:
    r.append( [pi["r"] for pi in p] )
    resolution.append( [pi["resolution"] for pi in p] )
    from_um_factor.append( [pi["from_um_factor"] for pi in p] )
    joint_from_um_factor.append([])
    joint_resolution.append([])
    for pi in p:
        if pi["from_um_factor"] == base_from_um_factor:            
            joint_from_um_factor[-1].append( pi["from_um_factor"] )
            joint_resolution[-1].append( pi["resolution"] )
        else:
            factor = base_from_um_factor / pi["from_um_factor"]
            joint_from_um_factor[-1].append( pi["from_um_factor"]*factor )
            joint_resolution[-1].append( int(pi["resolution"]*factor) )
    try:
        paper.append( [pi["paper"] for pi in p])
    except:
        paper.append( ["R" for pi in p] )
    index.append( [] )
    for pi in p:
        try:
            index[-1].append( pi["submerged_index"] )
        except KeyError:
            index[-1].append( 1 )

#%% GET THEORY

theory = [] # Scattering effiency
for di, ri, fi, resi, ppi, ii in zip(data, r, from_um_factor, resolution, paper, index):
    theory.append([])    
    for dj, rj, fj, resj, ppij, ij in zip(di, ri, fi, resi, ppi, ii):
        wlenj = dj[:,0] # nm
        freqj = 1 / wlenj # 1/nm
        freqmeepj = (1e3 * fj) / wlenj # Meep units
        mediumj = import_medium("Au", from_um_factor=fj, paper=ppij)
        theory[-1].append(np.array(
            [ps.MieQ(np.sqrt(mediumj.epsilon(fqm)[0,0]*mediumj.mu(fqm)[0,0]), 
                     wl, # Wavelength (nm)
                     2*rj*1e3*fj, # Diameter (nm)
                     nMedium=ij, # Refraction Index of Mediums
                     asDict=True)['Qsca'] 
             for wl, fq, fqm in zip(wlenj, freqj, freqmeepj)]))

#%% GET MAX WAVELENGTH

max_wlen = []
for d, sc in zip(data, series_column):
    max_wlen.append( [d[i][np.argmax(d[i][:,sc]), 0] for i in range(len(d))] )

max_wlen_theory = []
for t, d in zip(theory, data):
    max_wlen_theory.append( [d[i][np.argmax(t[i]), 0] for i in range(len(t))] )
    
max_wlen_diff = []
for md, mt in zip(max_wlen, max_wlen_theory):
    max_wlen_diff.append( [d - t for d,t in zip(md, mt)] )

#%% WAVELENGTH MAXIMUM DIFFERENCE VS RESOLUTION COMPARED

plt.title("Difference in wavelength maximum for " + plot_title)
for res, dif in zip(joint_resolution, max_wlen_diff):
    plt.plot(res, dif, '.', markersize=12)
plt.grid(True)
plt.legend(["R", "JC"])
plt.xlabel("Resolution")
plt.ylabel("Difference in wavelength $\lambda_{max}^{MEEP}-\lambda_{max}^{MIE}$")
vs.saveplot(plot_file("ComparedWLenDiff.png"), overwrite=True)

#%% WAVELENGTH MAXIMUM DIFFERENCE VS RESOLUTION JUST JC

def exponential_fit(X, A, b, C):
    return A * np.exp(-b*X) + C

rsq, parameters = va.nonlinear_fit(np.array(resolution[1]), 
                                   np.array(max_wlen_diff[1]), 
                                   exponential_fit,
                                   par_units=["nm", "", "nm"])

plt.title("Difference in scattering maximum")
# plt.plot(resolution, dif_max_wlen, '.k', markersize=12)
plt.grid(True)
plt.legend(["JC Meep Data", r"Fit $f(r)=a_0 e^{-a_1 r} + a_2$"])
plt.xlabel("Resolution")
plt.ylabel("Difference in wavelength $\lambda_{max}^{MEEP}-\lambda_{max}^{MIE}$")
vs.saveplot(plot_file("WLenDiff.png"), overwrite=True)

#%% GET ELAPSED TIME COMPARED

elapsed_time = [[p["elapsed"] for p in par] for par in params]
total_elapsed_time = [[sum(p["elapsed"]) for p in par] for par in params]

first_resolution = []
second_resolution = []
first_build_time = []
first_sim_time = []
second_flux_time = []
second_build_time = []
second_sim_time = []
for enl, res in zip(elapsed_time, joint_resolution):
    first_resolution.append( [] )
    first_build_time.append( [] )
    first_sim_time.append( [] )
    second_resolution.append( [] )
    second_flux_time.append( [] )
    second_build_time.append( [] )
    second_sim_time.append( [] )
    for e, rs in zip(enl, res):
        if len(e)==5:
            first_resolution[-1].append( rs )
            second_resolution[-1].append( rs )
            first_build_time[-1].append( e[0] )
            first_sim_time[-1].append( e[1] )
            second_build_time[-1].append( e[2] )
            second_flux_time[-1].append( e[3] )
            second_sim_time[-1].append( e[4] )
        elif len(e)==3:
            second_resolution[-1].append( rs )
            second_build_time[-1].append( e[0] )
            second_flux_time[-1].append( e[1] )
            second_sim_time[-1].append( e[2] )
        else:
            print(f"Unknown error in resolution {rs} of", res)

plt.figure()
plt.title("elapsed total time for simulation of " + plot_title)
for res, tot in zip(joint_resolution, total_elapsed_time):
    plt.plot(res, tot, '.', markersize=12)
plt.legend(["Meep R", "Meep JC"], loc="lower right")
plt.xlabel("Resolution")
plt.ylabel("elapsed time [s]")
vs.saveplot(plot_file("ComparedTotTime.png"), overwrite=True)
        
plt.figure()
plt.title("elapsed time for simulations of " + plot_title)
plt.plot(first_resolution[0], first_sim_time[0], 'D-', color="C0", label="R Sim I")
plt.plot(first_resolution[1], first_sim_time[1], 'D-', color="C1", label="JC Sim I")
plt.plot(second_resolution[0], second_sim_time[0], 's-', color="C0", label="R Sim II")
plt.plot(second_resolution[1], second_sim_time[1], 's-', color="C1", label="JC Sim II")
plt.xlabel("Resolution")
plt.ylabel("elapsed time in simulations [s]")
plt.legend()
plt.savefig(plot_file("ComparedSimTime.png"), bbox_inches='tight')

plt.figure()
plt.title("elapsed time for building of " + plot_title)
plt.plot(first_resolution[0], first_build_time[0], 'D-', color="C0", label="R Sim I")
plt.plot(first_resolution[1], first_build_time[1], 'D-', color="C1", label="JC Sim I")
plt.plot(second_resolution[0], second_build_time[0], 's-', color="C0", label="R Sim II")
plt.plot(second_resolution[1], second_build_time[1], 's-', color="C1", label="JC Sim II")
plt.xlabel("Resolution")
plt.ylabel("elapsed time in building [s]")
plt.legend()
plt.savefig(plot_file("ComparedBuildTime.png"), bbox_inches='tight')

plt.figure()
plt.title("elapsed time for loading flux of " + plot_title)
plt.plot(second_resolution[0], second_flux_time[0], 's-', color="C0", label="R Sim II")
plt.plot(second_resolution[1], second_flux_time[1], 's-', color="C1", label="JC Sim II")
plt.xlabel("Resolution")
plt.ylabel("elapsed time in loading flux [s]")
plt.savefig(plot_file("ComparedLoadTime.png"), bbox_inches='tight')

#%% GET ELAPSED TIME JC

def quartic_fit(X, A, b):
    return A * (X)**4 + b
rsq, parameters = va.nonlinear_fit(np.array(resolution[1][:-1]), 
                                   np.array(total_elapsed_time[1][:-1]), 
                                   quartic_fit,
                                   par_units=["s","s"])

plt.title("elapsed total time for simulation of " + plot_title)
# plt.plot(resolution, total_elapsed_time)
plt.legend(["Data", r"Fit $f(r)=a_0 r^4 + a_1$"], loc="lower right")
plt.xlabel("Resolution")
plt.ylabel("elapsed time [s]")
vs.saveplot(plot_file("TotTime.png"), overwrite=True)
        
plt.figure()
plt.title("elapsed time for simulations of " + plot_title)
plt.plot(first_resolution[1], first_sim_time[1], 'D-b', label="Sim I")
plt.plot(second_resolution[1], second_sim_time[1], 's-b', label="Sim II")
plt.xlabel("Resolution")
plt.ylabel("elapsed time in simulations [s]")
plt.legend()
plt.savefig(plot_file("SimTime.png"), bbox_inches='tight')

plt.figure()
plt.title("elapsed time for building of " + plot_title)
plt.plot(first_resolution[1], first_build_time[1], 'D-r', label="Sim I")
plt.plot(second_resolution[1], second_build_time[1], 's-r', label="Sim II")
plt.xlabel("Resolution")
plt.ylabel("elapsed time in building [s]")
plt.legend()
plt.savefig(plot_file("BuildTime.png"), bbox_inches='tight')

plt.figure()
plt.title("elapsed time for loading flux of " + plot_title)
plt.plot(second_resolution[1], second_flux_time[1], 's-m')
plt.xlabel("Resolution")
plt.ylabel("elapsed time in loading flux [s]")
plt.savefig(plot_file("LoadTime.png"), bbox_inches='tight')

#%% PLOT NORMALIZED

jc_wlens = data[1][0][:,0]
jc_theory = theory[1][0]
r_wlens = data[0][0][:,0]
r_theory = theory[0][0]

colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

plt.figure()
plt.title("Normalized scattering for " + plot_title)
for s, d, p, sc, psl, pc, pls in zip(series, data, params, series_column, 
                                      series_label, colors, series_linestyles):

    for ss, sd, sp, spc in zip(s, d, p, pc):
        if ss not in series[0]:
            plt.plot(sd[:,0], sd[:,sc] / max(sd[:,sc]), 
                      linestyle=pls, color=spc, label=psl(ss))

plt.plot(jc_wlens, jc_theory / max(jc_theory), 
          linestyle="dashed", color='red', label="JC Mie Theory")
plt.plot(r_wlens, r_theory / max(r_theory), 
          linestyle="dashed", color='k', label="R Mie Theory")

plt.xlabel("Wavelength [nm]")
plt.ylabel("Normalized Scattering Cross Section")
plt.legend()
if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
del mng
vs.saveplot(plot_file("AllScattNorm.png"), overwrite=True)

#%% PLOT EFFIENCIENCY

colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

plt.figure()
plt.title("Scattering effiency for " + plot_title)
for s, d, p, sc, psl, pc, pls in zip(series, data, params, series_column, 
                                      series_label, colors, series_linestyles):

    for ss, sd, sp, spc in zip(s, d, p, pc):
        if ss not in series[0]:
            plt.plot(sd[:,0], sd[:,sc], 
                      linestyle=pls, color=spc, label=psl(ss))

plt.plot(jc_wlens, jc_theory, 
          linestyle="dashed", color='red', label="JC Mie Theory")
plt.plot(r_wlens, r_theory, 
          linestyle="dashed", color='k', label="R Mie Theory")

plt.xlabel("Wavelength [nm]")
plt.ylabel("Scattering Effiency")
plt.legend()
if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
del mng
vs.saveplot(plot_file("AllScattEff.png"), overwrite=True)

#%% PLOT IN UNITS

colors = [sc(np.linspace(0,1,len(s)+3))[3:] 
          for sc, s in zip(series_colors, series)]

plt.figure()
plt.title("Scattering for " + plot_title)
for s, d, p, sc, psl, pc, pls in zip(series, data, params, series_column, 
                                      series_label, colors, series_linestyles):

    for ss, sd, sp, spc in zip(s, d, p, pc):
        if ss not in series[0]:
            plt.plot(sd[:,0], sd[:,sc] * np.pi * (sp["r"] * sp["from_um_factor"] * 1e3)**2, 
                      linestyle=pls, color=spc, label=psl(ss))

plt.plot(jc_wlens, jc_theory  * np.pi * (r[0][0] * from_um_factor[0][0] * 1e3)**2, 
          linestyle="dashed", color='red', label="JC Mie Theory")
plt.plot(r_wlens, r_theory  * np.pi * (r[1][0] * from_um_factor[1][0] * 1e3)**2, 
          linestyle="dashed", color='k', label="R Mie Theory")

plt.xlabel("Wavelength [nm]")
plt.ylabel(r"Scattering Cross Section [nm$^2$]")
plt.legend()
if plot_make_big:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
del mng
vs.saveplot(plot_file("AllScatt.png"), overwrite=True)
