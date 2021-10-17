#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 11:32:22 2021

@author: vall
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

#%%

r = 51.5
wlen_range_min = 450
wlen_range_max = 600
from_um_factor = 10e-3
pml_wlen_factor = 0.38
air_r_factor = 0.5
time_factor_cell = 1.2
second_time_factor = 10

#%%

def resolution_int(wlen_range_min):
    
    resolution = 8 * from_um_factor * 1e3 / wlen_range_min # 8
    # resolution = max( resolution,  5 * from_um_factor / r )
    resolution = int( max([1, resolution]) )
    
    return resolution

def width_points_int(wlen_range_min, wlen_range_max):

    resolution = resolution_int(wlen_range_min)
    
    pml_width = pml_wlen_factor * wlen_range_max # 0.5 * max(wlen_range)
    air_width = air_r_factor * r # 0.5 * max(wlen_range)
    
    pml_width = pml_width / (1e3 * from_um_factor)
    air_width = pml_width / (1e3 * from_um_factor)
    
    pml_width = pml_width - pml_width%(1/resolution)
    air_width = air_width - air_width%(1/resolution)
    
    cell_width = 2 * (pml_width + air_width + r)
    cell_width = cell_width - cell_width%(1/resolution)
    
    return resolution * cell_width

def width_points_array(wlen_range_min, wlen_range_max):
    
    width_points = []
   
    for wmin in wlen_range_min:

        width_points.append([])
        
        for wmax in wlen_range_max:
            
            width_points[-1].append( width_points_int(wmin, wmax) )
            
    width_points = np.array(width_points).T
    
    return width_points

#%%

def points_int(wlen_range_min, wlen_range_max):
    
    return width_points_int(wlen_range_min, wlen_range_max)**3

def points_array(wlen_range_min, wlen_range_max):
    
    return np.power( width_points_array(wlen_range_min, wlen_range_max) , 3)

#%%

wlen_range_min = np.linspace(250, 500, 100)
wlen_range_max = np.linspace(600, 850, 100)

width_points_surface = width_points_array(wlen_range_min, wlen_range_max)

wlen_range_min, wlen_range_max = np.meshgrid(wlen_range_min, wlen_range_max)

#%%

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

surf = ax.plot_surface(wlen_range_min, 
                       wlen_range_max, 
                       width_points_surface, 
                       cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.set_xlabel(r"$\lambda_{min}$ [nm]")
ax.set_ylabel(r"$\lambda_{max}$ [nm]")
ax.set_zlabel("Points per grid side")
np.power(np.array([2,3]),2)
#%%

wlen_range_min = np.linspace(250, 500, 100)
wlen_range_max = np.linspace(600, 850, 100)

points_surface = points_array(wlen_range_min, wlen_range_max)

wlen_range_min, wlen_range_max = np.meshgrid(wlen_range_min, wlen_range_max)

#%%

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

surf = ax.plot_surface(wlen_range_min, 
                       wlen_range_max, 
                       points_surface, 
                       cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.set_xlabel(r"$\lambda_{min}$ [nm]")
ax.set_ylabel(r"$\lambda_{max}$ [nm]")
ax.set_zlabel("Points in grid")
