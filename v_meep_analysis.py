#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 12:02:05 2021

@author: vall
"""

import numpy as np
from scipy.signal import find_peaks

#%% GENERAL: INDEX FUNCTION

def def_index_function(array):
    
    def index_function(value):
        return np.argmin(np.abs(np.asarray(array) - value))
    
    return index_function    

#%% FIELD ANALYSIS: CROP FIELD FUNCTIONS

def crop_field_zyplane(plane_field, y_plane_index, z_plane_index,
                       cell_width, pml_width):
    
    cropped = plane_field[: y_plane_index(cell_width/2 - pml_width), :]
    cropped = cropped[y_plane_index(-cell_width/2 + pml_width) :, :]
    cropped = cropped[:,: z_plane_index(cell_width/2 - pml_width)]
    cropped = cropped[:, z_plane_index(-cell_width/2 + pml_width) :]
    
    return cropped 
    
def crop_field_zprofile(zprofile_field, z_plane_index,
                        cell_width, pml_width):
    
    cropped = zprofile_field[: z_plane_index(cell_width/2 - pml_width)]
    cropped = cropped[z_plane_index(-cell_width/2 + pml_width) :]
    
    return cropped

#%% FIELD ANALYSIS: SOURCE FROM X LINE FIELD

def get_source_from_line(xline_field, x_line_index, 
                         source_center):
    
    return np.asarray(xline_field[x_line_index(source_center), :])

def get_peaks_from_source(source_field, 
                          peaks_sensitivity=0.1,
                          last_stable_periods=5):   
    
    peaks = find_peaks(source_field, height=0)[0]
    
    mean_diff = np.mean(np.diff(peaks)[-last_stable_periods:])
    def selection_criteria(k):
        eval_point = np.abs( mean_diff - (peaks[k+1] - peaks[k]) )
        return eval_point <= peaks_sensitivity * mean_diff
    
    selected_peaks = []
    for k in range(len(peaks)):
        if k == 0 and selection_criteria(k):
            selected_peaks.append(peaks[k])
            # print(k, " entered first if")
        elif k == len(peaks)-1 and selection_criteria(k-1):
            selected_peaks.append(peaks[k])
            # print(k, " entered first elif")
        elif selection_criteria(k):
            selected_peaks.append(peaks[k])
            # print(k, " entered second elif")

    return selected_peaks

def get_period_from_source(source_field, t_line,
                           peaks_sensitivity=0.1,
                           periods_sensitivity=0.05,
                           last_stable_periods=5):
    
    peaks = get_peaks_from_source(source_field, 
                                  peaks_sensitivity=peaks_sensitivity,
                                  last_stable_periods=last_stable_periods)
    periods = np.array(t_line[peaks[1:]] - t_line[peaks[:-1]])
    
    mean_periods = np.mean(periods[-last_stable_periods:])
    def selection_criteria(k):
        eval_point = np.abs(periods[k] - mean_periods)
        return eval_point <= periods_sensitivity * mean_periods
    
    keep_periods_from = 0
    for k in range(len(periods)):
        if not selection_criteria(k):
            keep_periods_from = max(keep_periods_from, k+1)
    stable_periods = periods[keep_periods_from:]
    
    return np.mean(stable_periods)

def get_amplitude_from_source(source_field,
                              peaks_sensitivity=0.1,
                              amplitude_sensitivity=0.05,
                              last_stable_periods=5):
    
    peaks = get_peaks_from_source(source_field, 
                                  peaks_sensitivity=peaks_sensitivity,
                                  last_stable_periods=last_stable_periods)
    heights = source_field[peaks]
    
    mean_height = np.mean(source_field[peaks[-last_stable_periods:]])
    def selection_criteria(k):
        eval_point = np.abs(heights[k] - mean_height)
        return eval_point <= amplitude_sensitivity * mean_height
    
    keep_periods_from = 0
    for k in range(len(heights)):
        if not selection_criteria(k):
            keep_periods_from = max(keep_periods_from, k+1)
    stable_heights = source_field[peaks[keep_periods_from:]]
    
    return np.mean(stable_heights)

#%% FIELD ANALYSIS: ZPROFILE FROM YZ PLANE

def get_zprofile_from_plane(yzplane_field, y_plane_index):
    
    return np.asarray(yzplane_field[y_plane_index(0), :, :])

def integrate_field_zprofile(zprofile_field, z_plane_index,
                             cell_width, pml_width, period_plane):
    
    integral = np.sum(crop_field_zprofile(zprofile_field, z_plane_index,
                                          cell_width, pml_width), axis=0) * period_plane
    
    return integral

def detect_sign_field_zprofile(zprofile_field):
    
    return -np.sign(zprofile_field[0,...])

def find_peaks_field_zprofile(zprofile_field, z_plane_index, 
                              r, cell_width, pml_width):
    
    zprofile_signs = detect_sign_field_zprofile(zprofile_field)
    
    zprofile_peaks = []
    zprofile_maxs = []
    for zprof, sign in zip(np.rollaxis(zprofile_field, -1), zprofile_signs):
            
        two_peaks = find_peaks(sign * crop_field_zprofile(zprof, z_plane_index,
                                                          cell_width, pml_width))[0]
        two_peaks = np.array(two_peaks) + z_plane_index(-cell_width/2 + pml_width)
        if len(two_peaks) > 2: two_peaks = two_peaks[[0,-1]]
        try:
            two_peaks[0] = min(two_peaks[0], z_plane_index(-r) - 1)
            two_peaks[1] = max(two_peaks[1], z_plane_index(r))
            chosen_peak = min(two_peaks[0], z_plane_index(-r))
            zprofile_peaks.append( chosen_peak )
            zprofile_maxs.append( zprof[chosen_peak] )
        except IndexError or RuntimeWarning:
            zprofile_peaks.append( -1 )
            zprofile_maxs.append( 0 )
            
    zprofile_peaks = np.array(zprofile_peaks)
    zprofile_maxs = np.array(zprofile_maxs)
    
    return zprofile_peaks, zprofile_maxs

#%% FIELD ANALYSIS: RESONANCE FROM ZPROFILE

def get_all_resonance_from_yzplanes(yzplane_field,
                                    y_plane_index, z_plane_index, 
                                    r, cell_width, pml_width):
    # zprofile_field has time as the last dimension
    
    zprofile_field = get_zprofile_from_plane(yzplane_field, y_plane_index)
    
    zprofile_peaks, zprofile_maxs  = find_peaks_field_zprofile(
        zprofile_field, z_plane_index, r, cell_width, pml_width)
    # z_resonance_max
    
    first_index = int(np.argwhere( zprofile_peaks >= 0 )[0])
    resonance_index = find_peaks(zprofile_maxs[first_index:], 
                                 height=(np.max(zprofile_maxs[first_index:])/2, None))[0]
    resonance_index = np.array(resonance_index) + first_index
    
    resonance_zprofile = [crop_field_zprofile(zprofile_field[...,k], z_plane_index, 
                                              cell_width, pml_width) 
                              for k in resonance_index]
    resonance_zprofile = np.array(resonance_zprofile).T
    
    resonance_yzplane = [crop_field_zyplane(yzplane_field[...,k], 
                                            y_plane_index, z_plane_index, 
                                            cell_width, pml_width) 
                              for k in resonance_index]
    resonance_yzplane = np.array(resonance_yzplane).T
    
    return resonance_index, resonance_zprofile, resonance_yzplane

def get_single_resonance_from_yzplanes(yzplane_field,
                                       y_plane_index, z_plane_index, 
                                       r, cell_width, pml_width,
                                       index_sensitivity=0.05,
                                       resonance_sensitivity=0.02,
                                       last_stable_periods=5):
    # zprofile_field has time as the last dimension
    
    all_index, all_zprofile, all_yzplane = get_all_resonance_from_yzplanes(
        yzplane_field, y_plane_index, z_plane_index, r, cell_width, pml_width)
        
    mean_diff = np.mean(np.diff(all_index)[-last_stable_periods:])
    def selection_criteria(k):
        eval_point = np.abs( mean_diff - (all_index[k+1] - all_index[k]) )
        return eval_point <= index_sensitivity * mean_diff
    
    selected_index = []
    for k in range(len(all_index)):
        if k == 0 and selection_criteria(k):
            selected_index.append(k)
            # print(k, " entered first if")
        elif k == len(all_index)-1 and selection_criteria(k-1):
            selected_index.append(k)
            # print(k, " entered first elif")
        elif selection_criteria(k):
            selected_index.append(k)
            # print(k, " entered second elif")
    
    max_values = [np.max(all_zprofile[...,k]) for k in selected_index]
   
    mean_value = np.mean(max_values[-last_stable_periods:])
    def selection_criteria(k):
        eval_point = np.abs(max_values[k] - mean_value)
        return eval_point <= resonance_sensitivity * mean_value
    
    keep_periods_from = 0
    for k in range(len(max_values)):
        if not selection_criteria(k):
            keep_periods_from = max(keep_periods_from, k+1)
    stable_index = selected_index[keep_periods_from:]
    stable_values = max_values[keep_periods_from:]
    
    max_index = np.argmax(stable_values)
    # min_index = np.argmin(max_values)

    resonance_index = all_index[stable_index[max_index]]
    resonance_zprofile = all_zprofile[..., stable_index[max_index]]
    resonance_yzplane = all_yzplane[..., stable_index[max_index]]
    
    return resonance_index, resonance_zprofile, resonance_yzplane
