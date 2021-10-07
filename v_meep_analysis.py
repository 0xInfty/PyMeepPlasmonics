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
    """
    Generates a function to get the index of the closest value inside an array.

    Parameters
    ----------
    array : np.array
        Input data.

    Returns
    -------
    index_function : function
        A function of only one argument `value` that searches for a value 
        inside the input array and returns the closest index.
    """
    
    if not isinstance(array, np.ndarray):
        raise Warning("The input data should be inside a np.ndarray instance")
    
    def index_function(value):
        return np.argmin(np.abs(np.asarray(array) - value))
    
    return index_function    

#%% FIELD ANALYSIS: CROP FIELD FUNCTIONS

def crop_single_field_yzplane(single_yzplane, y_plane_index, z_plane_index,
                              cell_width, pml_width):
    """Croppes a single YZ plane field array, dropping the outsides of the cell.
    
    Parameters
    ----------
    single_yzplane : np.array with dimension 2
        Bidimensional field array of shape (N,M) where N stands for positions 
        in the Y axis and M stands for positions in the Z axis.
    y_plane_index : function
        Function of a single argument that takes in an Y position and returns 
        the index of the closest value.
    z_plane_index : function
        Function of a single argument that takes in an Z position and returns 
        the index of the closest value.
    cell_width : int, float
        Cubic cell side's total length, generally expressed in Meep units to 
        be in the same units as the `y_plane` metadata array.
    pml_width : int, float
        Cell's isotropic PML's width, generally expressed in Meep units to be 
        in the same units as the `y_plane` metadata array.
    
    Returns
    -------
    cropped : np.array
        Bidimensional field cropped array of shape (n,m) where n<N stands for 
        positions in the Y axis and m<M stands for positions in the Z axis.
    """
    
    cropped = single_yzplane[: y_plane_index(cell_width/2 - pml_width) + 1, :]
    cropped = cropped[y_plane_index(-cell_width/2 + pml_width) :, :]
    cropped = cropped[:,: z_plane_index(cell_width/2 - pml_width) + 1]
    cropped = cropped[:, z_plane_index(-cell_width/2 + pml_width) :]
    
    return cropped 
    
def crop_single_field_zprofile(single_zprofile, z_plane_index,
                               cell_width, pml_width):
    """Croppes a single Z line field array, dropping the outsides of the cell.
    
    Parameters
    ----------
    single_zprofile : np.array with dimension 1
        One-dimensional field array of shape (N,) where n stands for positions 
        in the Z axis.
    z_plane_index : function
        Function of a single argument that takes in an Z position and returns 
        the index of the closest value.
    cell_width : int, float
        Cubic cell side's total length, generally expressed in Meep units to 
        be in the same units as the `y_plane` metadata array.
    pml_width : int, float
        Cell's isotropic PML's width, generally expressed in Meep units to be 
        in the same units as the `y_plane` metadata array.
    
    Returns
    -------
    cropped : np.array
        One-dimensional field cropped array of shape (n,) where n<N stands for 
        positions in the Z axis.
    """
    
    cropped = single_zprofile[: z_plane_index(cell_width/2 - pml_width) + 1]
    cropped = cropped[z_plane_index(-cell_width/2 + pml_width) :]
    
    return cropped

def crop_single_field_xprofile(single_xprofile, x_line_index,
                               cell_width, pml_width):
    """Croppes a single Z line field array, dropping the outsides of the cell.
    
    Parameters
    ----------
    single_xprofile : np.array with dimension 1
        One-dimensional field array of shape (N,) where n stands for positions 
        in the X axis.
    x_line_index : function
        Function of a single argument that takes in an X position and returns 
        the index of the closest value.
    cell_width : int, float
        Cubic cell side's total length, generally expressed in Meep units to 
        be in the same units as the `y_plane` metadata array.
    pml_width : int, float
        Cell's isotropic PML's width, generally expressed in Meep units to be 
        in the same units as the `y_plane` metadata array.
    
    Returns
    -------
    cropped : np.array
        One-dimensional field cropped array of shape (n,) where n<N stands for 
        positions in the X axis.
    """
    
    return crop_single_field_zprofile(single_xprofile, x_line_index, 
                                      cell_width, pml_width)

def crop_field_yzplane(yzplane_field, y_plane_index, z_plane_index,
                       cell_width, pml_width):
    """Croppes full YZ planes field array, dropping the outsides of the cell.
    
    Parameters
    ----------
    yzplane_field : np.array with dimension 3
        Three-dimensional field array of shape (N,M,K) where N stands for 
        positions in the Y axis, M stands for positions in the Z axis and K 
        stands for different time instants.
    y_plane_index : function
        Function of a single argument that takes in an Y position and returns 
        the index of the closest value.
    z_plane_index : function
        Function of a single argument that takes in an Z position and returns 
        the index of the closest value.
    cell_width : int, float
        Cubic cell side's total length, generally expressed in Meep units to 
        be in the same units as the `y_plane` metadata array.
    pml_width : int, float
        Cell's isotropic PML's width, generally expressed in Meep units to be 
        in the same units as the `y_plane` metadata array.
    
    Returns
    -------
    cropped : np.array
        Three-dimensional field array of shape (n,m,K) where n<N stands for 
        positions in the Y axis, m<M stands for positions in the Z axis and K 
        stands for different instants of time.
    """
    
    cropped = yzplane_field[: y_plane_index(cell_width/2 - pml_width) + 1, ...]
    cropped = cropped[y_plane_index(-cell_width/2 + pml_width) :, ...]
    cropped = cropped[:,: z_plane_index(cell_width/2 - pml_width) + 1, ...]
    cropped = cropped[:, z_plane_index(-cell_width/2 + pml_width) :, ...]
    
    return cropped 
    
def crop_field_zprofile(zprofile_field, z_plane_index,
                        cell_width, pml_width):
    """Croppes full Z lines field array, dropping the outsides of the cell.
    
    Parameters
    ----------
    zprofile_field : np.array with dimension 2
        Bidimensional field array of shape (N,K) where N stands for positions 
        in the Z axis and K stands for different time instants.
    z_plane_index : function
        Function of a single argument that takes in an Z position and returns 
        the index of the closest value.
    cell_width : int, float
        Cubic cell side's total length, generally expressed in Meep units to 
        be in the same units as the `y_plane` metadata array.
    pml_width : int, float
        Cell's isotropic PML's width, generally expressed in Meep units to be 
        in the same units as the `y_plane` metadata array.
    
    Returns
    -------
    cropped : np.array
        Bidimensional field cropped array of shape (n,K) where n<N stands for 
        positions in the Z axis and K stands for different time instants.
    """
    
    cropped = zprofile_field[: z_plane_index(cell_width/2 - pml_width) + 1, :]
    cropped = cropped[z_plane_index(-cell_width/2 + pml_width) :, :]
    
    return cropped

def crop_field_xprofile(xprofile_field, x_line_index,
                        cell_width, pml_width):
    """Croppes full X lines field array, dropping the outsides of the cell.
    
    Parameters
    ----------
    xprofile_field : np.array with dimension 2
        Bidimensional field array of shape (N,K) where N stands for positions 
        in the X axis and K stands for different time instants.
    x_plane_index : function
        Function of a single argument that takes in an X position and returns 
        the index of the closest value.
    cell_width : int, float
        Cubic cell side's total length, generally expressed in Meep units to 
        be in the same units as the `y_plane` metadata array.
    pml_width : int, float
        Cell's isotropic PML's width, generally expressed in Meep units to be 
        in the same units as the `y_plane` metadata array.
    
    Returns
    -------
    cropped : np.array
        Bidimensional field cropped array of shape (n,K) where n<N stands for 
        positions in the X axis and K stands for different time instants.
    """
    
    return crop_field_zprofile(xprofile_field, x_line_index,
                               cell_width, pml_width)

#%% FIELD ANALYSIS: SOURCE FROM X LINE FIELD

def get_source_from_line(xline_field, x_line_index, 
                         source_center):
    """Extracts data corresponding to source position from X lines field array.

    Parameters
    ----------
    xline_field : np.array with dimension 2
        Bidimensional field array of shape (L,K) where L stands for positions 
        in the X axis and K stands for different time instants.
    x_line_index : function
        Function of a single argument that takes in an X position and returns 
        the index of the closest value.
    source_center : int, float
        YZ wave front planewidth source's X position, generally expressed in 
        Meep units to be in the same units as the `y_plane` metadata array.

    Returns
    -------
    np.array
        One-dimensional field array of shape (K) where K stands for different 
        time instants.
    """
    
    return np.asarray(xline_field[x_line_index(source_center), :])

def get_peaks_from_source(source_field, 
                          peaks_sep_sensitivity=0.1,
                          last_stable_periods=5):
    """Finds maximum peaks of a periodic oscillating signal as the source field.

    Parameters
    ----------
    source_field : np.array of dimension 1
        One-dimensional array of a periodic oscillating signal. Generally, 
        field array of shape (K) where K stands for different time instants.
    peaks_sep_sensitivity=0.1 : float between zero and one, optional
        A factor representing the allowed variation percentage in consecutive 
        peaks separation. Any deviated value will be dropped.
    last_stable_periods=5 : int, optional
        Number of periods to take as reference of the stable signal, extracted 
        from the end as the signal is assumed to have both a transcient and a 
        stationary regimen.

    Returns
    -------
    selected_peaks : list
        List of int index for the maximum peaks found inside the input array.
    """
    
    # peaks = find_peaks(source_field, height=0)[0]
    peaks = find_peaks(np.abs(source_field), 
                       height=np.max(np.abs(source_field))/2)[0]
    # All peaks, no only maximums, but minimums too
    
    # Take the last periods as reference and define periodicity criteria
    mean_diff = np.mean(np.diff(peaks[-2*last_stable_periods:]))
    def selection_criteria(k):
        eval_point = np.abs( mean_diff - (peaks[k+1] - peaks[k]) )
        return eval_point <= peaks_sep_sensitivity * mean_diff
    
    # Filter peaks to make sure they are periodic; i.e. they have equispaced index
    selected_peaks = []
    for k in range(len(peaks)):
        if k == 0 and selection_criteria(k):
            selected_peaks.append(peaks[k])
        elif k == len(peaks)-1 and selection_criteria(k-1):
            selected_peaks.append(peaks[k])
        elif selection_criteria(k):
            selected_peaks.append(peaks[k])

    # Check if there's always a single maximum and a single minimum per period
    # If not, try to fix it or raise Warning.
    selected_sign = np.sign(source_field[selected_peaks])
    if np.min(np.abs(np.diff( selected_sign ))) != 2:
        error_index = np.argmin(np.abs(np.diff( selected_sign )))
        missing_index_lower_bound = selected_peaks[ error_index ]
        missing_index_upper_bound = selected_peaks[ error_index + 1 ]
        if missing_index_upper_bound - missing_index_lower_bound == 2:
            selected_peaks = [*selected_peaks[: error_index + 1],
                              missing_index_lower_bound + 1,
                              *selected_peaks[error_index + 1 :]]
            selected_sign = np.sign(source_field[selected_peaks])
            if np.min(np.abs(np.diff( selected_sign ))) != 2:
                print("Warning! Sign algorithm failed and it couldn't be fixed!")
        else:
            print("Warning! Sign algorithm must have failed!")

    return selected_peaks

def get_period_from_source(source_field, t_line=None,
                           peaks_sep_sensitivity=0.1,
                           periods_sensitivity=0.05,
                           last_stable_periods=5):
    """Finds period of a periodic oscillating signal as the source field.

    Parameters
    ----------
    source_field : np.array of dimension 1
        One-dimensional array of a periodic oscillating signal. Generally, 
        field array of shape (K) where K stands for different time instants.
    t_line=None : np.array of dimension 1, optional
        One-dimensional array of periodic oscillating signal's independent 
        variable. Generally, time array of shape (K) where K stands for 
        different time instants. If None is given, a default integer index 
        number array is created.
    peaks_sep_sensitivity=0.1 : float between zero and one, optional
        A factor representing the allowed variation percentage in consecutive 
        peaks separation. Any deviated value will be dropped.
    periods_sensitivity=0.05 : float between zero and one, optional
        A factor representing the allowed variation percentage in the period 
        of selected peaks. All values prior to a certain point will be dropped, 
        keeping only the last values identified to be stable.
    last_stable_periods=5 : int, optional
        Number of periods to take as reference of the stable signal, extracted 
        from the end as the signal is assumed to have both a transcient and a 
        stationary regimen.

    Returns
    -------
    float
        Mean period of the signal, taking into account only those peaks 
        identified to be stable.
    """
    
    if t_line is None:
        t_line = np.arange(len(source_field))
    
    peaks = get_peaks_from_source(source_field, 
                                  peaks_sep_sensitivity=peaks_sep_sensitivity,
                                  last_stable_periods=last_stable_periods)
    semiperiods = np.array(t_line[peaks[1:]] - t_line[peaks[:-1]])
    
    # Take the last periods as reference and define stability criteria
    mean_semiperiods = np.mean(semiperiods[-2*last_stable_periods:])
    def selection_criteria(k):
        eval_point = np.abs(semiperiods[k] - mean_semiperiods)
        return eval_point <= periods_sensitivity * mean_semiperiods
    
    # Choose only the latter stable periods to compute period
    keep_periods_from = 0
    for k in range(len(semiperiods)):
        if not selection_criteria(k):
            keep_periods_from = max(keep_periods_from, k+1)
    stable_semiperiods = semiperiods[keep_periods_from:]
    
    return 2*np.mean(stable_semiperiods)

def get_amplitude_from_source(source_field,
                              peaks_sep_sensitivity=0.1,
                              amplitude_sensitivity=0.05,
                              last_stable_periods=5):
    """Finds amplitude of a periodic oscillating signal as the source field.

    Parameters
    ----------
    source_field : np.array of dimension 1
        One-dimensional array of a periodic oscillating signal. Generally, 
        field array of shape (K) where K stands for different time instants.
    peaks_sep_sensitivity=0.1 : float between zero and one, optional
        A factor representing the allowed variation percentage in consecutive 
        peaks separation. Any deviated value will be dropped.
    amplitude_sensitivity=0.05 : float between zero and one, optional
        A factor representing the allowed variation percentage in the amplitude
        of selected peaks. All values prior to a certain point will be dropped, 
        keeping only the last values identified to be stable.
    last_stable_periods=5 : int, optional
        Number of periods to take as reference of the stable signal, extracted 
        from the end as the signal is assumed to have both a transcient and a 
        stationary regimen.

    Returns
    -------
    float
        Mean amplitude of the signal, taking into account only those peaks 
        identified to be stable.
    """
    
    peaks = get_peaks_from_source(source_field, 
                                  peaks_sep_sensitivity=peaks_sep_sensitivity,
                                  last_stable_periods=last_stable_periods)
    heights = np.abs(source_field[peaks])
    
    # Take the last periods as reference and define stability criteria
    mean_height = np.mean(heights[-2*last_stable_periods:])
    def selection_criteria(k):
        eval_point = np.abs(heights[k] - mean_height)
        return eval_point <= amplitude_sensitivity * mean_height
    
    # Choose only the latter stable periods to compute amplitude
    amp_keep_periods_from = 0
    for k in range(len(heights)):
        if not selection_criteria(k):
            amp_keep_periods_from = max(amp_keep_periods_from, k+1)
    stable_heights = np.abs(source_field[peaks[amp_keep_periods_from:]])
    
    return np.mean(stable_heights)   

#%% FIELD ANALYSIS: ZPROFILE FROM YZ PLANE

def get_background_from_plane(yzplane_field, y_plane_index, z_plane_index,
                              cell_width, pml_width):
    """Extracts data corresponding to background field from YZ planes field array.

    Parameters
    ----------
    yzplane_field : np.array with dimension 3
        Three-dimensional field array of shape (N,M,K) where N stands for 
        positions in the Y axis, M stands for positions in the Z axis and K 
        stands for different time instants.
    y_plane_index : function
        Function of a single argument that takes in an Y position and returns 
        the index of the closest value.
    z_plane_index : function
        Function of a single argument that takes in an Z position and returns 
        the index of the closest value.
    cell_width : int, float
        Cubic cell side's total length, generally expressed in Meep units to 
        be in the same units as the `y_plane` metadata array.
    pml_width : int, float
        Cell's isotropic PML's width, generally expressed in Meep units to be 
        in the same units as the `y_plane` metadata array.

    Returns
    -------
    np.array
        One-dimensional field array of shape (K,) where K stands for different 
        time instants.
    """
    
    y0_wall = np.asarray(yzplane_field[y_plane_index(-cell_width/2 + pml_width), z_plane_index(0), :])
    y1_wall = np.asarray(yzplane_field[y_plane_index(cell_width/2 - pml_width), z_plane_index(0), :])
    return np.mean(np.array([y0_wall,y1_wall]), axis=0)    

def get_zprofile_from_plane(yzplane_field, y_plane_index):
    """Extracts data corresponding to Z axis from YZ planes field array.

    Parameters
    ----------
    yzplane_field : np.array with dimension 3
        Three-dimensional field array of shape (N,M,K) where N stands for 
        positions in the Y axis, M stands for positions in the Z axis and K 
        stands for different time instants.
    y_plane_index : function
        Function of a single argument that takes in an Y position and returns 
        the index of the closest value.

    Returns
    -------
    np.array
        Bidimensional field array of shape (M,K) where M stands for 
        positions in the Z axis and K stands for different time instants.
    """
    
    return np.asarray(yzplane_field[y_plane_index(0), :, :])

def integrate_field_zprofile(zprofile_field, z_plane_index,
                             cell_width, pml_width, period_plane):
    """Integrates inside of the cell Z profile fields from Z lines field array.

    Parameters
    ----------
    zprofile_field : np.array with dimension 2
        Bidimensional field array of shape (N,K) where N stands for positions 
        in the Z axis and K stands for different time instants.
    z_plane_index : function
        Function of a single argument that takes in an Z position and returns 
        the index of the closest value.
    cell_width : int, float
        Cubic cell side's total length, generally expressed in Meep units to 
        be in the same units as the `y_plane` metadata array.
    pml_width : int, float
        Cell's isotropic PML's width, generally expressed in Meep units to be 
        in the same units as the `y_plane` metadata array.
    period_plane : float
        Sampling period of the data, generally expressed in Meep units to be 
        in the correct units to be input of `mp.at_every` step functions.

    Returns
    -------
    integral : np.array with dimension 1
        One-dimensional array of shape (K,) where K stands for different time 
        instants.
    """
    
    integral = np.sum(
        crop_field_zprofile(zprofile_field, z_plane_index,
                            cell_width, pml_width), 
        axis=0) * period_plane
    
    return integral

def detect_sign_field_zprofile(zprofile_field):
    """Detects sign of Z profile fields from Z lines field array.

    Parameters
    ----------
    zprofile_field : np.array with dimension at least 1
        Field array of shape (...,K) where K stands for different time instants.

    Returns
    -------
    np.array
        Sign array of shape (...,K) where K stands for different time instants.
    """
    
    return -np.sign(zprofile_field[0,...])

def find_zpeaks_zprofile(zprofile_field, z_plane_index, 
                         cell_width, pml_width):
    """Finds peaks in Z axis for a Z lines field array.
    
    Parameters
    ----------
    zprofile_field : np.array with dimension 2
        Bidimensional field array of shape (N,K) where N stands for positions 
        in the Z axis and K stands for different time instants.
    z_plane_index : function
        Function of a single argument that takes in an Z position and returns 
        the index of the closest value.
    cell_width : int, float
        Cubic cell side's total length, generally expressed in Meep units to 
        be in the same units as the `y_plane` metadata array.
    pml_width : int, float
        Cell's isotropic PML's width, generally expressed in Meep units to be 
        in the same units as the `y_plane` metadata array.

    Returns
    -------
    abs_max_z_index : int
        Index n<N identified to extract all data corresponding to the maximum 
        values of field in the Z axis.
    max_z_values : np.array with dimension 1
        One-dimensional array of shape (K,) where K stands for different time 
        instants.
    """
    
    cropped_zprofile = crop_single_field_zprofile(zprofile_field, z_plane_index, 
                                           cell_width, pml_width)
    
    # Find the position of the absolute maximum
    abs_max_index = np.argmax(np.abs(cropped_zprofile))
    abs_max_z_index = np.unravel_index(abs_max_index, cropped_zprofile.shape)[0]
    
    # Take its position in Z axis as reference and get all values of the field
    max_z_values = cropped_zprofile[abs_max_z_index, ...]
    
    return abs_max_z_index, max_z_values

def get_phase_field_peak_background(zprofmax_field, background_field,
                                    peaks_sep_sensitivity=0.1,
                                    periods_sensitivity=0.05,
                                    last_stable_periods=5):
    """Calculates relative phase between field in peaks Z location and background.

    Parameters
    ----------
    zprofmax_field : np.array with dimension 1
        One-dimensional oscillating signal. Generally, field array, 
        corresponding to Z peak location, of shape (K,) where K stands for 
        different time instants.
    background_field : np.array with dimension 1
        One-dimensional reference oscillating signal, of the same period. 
        Generally, background field array of shape (K,) where K stands for 
        different time instants.
    peaks_sep_sensitivity=0.1 : float between zero and one, optional
        A factor representing the allowed variation percentage in consecutive 
        peaks separation. Any deviated value will be dropped.
    periods_sensitivity=0.05 : float between zero and one, optional
        A factor representing the allowed variation percentage in the period 
        of selected peaks. All values prior to a certain point will be dropped, 
        keeping only the last values identified to be stable.
    last_stable_periods=5 : int, optional
        Number of periods to take as reference of the stable signal, extracted 
        from the end as the signal is assumed to have both a transcient and a 
        stationary regimen.

    Returns
    -------
    delta_phase : float
        Relative phase between field in peaks Z location and background field, 
        expressed in multiples of pi radians.
    """
    
    back_index = get_peaks_from_source(background_field, 
                                       peaks_sep_sensitivity=peaks_sep_sensitivity,
                                       last_stable_periods=last_stable_periods)
    zprof_index = get_peaks_from_source(zprofmax_field, 
                                        peaks_sep_sensitivity=peaks_sep_sensitivity,
                                        last_stable_periods=last_stable_periods)
    
    back_iperiod = get_period_from_source(background_field,
                                          peaks_sep_sensitivity=peaks_sep_sensitivity,
                                          periods_sensitivity=periods_sensitivity,
                                          last_stable_periods=last_stable_periods)
    zprof_iperiod = get_period_from_source(zprofmax_field,
                                           peaks_sep_sensitivity=peaks_sep_sensitivity,
                                           periods_sensitivity=periods_sensitivity,
                                           last_stable_periods=last_stable_periods)
    
    if zprofmax_field[ zprof_index[0] ]>0:
        zprof_imaxs = np.array( zprof_index[::2] )
        zprof_imins = np.array( zprof_index[1::2] )
    else:
        zprof_imins = np.array( zprof_index[::2] )
        zprof_imaxs = np.array( zprof_index[1::2] )
        
    if background_field[ back_index[0]] >0:
        back_imaxs = np.array( back_index[::2] )
        back_imins = np.array( back_index[1::2] )
    else:
        back_imins = np.array( back_index[::2] )
        back_imaxs = np.array( back_index[1::2] )
    
    min_number = np.min([len(zprof_imaxs), len(zprof_imins), 
                         len(back_imaxs), len(back_imins)])
    
    zprof_imaxs, zprof_imins = zprof_imaxs[-min_number:], zprof_imins[-min_number:]
    back_imaxs, back_imins = back_imaxs[-min_number:], back_imins[-min_number:]
        
    if np.abs(np.mean(zprof_imins - back_imins)) < np.abs(np.mean(zprof_imins - back_imaxs)):
        delta_i = np.mean([ np.abs(np.mean(zprof_imins - back_imins)),
                            np.abs(np.mean(zprof_imaxs - back_imaxs)) ])
    else:
        delta_i = np.mean([ np.abs(np.mean(zprof_imins - back_imaxs)),
                            np.abs(np.mean(zprof_imaxs - back_imins)) ])
        
    delta_phase = 2 * delta_i / np.mean([back_iperiod, zprof_iperiod]) # in multiples of pi radians
    
    return delta_phase

#%% FIELD ANALYSIS: RESONANCE FROM ZPROFILE

def get_all_field_peaks_from_yzplanes(yzplane_field,
                                      y_plane_index, z_plane_index, 
                                      cell_width, pml_width,
                                      peaks_sep_sensitivity=0.05,
                                      last_stable_periods=5):
    """Gets all maximum intensification field from a YZ planes field array.

    Parameters
    ----------
    yzplane_field : np.array with dimension 3
        Three-dimensional field array of shape (N,M,K) where N stands for 
        positions in the Y axis, M stands for positions in the Z axis and K 
        stands for different time instants.
    y_plane_index : function
        Function of a single argument that takes in an Y position and returns 
        the index of the closest value.
    z_plane_index : function
        Function of a single argument that takes in an Z position and returns 
        the index of the closest value.
    cell_width : int, float
        Cubic cell side's total length, generally expressed in Meep units to 
        be in the same units as the `y_plane` metadata array.
    pml_width : int, float
        Cell's isotropic PML's width, generally expressed in Meep units to be 
        in the same units as the `y_plane` metadata array.
    peaks_sep_sensitivity=0.1 : float between zero and one, optional
        A factor representing the allowed variation percentage in consecutive 
        peaks separation. Any deviated value will be dropped.
    last_stable_periods=5 : int, optional
        Number of periods to take as reference of the stable signal, extracted 
        from the end as the signal is assumed to have both a transcient and a 
        stationary regimen.

    Returns
    -------
    field_peaks_index : np.array
        Maximum intensification index k<K in an array of shape (Kfp,) where 
        Kfp<K stands for the selected time instants.
    field_peaks_amplitudes : np.array
        Absolute amplitude of maximum intensification in an array of shape 
        (Kfp,) where Kfp<K stands for the selected time instants.
    field_peaks_zprofile : np.array
        Bidimensional cropped field array of shape (n,Kfp) where n<N stands for 
        positions in the Z axis and Kfp<K stands for the selected time instants.
    field_peaks_yzplane : np.array
        Three-dimensional cropped field array of shape (n,m,Kfp) where n<N 
        stands for positions in the Y axis, m<M stands for positions in the 
        Z axis and Kfp<K stands for the selected time instants.
    """
    # zprofile_field has time as the last dimension
    
    zprofile_field = get_zprofile_from_plane(yzplane_field, y_plane_index)
    
    zprofile_maxs = find_zpeaks_zprofile(zprofile_field, z_plane_index, 
                                         cell_width, pml_width)[-1]
    
    # Find peaks in time for the maximum absolute intensification field values
    field_peaks_index = find_peaks(
        np.abs(zprofile_maxs), height=(np.max(np.abs(zprofile_maxs))/2, None))[0]
    
    # Take the last periods as reference and define periodicity criteria
    mean_diff = np.mean(np.diff(field_peaks_index)[-2*last_stable_periods:])
    def selection_criteria(k):
        eval_point = np.abs( mean_diff - (field_peaks_index[k+1] - field_peaks_index[k]) )
        return eval_point <= peaks_sep_sensitivity * mean_diff
    
    # Filter peaks to make sure they are periodic; i.e. they have equispaced index
    selected_index = []
    for k in range(len(field_peaks_index)):
        if k == 0 and selection_criteria(k):
            selected_index.append(k)
        elif k == len(field_peaks_index)-1 and selection_criteria(k-1):
            selected_index.append(k)
        elif selection_criteria(k):
            selected_index.append(k)
    
    # Check if there's always a single maximum and a single minimum per period
    # If not, try to fix it or raise Warning.
    selected_sign = np.sign(zprofile_maxs[field_peaks_index[selected_index]])
    if np.min(np.abs(np.diff( selected_sign ))) != 2:
        error_index = np.argmin(np.abs(np.diff( selected_sign )))
        missing_index_lower_bound = selected_index[ error_index ]
        missing_index_upper_bound = selected_index[ error_index + 1 ]
        if missing_index_upper_bound - missing_index_lower_bound == 2:
            selected_index = [*selected_index[: error_index + 1],
                              missing_index_lower_bound + 1,
                              *selected_index[error_index + 1 :]]
            selected_sign = np.sign(zprofile_maxs[field_peaks_index[selected_index]])
            if np.min(np.abs(np.diff( selected_sign ))) != 2:
                print("Warning! Sign algorithm failed and it couldn't be fixed!")
        else:
            print("Warning! Sign algorithm must have failed!")
    
    field_peaks_index = field_peaks_index[selected_index]
    
    field_peaks_amplitudes = np.abs(zprofile_maxs[field_peaks_index])
    field_peaks_sign = np.sign(zprofile_maxs[field_peaks_index])
        
    field_peaks_zprofile = [crop_single_field_zprofile(zprofile_field[...,k], 
                                                       z_plane_index, 
                                                       cell_width, pml_width) 
                            for k in field_peaks_index]
    field_peaks_zprofile = field_peaks_sign * np.array(field_peaks_zprofile).T
    
    field_peaks_yzplane = [crop_single_field_yzplane(yzplane_field[...,k], 
                                                     y_plane_index, z_plane_index, 
                                                     cell_width, pml_width) 
                           for k in field_peaks_index]
    field_peaks_yzplane = field_peaks_sign * np.array(field_peaks_yzplane).T
    
    return [field_peaks_index, field_peaks_amplitudes, 
            field_peaks_zprofile, field_peaks_yzplane]

def get_single_field_peak_from_yzplanes(yzplane_field,
                                        y_plane_index, z_plane_index, 
                                        cell_width, pml_width,
                                        peaks_sep_sensitivity=0.05,
                                        field_peaks_sensitivity=0.01,
                                        last_stable_periods=5):
    """Selects single maximum intensification field from a YZ planes field array.

    Parameters
    ----------
    yzplane_field : np.array with dimension 3
        Three-dimensional field array of shape (N,M,K) where N stands for 
        positions in the Y axis, M stands for positions in the Z axis and K 
        stands for different time instants.
    y_plane_index : function
        Function of a single argument that takes in an Y position and returns 
        the index of the closest value.
    z_plane_index : function
        Function of a single argument that takes in an Z position and returns 
        the index of the closest value.
    cell_width : int, float
        Cubic cell side's total length, generally expressed in Meep units to 
        be in the same units as the `y_plane` metadata array.
    pml_width : int, float
        Cell's isotropic PML's width, generally expressed in Meep units to be 
        in the same units as the `y_plane` metadata array.
    peaks_sep_sensitivity=0.1 : float between zero and one, optional
        A factor representing the allowed variation percentage in consecutive 
        peaks separation. Any deviated value will be dropped.
    field_peaks_sensitivity=0.01 : float between zero and one, optional
        A factor representing the allowed variation percentage in the absolute 
        amplitude of selected peaks. All values prior to a certain point will 
        be dropped, keeping only the last values identified to be stable.
    last_stable_periods=5 : int, optional
        Number of periods to take as reference of the stable signal, extracted 
        from the end as the signal is assumed to have both a transcient and a 
        stationary regimen.

    Returns
    -------
    field_peaks_index : int
        Maximum intensification index k<Kfp where Kfp<K stands for the 
        maximum intensification time instants.
    field_peaks_amplitude : float
        Absolute amplitude of maximum intensification corresponding to time 
        index k<Kfp where Kfp<K stands for the maximum intensification time 
        instants.
    field_peaks_zprofile : np.array
        Bidimensional cropped field array of shape (n,) where n<N stands for 
        positions in the Z axis. It also corresponds to time index k<Kfp where 
        Kfp<K stands for the maximum intensification time 
        instants.
    field_peaks_yzplane : np.array
        Three-dimensional cropped field array of shape (n,m) where n<N 
        stands for positions in the Y axis and m<M stands for positions in the 
        Z axis. It also corresponds to time index k<Kfp where Kfp<K stands for 
        the maximum intensification time 
        
    See also
    --------
    get_all_field_peaks_from_yzplanes
    """
    # zprofile_field has time as the last dimension
    
    all_index, all_amplitudes, all_zprofile, all_yzplane = get_all_field_peaks_from_yzplanes(
        yzplane_field, y_plane_index, z_plane_index, cell_width, pml_width,
        peaks_sep_sensitivity=peaks_sep_sensitivity, last_stable_periods=last_stable_periods)
    
    # Take the last periods as reference and define stability criteria
    mean_value = np.mean( all_amplitudes[-2*last_stable_periods:] )
    def selection_criteria(k):
        eval_point = np.abs(all_amplitudes[k] - mean_value)
        return eval_point <= field_peaks_sensitivity * mean_value
    
    # Choose only the latter stable periods to search for maximum amplitude
    keep_periods_from = 0
    for k in range(len(all_amplitudes)):
        if not selection_criteria(k):
            keep_periods_from = max(keep_periods_from, k+1)
    
    selected_index = np.argmax( all_amplitudes[keep_periods_from:] ) + keep_periods_from
    
    field_peaks_index = int(all_index[selected_index])
    field_peaks_amplitude = float(all_amplitudes[selected_index])
    field_peaks_zprofile = all_zprofile[..., selected_index]
    field_peaks_yzplane = all_yzplane[..., selected_index]
    
    return [field_peaks_index, field_peaks_amplitude, 
            field_peaks_zprofile, field_peaks_yzplane]

def get_mean_field_peak_from_yzplanes(yzplane_field,
                                      y_plane_index, z_plane_index, 
                                      cell_width, pml_width,
                                      peaks_sep_sensitivity=0.05,
                                      field_peaks_sensitivity=0.01,
                                      last_stable_periods=5):
    """Computes a mean maximum intensification field from a YZ planes field array.

    Parameters
    ----------
    yzplane_field : np.array with dimension 3
        Three-dimensional field array of shape (N,M,K) where N stands for 
        positions in the Y axis, M stands for positions in the Z axis and K 
        stands for different time instants.
    y_plane_index : function
        Function of a single argument that takes in an Y position and returns 
        the index of the closest value.
    z_plane_index : function
        Function of a single argument that takes in an Z position and returns 
        the index of the closest value.
    cell_width : int, float
        Cubic cell side's total length, generally expressed in Meep units to 
        be in the same units as the `y_plane` metadata array.
    pml_width : int, float
        Cell's isotropic PML's width, generally expressed in Meep units to be 
        in the same units as the `y_plane` metadata array.
    peaks_sep_sensitivity=0.1 : float between zero and one, optional
        A factor representing the allowed variation percentage in consecutive 
        peaks separation. Any deviated value will be dropped.
    field_peaks_sensitivity=0.01 : float between zero and one, optional
        A factor representing the allowed variation percentage in the absolute 
        amplitude of selected peaks. All values prior to a certain point will 
        be dropped, keeping only the last values identified to be stable.
    last_stable_periods=5 : int, optional
        Number of periods to take as reference of the stable signal, extracted 
        from the end as the signal is assumed to have both a transcient and a 
        stationary regimen.

    Returns
    -------
    field_peaks_index : int
        Maximum intensification index k<Kfp where Kfp<K stands for the 
        maximum intensification time instants.
    field_peaks_amplitude : float
        Absolute amplitude of maximum intensification corresponding to time 
        index k<Kfp where Kfp<K stands for the maximum intensification time 
        instants.
    field_peaks_zprofile : np.array
        Bidimensional cropped field array of shape (n,) where n<N stands for 
        positions in the Z axis. It also corresponds to time index k<Kfp where 
        Kfp<K stands for the maximum intensification time 
        instants.
    field_peaks_yzplane : np.array
        Three-dimensional cropped field array of shape (n,m) where n<N 
        stands for positions in the Y axis and m<M stands for positions in the 
        Z axis. It also corresponds to time index k<Kfp where Kfp<K stands for 
        the maximum intensification time 
        
    See also
    --------
    get_all_field_peaks_from_yzplanes
    """
    # zprofile_field has time as the last dimension
    
    all_index, all_amplitudes, all_zprofile, all_yzplane = get_all_field_peaks_from_yzplanes(
        yzplane_field, y_plane_index, z_plane_index, cell_width, pml_width,
        peaks_sep_sensitivity=peaks_sep_sensitivity, last_stable_periods=last_stable_periods)
    
    # Take the last periods as reference and define stability criteria
    mean_value = np.mean( all_amplitudes[-2*last_stable_periods:] )
    def selection_criteria(k):
        eval_point = np.abs(all_amplitudes[k] - mean_value)
        return eval_point <= field_peaks_sensitivity * mean_value
    
    # Choose only the latter stable periods to search for maximum amplitude
    keep_periods_from = 0
    for k in range(len(all_amplitudes)):
        if not selection_criteria(k):
            keep_periods_from = max(keep_periods_from, k+1)
    
    selected_index = np.arange(keep_periods_from, len(all_amplitudes))
    
    # maximum_index = np.argmax( all_amplitudes[keep_periods_from:] ) + keep_periods_from
    # field_peaks_index = int(all_index[maximum_index])    
    
    field_peaks_amplitude = np.mean(all_amplitudes[selected_index])
    field_peaks_zprofile = np.mean(all_zprofile[..., selected_index], axis=-1)
    field_peaks_yzplane = np.mean(all_yzplane[..., selected_index], axis=-1)
    
    closest_index = np.argmin( np.abs(all_amplitudes[keep_periods_from:] - np.mean(all_amplitudes[keep_periods_from:])) ) + keep_periods_from
    field_peaks_index = int(all_index[closest_index])
    
    return [field_peaks_index, field_peaks_amplitude, 
            field_peaks_zprofile, field_peaks_yzplane]
   