#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains tools for plotting with customized style.

The functions contained right now are...

set_style : function 
    Gives NanoMeepPlasmonics thesis style to a figure.
add_style : function
    Gives style to figures to include in Latex PDF files.
add_subplot_axes : function
    Adds a sub-set of axes inside an existing set of axes.

@author: Vall
"""

#%%

from matplotlib import rcParams, ticker
import matplotlib.pyplot as plt

#%% CUSTOMIZATION OPTIONS

def set_style():
    """Gives NanoMeepPlasmonics thesis style to a figure."""
    
    plt.rcParams.update({'text.usetex': False, 
                         'font.family':'serif',
                         'font.sans-serif': ['MS Reference Sans Serif', 'sans-serif'], 
                         'mathtext.fontset': 'cm', # Computer Modern
                         'font.weight':500,
                         'figure.titlesize':13,
                         'axes.titlesize':12,
                         'axes.labelsize':11,
                         'legend.fontsize':11,
                         'xtick.labelsize':10,
                         'ytick.labelsize':10,
                         'xtick.minor.visible':True,
                         'ytick.minor.visible':True,
                         'grid.alpha':0.4,
                         'axes.grid':True,
                         'xtick.color':'b0b0b0',
                         'ytick.color':'b0b0b0',
                         'xtick.labelcolor':'black',
                         'ytick.labelcolor':'black',
                         'legend.frameon':False,
                         'legend.framealpha':0,
                         'lines.markersize':8
                         })

def add_style(figure_id=None, new_figure=False, **kwargs):
    """Gives style to figures to include in Latex PDF files.
    
    This function...
        ...increases font size;
        ...increases linewidth;
        ...increases markersize;
        ...gives format to axis ticks if specified;
        ...stablishes new figure dimensions if specified;
        ...activates grid.
    
    Parameters
    ----------
    figure_id : int, optional
        ID of the figure where the text will be printed.
        If none is given, the current figure is taken as default.
    new_figure=False : bool, optional
        Indicates whether to make a new figure or not when 
        figure_id=None.
    
    Other Parameters
    ----------------
    xaxisformat : format-like str, optional.
        Used to update x axis ticks format; i.e.: '%.2e'
    yaxisformat : format-like str, optional.
        Used to update y axis ticks format; i.e.: '%.2e'
    dimensions: list with length 4, optional.
        Used to update plot dimensions: [xmin, xmax, ymin, ymax]. Each 
        one should be a number expressed as a fraction of current 
        dimensions.
    
    See Also
    --------
    matplotlib.pyplot.axis
    matplotlib.pyplot.gcf
    
    """
    
    if figure_id is not None:
        fig = plt.figure(figure_id)
    elif new_figure:
        fig = plt.figure()
    else:
        fig = plt.gcf()

    try:
        ax = fig.axes
        ax[0]
    except IndexError:
        ax = [plt.axes()]
    
    kwargs_default = dict(
            fontsize=12,
            linewidth=3,
            markersize=6,
            dimensions=[1.15,1.05,1,1],
            tight_layout=True,
            grid=False,
            xaxisformat=None,
            yaxisformat=None)
    
    kwargs = {key:kwargs.get(key, value) 
              for key, value in kwargs_default.items()}
    
    rcParams.update({'font.size': kwargs['fontsize']})
    rcParams.update({'lines.linewidth': kwargs['linewidth']})
    rcParams.update({'lines.markersize': kwargs['markersize']})
    for a in ax:
        box = a.get_position()
        a.set_position([kwargs['dimensions'][0]*box.x0,
                        kwargs['dimensions'][1]*box.y0,
                        kwargs['dimensions'][2]*box.width,
                        kwargs['dimensions'][3]*box.height])
    
    if kwargs['xaxisformat'] is not None:
        for a in ax:
            a.xaxis.set_major_formatter(ticker.FormatStrFormatter(
                kwargs['xaxisformat']))
        
    if kwargs['yaxisformat'] is not None:
        for a in ax:
            a.yaxis.set_major_formatter(ticker.FormatStrFormatter(
                kwargs['yaxisformat']))
        
    for a in ax:
        a.grid(kwargs['grid'])
        
    fig.tight_layout = kwargs['tight_layout']
    
    plt.show()
   
#%%

def add_subplot_axes(ax, rect):
    """Adds a sub-set of axes inside an existing set of axes.
    
    This function was taken from StackOverflow on September 2021.
    https://stackoverflow.com/questions/17458580/embedding-small-plots-inside-subplots-in-matplotlib
    
    Parameters
    ----------
    ax : plt.Axes or plt.AxesSubplot instance
        The currently existing set of axes.
    rect : list or iterable of length 4
        The desired dimensions for the new sub-set of axes: [x, y, width, height]
    
    Returns
    -------
    subax : plt.AxesSubplot instance
        The newly created sub-set of axes.
    """
    
    
    fig = ax.figure #plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    
    subax = fig.add_axes([x,y,width,height])
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    
    return subax