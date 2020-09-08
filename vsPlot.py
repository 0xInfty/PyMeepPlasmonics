#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The 'vsPlot' module contains tools for plotting.

The only tool contained right now is...
add_style : function
    Gives a specific style to figure.


@author: Vall
"""

#%%

from matplotlib import rcParams, ticker
import matplotlib.pyplot as plt

#%% CUSTOMIZATION OPTIONS

def add_style(figure_id=None, new_figure=False, **kwargs):
    """Gives a specific style to figure.
    
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
        fig = plt.figure(figure_id, tight_layout=True)
    elif new_figure:
        fig = plt.figure(tight_layout=True)
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
    
    plt.show()