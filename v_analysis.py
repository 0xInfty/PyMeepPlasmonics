# -*- coding: utf-8 -*-
"""
This module contains tools for data analysis.

Some of its most useful tools are:

linear_fit : function
    Applies linear fit and returns m, b and Rsq. Can also plot it.
nonlinear_fit : function
    Applies nonlinear fit and returns parameters and Rsq. Plots it.
error_value : function
    Rounds up value and error of a measure.
@author: Vall
"""

import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
from scipy.optimize import curve_fit
import v_utilities as vu

#%%

def linear_fit(X, Y, dY=None, showplot=True,
               plot_some_errors=(False, 20), **kwargs):

    """Applies linear fit and returns m, b and Rsq. Can also plot it.
    
    By default, it applies minimum-square linear fit 'y = m*x + b'. If 
    dY is specified, it applies weighted minimum-square linear fit.
    
    Parameters
    ----------
    X : np.array, list
        Independent X data to fit.
    Y : np-array, list
        Dependent Y data to fit.
    dY : np-array, list
        Dependent Y data's associated error.
    showplot : bool
        Says whether to plot or not.
    plot_some_errors : tuple (bool, int)
        Says wehther to plot only some error bars (bool) and specifies 
        the number of error bars to plot.
    
    Returns
    -------
    rsq : float
        Linear fit's R Square Coefficient.
    (m, dm): tuple (float, float)
        Linear fit's slope: value and associated error, both as float.
    (b, db): tuple (float, float)
        Linear fit's origin: value and associated error, both as float.
        
    Other Parameters
    ----------------
    txt_position : tuple (horizontal, vertical), optional
        Indicates the parameters' text position. Each of its values 
        should be a number (distance in points measured on figure). 
        But vertical value can also be 'up' or 'down'.
    mb_units : tuple (m_units, b_units), optional
        Indicates the parameter's units. Each of its values should be a 
        string.
    mb_error_digits : tuple (m_error_digits, b_error_digits), optional
        Indicates the number of digits to print in the parameters' 
        associated error. Default is 3 for slope 'm' and 2 for intercept 
        'b'.
    mb_string_scale : tuple (m_string_scale, b_string_scale), optional
        Indicates whether to apply string prefix's scale to printed 
        parameters. Each of its values should be a bool; i.e.: 'True' 
        means 'm=1230.03 V' with 'dm = 10.32 V' would be printed as 
        'm = (1.230 + 0.010) V'. Default is '(False, False)'.
    rsq_decimal_digits : int, optional.
        Indicates the number of digits to print in the Rsq. Default: 3.
        
    Warnings
    --------
    The returned Rsq doesn't take dY weights into account.
    
    """

    # ¿Cómo hago Rsq si tengo pesos?
    
    if dY is None:
        W = None
    else:
        W = 1/dY**2
        
    fit_data = np.polyfit(X, Y, 1, cov=True, w=W)
    
    m = fit_data[0][0]
    dm = sqrt(fit_data[1][0,0])
    b = fit_data[0][1]
    db = sqrt(fit_data[1][1,1])
    rsq = 1 - sum( (Y - m*X - b)**2 )/sum( (Y - np.mean(Y))**2 )

    try:
        kwargs['text_position']
    except KeyError:
        if m > 1:
            aux = 'up'
        else:
            aux = 'down'
        kwargs['text_position'] = (.02, aux)

    if showplot:

        plt.figure()
        if dY is None:
            plt.plot(X, Y, 'b.', zorder=0)
        else:
            if plot_some_errors[0] == False:
                plt.errorbar(X, Y, yerr=dY, linestyle='', marker='.',
                             ecolor='b', elinewidth=1.5, zorder=0)
            else:
                plt.errorbar(X, Y, yerr=dY, linestyle='', marker='.',
                             color='b', ecolor='b', elinewidth=1.5,
                             errorevery=len(Y)/plot_some_errors[1], 
                             zorder=0)
        plt.plot(X, m*X+b, 'r-', zorder=100)
        plt.legend(['Datos', 'Ajuste'])
        
        kwargs_list = ['mb_units', 'mb_string_scale', 
                       'mb_error_digits', 'rsq_decimal_digits']
        kwargs_default = [('', ''), (False, False), (3, 2), 3]

        for key, value in zip(kwargs_list, kwargs_default):
            try:
                kwargs[key]
            except KeyError:
                kwargs[key] = value
        
        if kwargs['text_position'][1] == 'up':
            vertical = [.9, .82, .74]
        elif kwargs['text_position'][1] == 'down':
            vertical = [.05, .13, .21]
        else:
            if kwargs['text_position'][1] <= .08:
                fact = .08
            else:
                fact = -.08
            vertical = [kwargs['text_position'][1]+fact*i for i in range(3)]
        

        plt.annotate('m = {}'.format(error_value_latex(
                        m, 
                        dm,
                        error_digits=kwargs['mb_error_digits'][0],
                        units=kwargs['mb_units'][0],
                        string_scale=kwargs['mb_string_scale'][0],
                        one_point_scale=True)),
                    (kwargs['text_position'][0], vertical[0]),
                    xycoords='axes fraction')
        plt.annotate('b = {}'.format(error_value_latex(
                        b, 
                        db,
                        error_digits=kwargs['mb_error_digits'][1],
                        units=kwargs['mb_units'][1],
                        string_scale=kwargs['mb_string_scale'][1],
                        one_point_scale=True)),
                    (kwargs['text_position'][0], vertical[1]),
                    xycoords='axes fraction')
        rsqft = r'$R^2$ = {:.' + str(kwargs['rsq_decimal_digits']) + 'f}'
        plt.annotate(rsqft.format(rsq),
                    (kwargs['text_position'][0], vertical[-1]),
                    xycoords='axes fraction')
        
        plt.show()

    return rsq, (m, dm), (b, db)

#%%

def nonlinear_fit(X, Y, fitfunction, initial_guess=None, dY=None, 
                  showplot=True, plot_some_errors=(False, 20), 
                  parameters_bounds=(-np.inf, np.inf), **kwargs):

    """Applies nonlinear fit and returns parameters and Rsq. Plots it.
    
    By default, it applies minimum-square fit. If dY is specified, it 
    applies weighted minimum-square fit.
    
    Parameters
    ----------
    X : np.array, list
        Independent X data to fit.
    Y : np-array, list
        Dependent Y data to fit.
    fitfunction : function
        The function you want to apply. Its arguments must be 'X' as 
        np.array followed by the other parameters 'a0', 'a1', etc as 
        float. Must return only 'Y' as np.array.
    initial_guess=None : list, optional
        A list containing a initial guess for each parameter.
    parameters_bounds=None : list, optional
        A list containing a tuple of bounds for each parameter.
    dY : np-array, list, optional
        Dependent Y data's associated error.
    shoplot : bool
        Says whether to plot or not.
    plot_some_errors : tuple (bool, int)
        Says wehther to plot only some error bars (bool) and specifies 
        the number of error bars to plot.
    
    Returns
    -------
    rsq : float
        Fit's R Square Coefficient.
    parameters : list of tuples
        Fit's parameters, each as a tuple containing value and error, 
        both as tuples.
    
    Other Parameters
    -----------------
    txt_position : tuple (horizontal, vertical), optional
        Indicates the parameters' text position. Each of its values 
        should be a number (distance in points measured on figure). 
        But vertical value can also be 'up' or 'down'.
    par_units : list, optional
        Indicates the parameters' units. Each of its values should be a 
        string.
    par_error_digits : list, optional
        Indicates the number of digits to print in the parameters' 
        associated error. Default is 3 for all of them.
    par_string_scale : list, optional
        Indicates whether to apply string prefix's scale to printed 
        parameters. Each of its values should be a bool. Default is 
        False for all of them.
    rsq_decimal_digits : int, optional.
        Indicates the number of digits to print in the Rsq. Default: 3.
        
    Warnings
    --------
    The returned Rsq doesn't take dY weights into account.
    
    """
    
    if not isinstance(X, np.ndarray):
        raise TypeError("X should be a np.array")
    if not isinstance(Y, np.ndarray):
        raise TypeError("Y should be a np.array")
    if not isinstance(dY, np.ndarray) and dY is not None:
        raise TypeError("dY shouuld be a np.array")
    if len(X) != len(Y):
        raise IndexError("X and Y must have same lenght")
    if dY is not None and len(dY) != len(Y):
        raise IndexError("dY and Y must have same lenght")
    
    if dY is None:
        W = None
    else:
        W = 1/dY**2
    
    parameters, covariance = curve_fit(fitfunction, X, Y,
                                       p0=initial_guess, sigma=W,
                                       bounds=parameters_bounds)
    n = len(parameters)
    rsq = sum( (Y - fitfunction(X, *parameters))**2 )
    rsq = rsq/sum( (Y - np.mean(Y))**2 )
    rsq = 1 - rsq

    if showplot:
        
        plt.figure()
        if dY is None:
            plt.plot(X, Y, 'b.', zorder=0)
        else:
            if plot_some_errors[0] == False:
                plt.errorbar(X, Y, yerr=dY, linestyle='b', marker='.',
                             ecolor='b', elinewidth=1.5, zorder=0)
            else:
                plt.errorbar(X, Y, yerr=dY, linestyle='-', marker='.',
                             color='b', ecolor='b', elinewidth=1.5,
                             errorevery=len(Y)/plot_some_errors[1], 
                             zorder=0)
        plt.plot(np.linspace(min(X), max(X), 500), 
                 fitfunction(np.linspace(min(X), max(X), 500), *parameters), 
                 'r-', zorder=100)      
        plt.legend(['Datos', 'Ajuste'])        
        
        kwargs_list = ['text_position', 'par_units', 'par_string_scale', 
                       'par_error_digits', 'rsq_decimal_digits']
        kwargs_default = [(.02,'up'), ['' for i in range(n)], 
                          [False for i in range(n)], 
                          [3 for i in range(n)], 3]
        for key, value in zip(kwargs_list, kwargs_default):
            try:
                kwargs[key]
                if key != 'text_position':
                    try:
                        if len(kwargs[key]) != n:
                            print("Wrong number of parameters",
                                  "on '{}'".format(key))
                            kwargs[key] = value
                    except TypeError:
                        kwargs[key] = [kwargs[key] for i in len(n)]
            except KeyError:
                kwargs[key] = value
        
        if kwargs['text_position'][1] == 'up':
            vertical = [.9-i*.08 for i in range(n+1)]
        elif kwargs['text_position'][1] == 'down':
            vertical = [.05+i*.08 for i in range(n+1)]
        else:
            if kwargs['text_position'][1] <= .08:
                fact = .08
            else:
                fact = -.08
            vertical = [
                kwargs['text_position'][1]+fact*i for i in range(n+1)]
        
        for i in range(n):
            plt.annotate(
                    '$a_{}$ = {}'.format(
                        i,
                        error_value_latex(
                            parameters[i], 
                            sqrt(covariance[i,i]),
                            error_digits=kwargs['par_error_digits'][i],
                            units=kwargs['par_units'][i],
                            string_scale=kwargs['par_string_scale'][i],
                            one_point_scale=True)),
                    (kwargs['text_position'][0], vertical[i]),
                    xycoords='axes fraction')
        rsqft = r'$R^2$ = {:.' + str(kwargs['rsq_decimal_digits'])+'f}'
        plt.annotate(rsqft.format(rsq),
                    (kwargs['text_position'][0], vertical[-1]),
                    xycoords='axes fraction')
        
        plt.show()
    
    parameters_error = np.array(
            [sqrt(covariance[i,i]) for i in range(n)])
    parameters = list(zip(parameters, parameters_error))
    
    return rsq, parameters

#%%

def error_value(X, dX, error_digits=2, units='',
               string_scale=True, one_point_scale=False):
    
    """Rounds up value and error of a measure. Returns list of strings.
    
    This function takes a measure and its error as input. Then, it 
    rounds up both of them in order to share the same amount of decimal 
    places.
    
    After that, it generates a latex string containing the rounded up 
    measure. For that, it can rewrite both value and error so that the 
    classical prefix scale of units can be applied.
    
    Parameters
    ----------
    X : float
        Measurement's value.
    dX : float
        Measurement's associated error.
    error_digits=2 : int, optional.
        Desired number of error digits.
    units='' : str, optional.
        Measurement's units.
    string_scale=True : bool, optional.
        Whether to apply the classical prefix scale or not.        
    one_point_scale=False : bool, optional.
        Applies prefix with one order less.
    
    Returns
    -------
    string : list
        List of strings containing value and error.
    
    Examples
    --------
    >> errorValue(1.325412, 0.2343413)
    ['1.33', '0.23']
    >> errorValue(1.325412, 0.2343413, error_digits=3)
    ['1.325', '0.234']
    >> errorValue(.133432, .00332, units='V')
    ['133.4 mV', '3.3 mV']
    >> errorValue(.133432, .00332, one_point_scale=True, units='V')
    ['0.1334 V', '0.0033 V']
    >> errorValue(.133432, .00332, string_scale=False, units='V')
    ['1.334E-1 V', '0.033E-1 V']
    
    See Also
    --------
    copy
    
    """
    
    # First, I string-format the error to scientific notation with a 
    # certain number of digits
    if error_digits >= 1:
        aux = '{:.' + str(error_digits) + 'E}'
    else:
        print("Unvalid 'number_of_digits'! Changed to 1 digit")
        aux = '{:.0E}'
    error = aux.format(dX)
    error = error.split("E") # full error (string)
    
    error_order = int(error[1]) # error's order (int)
    error_value = error[0] # error's value (string)

    # Now I string-format the measure to scientific notation
    measure = '{:E}'.format(X)
    measure = measure.split("E") # full measure (string)
    measure_order = int(measure[1]) # measure's order (int)
    measure_value = float(measure[0]) # measure's value (string)
    
    # Second, I choose the scale I will put both measure and error on
    # If I want to use the string scale...
    if -12 <= measure_order < 12 and string_scale:
        prefix = ['p', 'n', r'$\mu$', 'm', '', 'k', 'M', 'G']
        scale = [-12, -9, -6, -3, 0, 3, 6, 9, 12]
        for i in range(8):
            if not one_point_scale:
                if scale[i] <= measure_order < scale[i+1]:
                    prefix = prefix[i] # prefix to the unit
                    scale = scale[i] # order of both measure and error
                    break
            else:
                if scale[i]-1 <= measure_order < scale[i+1]-1:
                    prefix = prefix[i]
                    scale = scale[i]
                    break
        used_string_scale = True
    # ...else, if I don't or can't...
    else:
        scale = measure_order
        prefix = ''
        used_string_scale = False
    
    # Third, I actually scale measure and error according to 'scale'
    # If error_order is smaller than scale...
    if error_order < scale:
        if error_digits - error_order + scale - 1 >= 0:
            aux = '{:.' + str(error_digits - error_order + scale - 1)
            aux = aux + 'f}'
            error_value = aux.format(
                    float(error_value) * 10**(error_order - scale))
            measure_value = aux.format(
                    measure_value * 10**(measure_order - scale))
        else:
            error_value = float(error_value) * 10**(error_order - scale)
            measure_value = float(measure_value)
            measure_value = measure_value * 10**(measure_order - scale)
    # Else, if error_order is equal or bigger than scale...
    else:
        aux = '{:.' + str(error_digits - 1) + 'f}'
        error_value = aux.format(
                float(error_value) * 10**(error_order - scale))
        measure_value = aux.format(
                float(measure_value) * 10**(measure_order - scale))
    
    # Forth, I make a string for each of them. Ex.: '1.34 kV'    
    string = [measure_value, error_value]
    if not used_string_scale and measure_order != 0:
        string = [st + 'E' + '{:.0f}'.format(scale) + ' ' + units 
                  for st in string]
    elif used_string_scale:
        string = [st + ' ' + prefix + units for st in string]        
    else:
        string = [st + ' ' + units for st in string]
    aux = []
    for st in string:
        if st[-1]==' ':
            aux.append(st[:len(st)-1])
        else:
            aux.append(st)
    string = aux
    
    return string

#%%

def error_value_latex(X, dX, error_digits=2, symbol='$\pm$', units='',
                      string_scale=True, one_point_scale=False):
    
    """Rounds up value and error of a measure. Also makes a latex string.
    
    This function takes a measure and its error as input. Then, it 
    rounds up both of them in order to share the same amount of decimal 
    places.
    
    After that, it generates a latex string containing the rounded up 
    measure. For that, it can rewrite both value and error so that the 
    classical prefix scale of units can be applied.
    
    Parameters
    ----------
    X : float
        Measurement's value.
    dX : float
        Measurement's associated error.
    error_digits=2 : int, optional.
        Desired number of error digits.
    units='' : str, optional.
        Measurement's units.
    string_scale=True : bool, optional.
        Whether to apply the classical prefix scale or not.        
    one_point_scale=False : bool, optional.
        Applies prefix with one order less.
    
    Returns
    -------
    latex_string : str
        Latex string containing value and error.
    
    Examples
    --------
    >> errorValueLatex(1.325412, 0.2343413)
    '(1.33$\\pm$0.23)'
    >> errorValueLatex(1.325412, 0.2343413, error_digits=3)
    '(1.325$\\pm$0.234)'
    >> errorValueLatex(.133432, .00332, units='V')
    '(133.4$\\pm$3.3) mV'
    >> errorValueLatex(.133432, .00332, one_point_scale=True, units='V')
    '(0.1334$\\pm$0.0033) V'
    >> errorValueLatex(.133432, .00332, string_scale=False, units='V')
    '(1.334$\\pm$0.033)$10^{-1}$ V'
    
    See Also
    --------
    copy
    errorValue
    
    """

    string = error_value(X, dX, error_digits, units,
                         string_scale, one_point_scale)

    try:
        measure = string[0].split(' ')[0].split("E")[0]
        error = string[1].split(' ')[0].split("E")[0]            
    except:
        measure = string[0].split("E")[0]
        error = string[1].split("E")[0]
        
    string_format = string[0].split(measure)[1].split(' ')

    if len(string_format)==2:
        if string_format[0]!='':
            order = vu.find_numbers(string_format[0])[0]
        else:
            order = 0
        unit = ' ' + string_format[1]
    elif string_format[0]=='':
        order = 0
        unit = ''
    elif units!='':
        order = 0
        unit = ' ' + string_format[0]
    else:
        order = vu.find_numbers(string_format[0])[0]
        unit = ''
    
    latex_string = r'({}{}{})'.format(measure, symbol, error)
    if order!=0:
        latex_string = latex_string + r'$10^{' + str(order) + r'}$' + unit
    else:
        latex_string = latex_string + unit
    
    return latex_string