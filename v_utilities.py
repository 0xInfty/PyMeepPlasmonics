# -*- coding: utf-8 -*-
"""The 'v_utilities' module contains tools appliable to a veriaty of tasks.

It could be divided into 3 main sections:

    (1) transforming to and from string other classes 
    ('nparray_to_string', 'dict_to_string', 'find_numbers', etc)
    (2) treating lists of strings ('filter_by_string_must', 'sort_by_number')
    (3) fixing bugs in my thesis' data due to code mistakes
    
Some of its most useful tools are...

find_numbers : function

sort_by_number : function
    Sorts list of strings by a variable number of recurrent position.
fix_params_dict : function
    Fixes the faulty params dict caused by wlen_range np.array on vs.savetxt

@author: Vall
"""

import numpy as np
from re import findall

#%%

def nparray_to_string(my_nparray):

    """Transforms np.ndarray to string in a retrievable way.
    
    Parameters
    ----------
    my_nparray : np.ndarray
        The np.ndarray to transform (could have ndim>1).
    
    Returns
    -------
    my_string : str
        The np.ndarray transformed to string.
    
    See also
    --------
    string_to_nparray
    """
                
    this_string = []
    for n in my_nparray:
        if not isinstance(n, np.ndarray):
            this_string.append(str(n))
        else:
            this_string.append( nparray_to_string(n) )
    my_string = "[" + ", ".join(this_string) + "]"
    
    return my_string

#%%

def dict_to_string(my_dict):

    """Transforms dict to string in a retrievable way.
    
    Parameters
    ----------
    my_dict : dict
        The dict to transform.
    
    Returns
    -------
    my_string : str
        The dict transformed to string.
    
    See also
    --------
    string_to_dict
    """
    
    aux = []
    for key, value in my_dict.items():
        if isinstance(value, str):
            value = '"{}"'.format(value)
        elif isinstance(value, np.ndarray):
            value = "np.array(" + nparray_to_string(value) + ")"
        elif isinstance(value, tuple) and len(value) == 2:
            condition = isinstance(value[0], str)
            if not condition and isinstance(value[1], str):
                value = '"{}, {}"'.format(*value)
        aux.append(f'{key}={value}' + ', ')
    my_string = ''.join(aux)[:-2]
                
    return my_string

#%%

def string_to_nparray(my_nparray_string):

    """Retrieves np.ndarray from string.
    
    Parameters
    ----------
    my_nparray_string : str
        The np.ndarray transformed to string.
    
    Returns
    -------
    my_nparray : np.ndarray
        The np.ndarray retrieved (could have ndim>1).
    
    See also
    --------
    nparray_to_string
    """
    
    return eval(f"np.array({my_nparray_string})")

#%%

def string_to_dict(my_dict_string):

    """Retrieves dict from string.
    
    Parameters
    ----------
    my_dict_string : str
        The dict transformed to string.
    
    Returns
    -------
    my_dict : dict
        The dict retrieved.
    
    See also
    --------
    dict_to_string
    """
    
    return eval(f"dict({my_dict_string})")

#%%

def find_numbers(string):

    """Returns a list of numbers found on a given string
    
    Parameters
    ----------
    string: str
        The string where you search.
    
    Returns
    -------
    list
        A list of numbers (each an int or float).
    
    Raises
    ------
    "There's no number in this string" : TypeError
        If no number is found.
    
    Warnings
    --------
    Doesn't recognize scientific notation.
    """
    
    numbers = findall(r"[-+]?\d*\.\d+|[-+]?\d+", string)
    
    if not numbers:
        raise TypeError("There's no number in this string")
    
    for i, n in enumerate(numbers):
        if '.' in n:
            numbers[i] = float(n)
        else:
            numbers[i] = int(n) 
    
    return numbers

#%%

def filter_by_string_must(string_list, string_must, must=True):

    """Filters list of str by a str required to be always present or absent.
    
    Parameters
    ----------
    string_list : list of str
        The list of strings to filter.
    string_must : str
        The string that must always be present or always absent on each of the 
        list elements.
    must=True : bool
        If true, then the string must always be present. If not, the string 
        must always be absent.
    
    Returns
    -------
    filtered_string_list: list of str
        The filtered list of strings.
    """
    
    filtered_string_list = []
    for s in string_list:
        if must and string_must in s:
            filtered_string_list.append(s)
        elif not must and string_must not in s:
            filtered_string_list.append(s)
    
    return filtered_string_list

#%%

def filter_to_only_directories(string_list):

    """Filters list of strings by returning only directories.
    
    Parameters
    ----------
    string_list : list of str
        The list of strings to filter.
    
    Returns
    -------
    filtered_string_list: list of str
        The filtered list of strings.
    """
    
    filtered_string_list = []
    for s in string_list:
        if '.' not in s:
            filtered_string_list.append(s)
    
    return filtered_string_list

#%%

def sort_by_number(string_list, number_index=0):

    """Sorts list of strings by a variable number of recurrent position.
    
    Parameters
    ----------
    string_list : list of str
        The list of strings to order.
    number_index=0 : int, optional
        The index of the recurrent number inside the expression (0 would be 
        the 1st number, 1 the 2nd and so on)
    
    Returns
    -------
    sorted_string_list: list of str
        The ordered list of strings.
    """
    
    numbers = [find_numbers(s)[number_index] for s in string_list]
    index = np.argsort(numbers)
    sorted_string_list = [string_list[i] for i in index]
    
    return sorted_string_list


#%%

def fix_params_dict(faulty_params):

    """Fixes the faulty params dict caused by wlen_range np.array on vs.savetxt
    
    Parameters
    ----------
    faulty_params : str
        The faulty params dict wrongly expressed as a string.
    
    Returns
    -------
    fixed_params : dict
        The fixed params dict correctly expressed as a dict.
    """
    
    problem = faulty_params.split("wlen_range=")[1].split(", nfreq")[0]
    solved = str(find_numbers(problem))
    fixed_params = solved.join(faulty_params.split(problem))
    fixed_params = string_to_dict(fixed_params)

    return fixed_params