# -*- coding: utf-8 -*-
"""
This module contains miscellaneous tools appliable to a variaty of tasks.

It could be divided into 4 main sections:

    (1) transforming to and from string other classes 
    (`nparray_to_string`, `dict_to_string`, `find_numbers`, etc)
    (2) useful custom-made classes to manage data
    (`WrapperDict`, `DottableWrapper`, `NumpyArrayParamType`)
    (3) treating lists of strings (`filter_by_string_must`, `sort_by_number`) 
    and dicts of strings (`join_strings_dict`)
    (4) fixing bugs in my thesis' data due to code mistakes
    
Some of its most useful tools are...

find_numbers : function
    Returns a list of numbers found on a given string
sort_by_number : function
    Sorts list of strings by a variable number of recurrent position.
WrapperList : class
    A list subclass that applies methods to a list of instances.
WrapperDict : class
    A dict subclass that applies methods to a dict of instances.
DottableWrapper : class
    Example of a class which allows dot calling instances.
fix_params_dict : function
    Fixes the faulty params dict caused by wlen_range np.array on vs.savetxt

@author: Vall
"""

try:
    import click as cli
except:
    print("Click console line module arguments classes not available. " + 
          "Don't worry! You can still use all other features of this module.")
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

class WrapperList(list):
    
    """A list subclass that applies methods to a list of instances.
    
    Examples
    --------
    >> class MyClass:
    
        def __init__(self, value=10):
            self.sub_prop = value
            self._prop = value
        
        @property
        def prop(self):
            return self._prop
        
        @prop.setter
        def prop(self, value):
            self._prop = value
            
        def sub_method(self, item):
            return item * self.sub_prop
        
        def method(self, item):
            return item * self.prop
        
    >> Z = WrapperList([MyClass(), MyClass(2)])
    >> Z.prop
    [10, 2]
    >> Z._prop
    [10, 2]
    >> Z.sub_prop
    [10, 2]
    >> Z.method(2)
    [20, 4]
    >> Z.prop = 3
    >> Z.prop
    [3, 3]
    >> Z.append(MyClass(1))
    >> Z.prop
    [3, 3, 1]
    >> Z.prop = [10, 2, 1]
    >> Z.prop
    [10, 2, 1]
    
    Warnings
    --------
    >> Z.prop = [2, 3]
    >> Z.prop
    [[2,3], [2,3], [2,3]]
    
    """

    def __init__(self, iterable=[]):
        
        super().__init__(iterable)

    def __transf_methods__(self, methods_list):
        
        def function(*args, **kwargs):
            results = [m(*args, **kwargs) for m in methods_list]
            return results
        
        return function

    def __getattr__(self, name):
        
        if name in dir(self):
        
            super().__getattribute__(name)
        
        else:
            
            result = []
            for ins in self:
                result.append(ins.__getattribute__(name))
            if callable(result[0]):
                result = self.__transf_methods__(result)
            return result
    
    def __setattr__(self, name, value):
        
        if isinstance(value, list) and len(value) == len(self):
            for ins, v in zip(self, value):
                ins.__setattr__(name, v)
        else:
            for ins in self:
                ins.__setattr__(name, value)

#%%

class WrapperDict(dict):
    
    """A dict subclass that applies methods to a dict of instances.
    
    Examples
    --------
    >> class MyClass:
    
        def __init__(self, value=10):
            self.sub_prop = value
            self._prop = value
        
        @property
        def prop(self):
            return self._prop
        
        @prop.setter
        def prop(self, value):
            self._prop = value
            
        def sub_method(self, item):
            return item * self.sub_prop
        
        def method(self, item):
            return item * self.prop
        
    >> Z = WrapperDict(a=MyClass(), b=MyClass(2))
    >> Z.prop
    {'a': 10, 'b': 2}
    >> Z.update(c=MyClass(1))
    >> Z.prop
    {'a': 10, 'b': 2, 'c': 1}
    >> Z.method(2)
    {'a': 20, 'b': 4, 'c': 2}
    >> Z.prop = 3
    >> Z.prop
    {'a': 3, 'b': 3, 'c': 3}
    >> Z.prop = {'a': 10, 'b': 2, 'c': 1}
    >> Z.prop
    {'a': 10, 'b': 2, 'c': 1}
    
    """

    def __init__(self, **elements):
        
        super().__init__(**elements)
    
    def __transf_methods__(self, methods_dic):
        
        def function(*args, **kwargs):
            results = {}
            for key, method in methods_dic.items():
                results.update({key: method(*args, **kwargs)})
            return results
        
        return function
    
    def __getattr__(self, name):
        
        if name in dir(self):
            super().__getattribute__(name)        
        else:
            result = {n : ins.__getattribute__(name) 
                      for n, ins in self.items()}
            if callable(list(result.values())[0]):
                result = self.__transf_methods__(result)
            return result
    
    def __setattr__(self, name, value):
        
        if isinstance(value, dict):
            for ins, v in zip(self.values(), value.values()):
                ins.__setattr__(name, v)
        else:
            for ins in self.values():
                ins.__setattr__(name, value)

#%%

class DottableWrapper:
    
    """Example of a class which allows dot calling instances.
    
    Examples
    --------
    >> class MyClass:
    
        def __init__(self, value=10):
            self.sub_prop = value
            self._prop = value
        
        @property
        def prop(self):
            return self._prop
        
        @prop.setter
        def prop(self, value):
            self._prop = value
            
        def sub_method(self, item):
            return item * self.sub_prop
        
        def method(self, item):
            return item * self.prop
        
    >> Z = DottableWrapper(a=MyClass(), b=MyClass(2))
    >>
    >> # Let's check dot calling
    >> Z.a.sub_prop 
    10
    >> Z.a.sub_prop = 1
    >> Z.a.sub_prop
    1
    >>
    >> # Let's check dot calling all instances at once
    >> Z.all.sub_prop
    {'a': 1, 'b': 2}
    >> Z.all.sub_prop = 3
    >> Z.all.sub_prop
    {'a': 3, 'b': 3}
    >> Z.all.sub_prop = {'a': 1}
    >> Z.all.sub_prop
    {'a': 1, 'b': 3}
    >> 
    >> # This also works with methods
    >> Z.a.prop
    10
    >> Z.a.method(2)
    20
    >> Z.all.prop
    {'a': 10, 'b': 2}
    >> Z.all.method(2)
    {'a': 20, 'b': 4}
    >>
    >> # This is an updatable class too
    >> Z.add(c=MyClass(4))
    >> Z.all.sub_prop
    {'a': 1, 'b': 3, 'c': 4}
    >> Z.c.sub_prop
    4
    
    """
    
    def __init__(self, **instances):
        
        self.all = WrapperDict()
        self.add(**instances)

    # def __print__(self):
        
    #     self.all.__print__()
        
    # def __repr__(self):
        
    #     self.all.__repr__()
    
    def add(self, **instances):
        
        instances = dict(instances)
        self.__dict__.update(instances)
        self.all.update(instances)

#%%

class NumpyArrayParamType(cli.ParamType):
    
    """A np.ndarray parameter class for Click console-line commands' arguments."""
    
    name = "nparray"

    def convert(self, value, param, ctx):
        try:
            a = eval(value)
            if type(a) is np.ndarray:
                return a
            elif type(a) is list:
                return np.array(a)
        except TypeError:
            self.fail(
                "expected string for np.array() conversion, got "
                f"{value!r} of type {type(value).__name__}",
                param,
                ctx,
            )
        except ValueError:
            self.fail(f"{value!r} is not a valid np.array", param, ctx)
            
NUMPY_ARRAY = NumpyArrayParamType()

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

def counting_sufix(number):
    
    """Returns a number's suffix string to use for counting.
    
    Parameters
    ----------
    number: int, float
        Any number, though it is designed to work with integers.
    
    Returns
    -------
    ans: str
        A string representing the integer number plus a suffix.
    
    Examples
    --------
    >> counting_sufix(1)
    '1st'
    >> counting_sufix(22)
    '22nd'
    >> counting_sufix(1.56)
    '2nd'
    
    """
    
    number = round(number)
    unit = int(str(number)[-1])
    
    if unit == 1:
        ans = 'st'
    elif unit == 2:
        ans = 'nd'
    elif unit == 3:
        ans = 'rd'
    else:
        ans = 'th'
    
    return ans

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

def enumerate_string(str_list, str_sep="and", str_sep_always=False):
    """Returns a one phrase enumeration from a list of strings.
    
    Parameters
    ----------
    str_list : list of str
        The list of strings to join into a one phrase enumeration.
    str_sep="and" : str, optional
        The final separator that isn't a comma. The default is "and".
    str_sep_always=False : bool, optional
        If true, the string separator is used between all strings and not just 
        the final two. The default is `False`.

    Returns
    -------
    answer : str
        The phrase enumeration as a unique string.

    """
    
    if str_sep_always:
        answer = f" {str_sep} ".join(str_list)
    else:
        answer = ", ".join(str_list[:-1])
        answer += f" {str_sep} " + str_list[-1]

    return answer

#%%

def join_strings_dict(str_dict, str_joiner="for"):
    """
    Returns a list of strings joining key and value from a strings dictionary.
    
    Parameters
    ----------
    str_dict : dict of str
        Dictionary of strings. Both key and values must be strings.
    str_joiner="for" : str
        Joiner for formatting each pair of key-value as 'key str_joiner value'.
    
    Returns
    -------
    str_list : list of str
        List of strings already formatted to join each pair of key-value.
    """

    str_list = []
    for k, v in str_dict.items():
        str_list.append( f"'{k}' for {v}")
    return str_list

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
