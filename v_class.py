#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The 'v_class' module contains customed classes.

WrapperList : class
    A list subclass that applies methods to a list of instances.
WrapperDict : class
    A dict subclass that applies methods to a dict of instances.
DottableWrapper : class
    Example of a class which allows dot calling instances.

@author: Vall
"""

import meep as mp

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

class Line(mp.Volume):
    
    """A Meep Volume subclass that holds a line instead of a whole volume"""

    def __init__(self, center=mp.Vector3(), size=mp.Vector3(), 
                 is_cylindrical=False, vertices=[]):
        
        super().__init__(center=center, size=size, dims=1, 
                         is_cylindrical=is_cylindrical, vertices=vertices)
        
        nvertices = len(self.get_vertices())
        if nvertices>2:
            raise TypeError(f"Must have 2 vertices and not {nvertices}")

#%%

class Plane(mp.Volume):
    
    """A Meep Volume subclass that holds a line instead of a whole volume"""

    def __init__(self, center=mp.Vector3(), size=mp.Vector3(), 
                 is_cylindrical=False, vertices=[]):
        
        super().__init__(center=center, size=size, dims=1, 
                         is_cylindrical=is_cylindrical, vertices=vertices)
        
        nvertices = len(self.get_vertices())
        if nvertices<3 or nvertices>4:
            raise TypeError(f"Must have 3 or 4 vertices and not {nvertices}")
