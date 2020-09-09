#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The 'vsClasses' module contains customed classes.

The only class contained right now is...
DottableWrapper : class
    Example of a class which allows dot calling instances.


@author: Vall
"""

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
    
    def add(self, **instances):
        
        instances = dict(instances)
        self.__dict__.update(instances)
        self.all.update(instances)