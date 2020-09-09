# -*- coding: utf-8 -*-
"""
This module holds classes developed by our group.
@author: Usuario
"""

#%%       

class DynamicList(list):
    
    """Subclass that initializes a callable list
    
    Examples
    --------
    >>> a = DynamicList([1,2,3])
    >>> a(0, 1)
    [1,2]
    >> a(0,2)
    [1,3]
    >>> a.append(4)
    >>> a
    [1,2,3,4]
    
    """
    
    def __init__(self, iterable=[]):
        super().__init__(iterable)
    
    def __call__(self, *index):
        
        if len(index) == 1:
            return self[index[0]]
        
        else:
            return [self[i] for i in index]

class DynamicDict(dict):
    
    """Subclass that initializes a callable dictionary.
    
    Examples
    --------
    >>> a = DynamicDict()
    >>> a.update({'color': 'blue', 'age': 22})
    >>> a
    {'age': 22, 'color': 'blue'}
    >>> a('age', 'color')
    [22, 'blue']
    
    """
    
    def __init__(self, **elements):
        
        super().__init__(**elements)
        return
    
    def __call__(self, *key):

        if len(key) == 1:
            return self[key[0]]
        
        else:
            return [self[k] for k in key]
    
    def is_empty(self, key=None):
        
        if key is None:
            return not bool(self)
        elif key in self.keys():
            return False
        else:
            return True  

#%%

class InstancesDict:
    
    """Example of a class that holds a callable dictionary of instances.
    Examples
    --------
    >> class MyClass:
        def __init__(self, value=10):
            self.sub_prop = value
    >> instance_a, instance_b = MyClass(), MyClass(12)
    >> instance_c = ClassWithInstances(dict(a=instance_a,
                                            b=instance_b))
    >> Z = InstancesDict({1: instance_a,
                          2: instance_b,
                          3: instance_c})
    >> Z(1)
    <__main__.MyClass at 0x2e5572a2dd8>
    >> Z(1).sub_prop
    10
    >> Z(1).sub_prop = 30
    >> Z(1).sub_prop
    >> Z(3).b.sub_prop
    12
    >> Z(1,2)
    [<__main__.MyClass at 0x2e5573cfb00>, 
    <__main__.MyClass at 0x2e5573cf160>]
    
    Warnings
    --------
    'Z(1,2).prop' can't be done.
    
    """
    
    
    def __init__(self, dic):#, methods):
        
        self.__dict__.update(dic)
    
    def __call__(self, *key):

        if len(key) == 1:
            return self.__dict__[key[0]]
        
        else:
            return [self.__dict__[k] for k in key]
                
    def update(self, dic):
        
        self.__dict__.update(dic)
    
    def is_empty(self, key):
        
        if key in self.__dict__.keys():
            return False
        else:
            return True

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
    
    def add(self, **instances):
        
        instances = dict(instances)
        self.__dict__.update(instances)
        self.all.update(instances)

class DottableMultiWrapper():
    
    """A class that holds several dot-callable dict of instances.
    
    Examples
    --------
    >> class ItemClass:
    
        def __init__(self, sumando=10):
            self.sumando = sumando
        
        def sum(self, item):
            return item + self.sumando
    
    >> class OtherItemClass:
        def __init__(self, restando=10):
            self.restando = restando
        
        def rest(self, item):
            return item - self.restando
        
    >> Z = DottableMultiWrapper(a=ItemClass(), b=ItemClass(2))
    >>
    >> # Let's check dot calling
    >> Z.a.sumando
    10
    >> Z.a.sumando = 1
    >> Z.a.sumando
    1
    >>
    >> # Let's check dot calling all instances at once
    >> Z.all.sumando
    {'a': 1, 'b': 2}
    >> Z.all.sumando = 3
    >> Z.all.sumando
    {'a': 3, 'b': 3}
    >> Z.all.sumando = {'a': 1}
    >> Z.all.sumando
    {'a': 1, 'b': 3}
    >> 
    >> # This also works with methods
    >> Z.all.sumando
    1
    >> Z.all.sum(2)
    3
    >> Z.all.sumando
    {'a': 10, 'b': 3}
    >> Z.all.sum(2)
    {'a': 3, 'b': 5}
    >>
    >> # This is an updatable class too
    >> Z.add(c=ItemClass(4))
    >> Z.all.sumando
    {'a': 1, 'b': 3, 'c': 4}
    >> Z.all.sumando
    4
    >>
    >> # This class allows to keep adding other instances
    >> Z.add(alias='r', separator='',
             **{'1': OtherItemClass(1), '2': OtherItemClass(2)})
    >> Z.r.rest(5)
    {'1': 4, '2': 3}
    
    """
    
    def __init__(self, alias='all', separator=None, **instances):
        
        if instances != {}:
            self.add(alias, separator, **instances)
    
    def add(self, alias='all', separator=None, **instances):
        
        try:
            eval('self.{}'.format(alias))
        except:
            self.__setattr__(alias, WrapperDict())
        eval('self.{}.update(instances)'.format(alias))
        
        if separator is not None:
            instances = {alias+separator+k: v 
                         for k, v in instances.items()}
            
        instances = dict(instances)
        self.__dict__.update(instances)
    
#%%

class BigClass:
    
        def __init__(self, value=10):
            self.multiplier = value
            
        def multiply(self, item):
            return item * self.multiplier

class Wrapper(BigClass):
    
    """Subclass which holds a dot-callable dict of instances.
    
    Examples
    --------
    >> class ItemClass:
    
        def __init__(self, sumando=10):
            self.sumando = sumando
        
        def sum(self, item):
            return item + self.sumando
        
    >> Z = Wrapper()
    >> 
    >> # Let's check it works like BigClass
    >> Z.multiplier
    10
    >> Z.multiply(3)
    30
    >>
    >> # Now let's add instances of another class
    >> Z.add(a=ItemClass(), b=ItemClass(2))
    >>
    >> # Let's check dot calling
    >> Z.a.sumando
    10
    >> Z.a.sumando = 1
    >> Z.a.sumando
    1
    >>
    >> # Let's check dot calling all instances at once
    >> Z.all.sumando
    {'a': 1, 'b': 2}
    >> Z.all.sumando = 3
    >> Z.all.sumando
    {'a': 3, 'b': 3}
    >> Z.all.sumando = {'sa': 1}
    >> Z.all.sumando
    {'a': 1, 'b': 3}
    >> 
    >> # This also works with methods
    >> Z.a.sumando
    1
    >> Z.a.sum(2)
    3
    >> Z.all.sumando
    {'a': 10, 'b': 3}
    >> Z.all.sum(2)
    {'a': 3, 'b': 5}
    >>
    >> # This is an updatable class too
    >> Z.add(c=ItemClass(4))
    >> Z.all.sumando
    {'a': 1, 'b': 3, 'c': 4}
    >> Z.c.sumando
    4
    
    """

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.all = WrapperDict()
    
    def add(self, **instances):
        
        instances = dict(instances)
        self.__dict__.update(instances)
        self.all.update(instances)

class MultiWrapper(BigClass):
    
    """Subclass which holds several dot-callable dict of instances.
    
    Examples
    --------
    >> class ItemClass:
    
        def __init__(self, sumando=10):
            self.sumando = sumando
        
        def sum(self, item):
            return item + self.sumando
    
    >> class OtherItemClass:
        def __init__(self, restando=10):
            self.restando = restando
        
        def rest(self, item):
            return item - self.restando
        
    >> Z = MultiWrapper()
    >> 
    >> # Let's check it works like BigClass
    >> Z.multiplier
    10
    >> Z.multiply(3)
    30
    >>
    >> # Now let's add instances of another class
    >> Z.add('s', a=ItemClass(), b=ItemClass(2))
    >>
    >> # Let's check dot calling
    >> Z.sa.sumando
    10
    >> Z.sa.sumando = 1
    >> Z.sa.sumando
    1
    >>
    >> # Let's check dot calling all instances at once
    >> Z.s.sumando
    {'a': 1, 'b': 2}
    >> Z.s.sumando = 3
    >> Z.s.sumando
    {'a': 3, 'b': 3}
    >> Z.s.sumando = {'sa': 1}
    >> Z.s.sumando
    {'a': 1, 'b': 3}
    >> 
    >> # This also works with methods
    >> Z.sa.sumando
    1
    >> Z.sa.sum(2)
    3
    >> Z.s.sumando
    {'a': 10, 'b': 3}
    >> Z.s.sum(2)
    {'a': 3, 'b': 5}
    >>
    >> # This is an updatable class too
    >> Z.add('s', sc=ItemClass(4))
    >> Z.s.sumando
    {'sa': 1, 'sb': 3, 'sc': 4}
    >> Z.sc.sumando
    4
    >>
    >> # This class allows to keep adding other instances
    >> Z.add(alias='r', 
             **{'1': OtherItemClass(1), '2': OtherItemClass(2)})
    >> Z.r1.rest(5)
    {'ra': 4, 'rb': 3}
    
    """
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
    
    def add(self, alias='all', separator='', **instances):
        
        try:
            eval('self.{}'.format(alias))
        except:
            self.__setattr__(alias, WrapperDict())
        eval('self.{}.update(instances)'.format(alias))
        
        instances = {alias+separator+k: v for k, v in instances.items()}
            
        instances = dict(instances)
        self.__dict__.update(instances)