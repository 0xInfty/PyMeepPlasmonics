#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 17:06:46 2021

The 'v_click' module holds classes for Click console-line commands' arguments.


@author: Vall
"""

import click as cli
import numpy as np

#%%

class NumpyArrayParamType(cli.ParamType):
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