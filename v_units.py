#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The 'v_units' module contains functions to manage units in Meep.

@author: Vall
"""

from numpy import pi
from v_class import DottableWrapper

class MeepUnitsManager:
    
    def __init__(self, from_um_factor=1):
        
        self._from_um_factor = from_um_factor
        self._a = from_um_factor * 1e-6 # Meep length unit [m]
        
        self.constants = DottableWrapper(**dict(
            c = 299792458 , # Speed of light in vacuum c [m/s]
            e = 1.6021892 * 10e-19 , # Electron charge e [C]
            me = 9.109534 * 10e-31 , # Electron rest mass [kg]
            mp = 1.6726485 * 10e-27 # Proton rest mass [kg]
            ))
        
        self.constants.add(**dict(
            # Vacuum Permitivity ε0 [F/m]
            epsilon0 = 1/(4*pi*self.constants.c**2) * 10e7 , 
            # Vacuum Permeability μ0 [H/m])
            mu0 = 4*pi * 10e-7 
            ))
        
        self.Meep_to_SI = DottableWrapper()
        
        self.SI_to_Meep = DottableWrapper()
    
    @property
    def from_um_factor(self):
        """Conversion factor from um to Meep getter"""
        return self._from_um_factor
    
    @from_um_factor.setter
    def from_um_factor(self, value):
        """Conversion factor from um to Meep setter (also updates a)"""
        self._from_um_factor = value
        self._a = value * 1e-6 # Meep length unit [m]
    
    @property
    def a(self):
        """Length unit a getter"""
        return self._a
    
    @a.setter
    def a(self, value):
        """Length unit a setter (also updates from_um_factor)"""
        self._a = value
        self._from_um_factor = value * 1e6
        
    def len_Meep_to_SI(self, len_Meep):
        """Converts Meep length to SI units [m]"""
        
        return self.a * len_Meep
    
    def len_SI_to_Meep(self, len_SI):
        """Converts SI length [m] to Meep units"""
        
        return len_SI / self.a
    
    def time_Meep_to_SI(self, time_Meep):
        """Converts Meep time to SI units [s]"""
        
        return time_Meep * self.a / self.constants.c
    
    def time_SI_to_Meep(self, time_SI):
        """Converts SI time [s] to Meep units"""
        
        return time_SI * self.constants.c / self.a
    
    def vel_Meep_to_SI(self, vel_Meep):
        """Converts Meep velocity to SI units [m/s]"""
        
        return vel_Meep * self.constants.c
    
    def vel_SI_to_Meep(self, vel_SI):
        """Converts SI velocity [m/s] to Meep units"""
        
        return vel_SI / self.constants.c
    
    def freq_Meep_to_SI(self, freq_Meep):
        """Converts Meep frequency to SI units [Hz]"""
        
        return freq_Meep * self.constants.c / self.a
    
    def freq_SI_to_Meep(self, freq_SI):
        """Converts SI frequency [Hz] to Meep units"""
        
        return freq_SI * self.a / self.constants.c