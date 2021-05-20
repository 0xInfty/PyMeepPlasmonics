#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The `v_meep` module contains functions used to complement Meep.

Some of its most useful tools are...

import_module : function
    Returns Medium instance from string with specified length scale
verify_stability_dim_index : function
    Verifies stability via dimensions, refractive index and Courant factor.
MeepUnitsManager : class
    Incomplete class to manage units in Meep.

It's widely based on Meep Materials Library.

@author: vall
"""

import meep as mp
import numpy as np
import v_utilities as vu

#%%

def import_medium(name, from_um_factor=1, paper="R"):
    
    """Returns Medium instance from string with specified length scale
    
    It's widely based on Meep Materials Library, merely adapted to change 
    according to the length unit scale used.
    
    Parameters
    ----------
    name: str
        Name of the desired material.
    from_um_factor=1 : int, float, optional
        Factor to transform from SI μm to the chosen length unit. For example, 
        to work with 100 nm as 1 Meep unit, from_um_factor=.1 must be specified.
    paper="R": str
        Name of desired source for experimental input data of medium.
    
    Returns
    -------
    medium : mp.Medium
        The mp.Medium instance of the desired material.
    
    Raises
    ------
    "No media called ... is available" : SyntaxError
        If no material is found whose name matches the string given.
    """
    
    if "Ag" not in name and "Au" not in name:
        raise SyntaxError("No media called {} is available".format(name))
        return
           
    # Default unit length is 1 um
    eV_from_um_factor = from_um_factor/1.23984193 # Conversion factor: eV to 1/um [=1/hc]
    # from_um_factor used to be um_scale (but that's a shady name)
    
    ############ GOLD #################################################
    
    if name=="Au" and paper=="R":
        
    #------------------------------------------------------------------
    # Elemental metals from A.D. Rakic et al., Applied Optics, Vol. 37, No. 22, pp. 5271-83, 1998
    # Wavelength range: 0.2 - 12.4 um
    # Gold (Au)

        metal_range = mp.FreqRange(min=from_um_factor*1e3/6199.2, 
                                   max=from_um_factor*1e3/247.97)
        
        Au_plasma_frq = 9.03*eV_from_um_factor
        Au_f0 = 0.760
        Au_frq0 = 1e-10
        Au_gam0 = 0.053*eV_from_um_factor
        Au_sig0 = Au_f0*Au_plasma_frq**2/Au_frq0**2
        Au_f1 = 0.024
        Au_frq1 = 0.415*eV_from_um_factor      # 2.988 um
        Au_gam1 = 0.241*eV_from_um_factor
        Au_sig1 = Au_f1*Au_plasma_frq**2/Au_frq1**2
        Au_f2 = 0.010
        Au_frq2 = 0.830*eV_from_um_factor      # 1.494 um
        Au_gam2 = 0.345*eV_from_um_factor
        Au_sig2 = Au_f2*Au_plasma_frq**2/Au_frq2**2
        Au_f3 = 0.071
        Au_frq3 = 2.969*eV_from_um_factor      # 0.418 um
        Au_gam3 = 0.870*eV_from_um_factor
        Au_sig3 = Au_f3*Au_plasma_frq**2/Au_frq3**2
        Au_f4 = 0.601
        Au_frq4 = 4.304*eV_from_um_factor      # 0.288 um
        Au_gam4 = 2.494*eV_from_um_factor
        Au_sig4 = Au_f4*Au_plasma_frq**2/Au_frq4**2
        Au_f5 = 4.384
        Au_frq5 = 13.32*eV_from_um_factor      # 0.093 um
        Au_gam5 = 2.214*eV_from_um_factor
        Au_sig5 = Au_f5*Au_plasma_frq**2/Au_frq5**2
        
        Au_susc = [mp.DrudeSusceptibility(frequency=Au_frq0, gamma=Au_gam0, sigma=Au_sig0),
                   mp.LorentzianSusceptibility(frequency=Au_frq1, gamma=Au_gam1, sigma=Au_sig1),
                   mp.LorentzianSusceptibility(frequency=Au_frq2, gamma=Au_gam2, sigma=Au_sig2),
                   mp.LorentzianSusceptibility(frequency=Au_frq3, gamma=Au_gam3, sigma=Au_sig3),
                   mp.LorentzianSusceptibility(frequency=Au_frq4, gamma=Au_gam4, sigma=Au_sig4),
                   mp.LorentzianSusceptibility(frequency=Au_frq5, gamma=Au_gam5, sigma=Au_sig5)]
        
        Au = mp.Medium(epsilon=1.0, E_susceptibilities=Au_susc, 
                       valid_freq_range=metal_range)
        
        return Au

    elif name=="Au" and paper=="JC":
        
    #------------------------------------------------------------------
    # Metals from D. Barchiesi and T. Grosges, J. Nanophotonics, Vol. 8, 08996, 2015
    # Wavelength range: 0.4 - 0.8 um
    # Gold (Au)
    # Fit to P.B. Johnson and R.W. Christy, Physical Review B, Vol. 6, pp. 4370-9, 1972
        
        metal_visible_range = mp.FreqRange(min=from_um_factor*1e3/800, 
                                           max=from_um_factor*1e3/400)
    
        Au_JC_visible_frq0 = 1*from_um_factor/(0.139779231751333)
        Au_JC_visible_gam0 = 1*from_um_factor/(26.1269913352870)
        Au_JC_visible_sig0 = 1
        
        Au_JC_visible_frq1 = 1*from_um_factor/(0.404064525036786)
        Au_JC_visible_gam1 = 1*from_um_factor/(1.12834046202759)
        Au_JC_visible_sig1 = 2.07118534879440
        
        Au_JC_visible_susc = [mp.DrudeSusceptibility(frequency=Au_JC_visible_frq0, gamma=Au_JC_visible_gam0, sigma=Au_JC_visible_sig0),
                              mp.LorentzianSusceptibility(frequency=Au_JC_visible_frq1, gamma=Au_JC_visible_gam1, sigma=Au_JC_visible_sig1)]
        
        Au_JC_visible = mp.Medium(epsilon=6.1599, 
                                  E_susceptibilities=Au_JC_visible_susc, 
                                  valid_freq_range=metal_visible_range)
        
        return Au_JC_visible

    elif name=="Au" and paper=="P":
        
    #------------------------------------------------------------------
    # Metals from D. Barchiesi and T. Grosges, J. Nanophotonics, Vol. 8, 08996, 2015
    # Wavelength range: 0.4 - 0.8 um
    # Gold (Au)
    # Fit to E.D. Palik, Handbook of Optical Constants, Academic Press, 1985 

        metal_visible_range = mp.FreqRange(min=from_um_factor*1e3/800, 
                                           max=from_um_factor*1e3/400)
        
        Au_visible_frq0 = 1/(0.0473629248511456*from_um_factor)
        Au_visible_gam0 = 1/(0.255476199605166*from_um_factor)
        Au_visible_sig0 = 1
        
        Au_visible_frq1 = 1/(0.800619321082804*from_um_factor)
        Au_visible_gam1 = 1/(0.381870287531951*from_um_factor)
        Au_visible_sig1 = -169.060953137985
        
        Au_visible_susc = [mp.DrudeSusceptibility(frequency=Au_visible_frq0, gamma=Au_visible_gam0, sigma=Au_visible_sig0),
                           mp.LorentzianSusceptibility(frequency=Au_visible_frq1, gamma=Au_visible_gam1, sigma=Au_visible_sig1)]
        
        Au_visible = mp.Medium(epsilon=0.6888, E_susceptibilities=Au_visible_susc, 
                               valid_freq_range=metal_visible_range)
        
        return Au_visible
    
    elif name=="Au":
        raise ValueError("No source found for Au with that name")

    ############ SILVER ###############################################

    if name=="Ag" and paper=="R":
        
    #------------------------------------------------------------------
    # Elemental metals from A.D. Rakic et al., Applied Optics, Vol. 37, No. 22, pp. 5271-83, 1998
    # Wavelength range: 0.2 - 12.4 um
    # Silver (Ag)
        
        metal_range = mp.FreqRange(min=from_um_factor*1e3/12398, 
                                   max=from_um_factor*1e3/247.97)      
        
        Ag_plasma_frq = 9.01*eV_from_um_factor
        Ag_f0 = 0.845
        Ag_frq0 = 1e-10
        Ag_gam0 = 0.048*eV_from_um_factor
        Ag_sig0 = Ag_f0*Ag_plasma_frq**2/Ag_frq0**2
        Ag_f1 = 0.065
        Ag_frq1 = 0.816*eV_from_um_factor      # 1.519 um
        Ag_gam1 = 3.886*eV_from_um_factor
        Ag_sig1 = Ag_f1*Ag_plasma_frq**2/Ag_frq1**2
        Ag_f2 = 0.124
        Ag_frq2 = 4.481*eV_from_um_factor      # 0.273 um
        Ag_gam2 = 0.452*eV_from_um_factor
        Ag_sig2 = Ag_f2*Ag_plasma_frq**2/Ag_frq2**2
        Ag_f3 = 0.011
        Ag_frq3 = 8.185*eV_from_um_factor      # 0.152 um
        Ag_gam3 = 0.065*eV_from_um_factor
        Ag_sig3 = Ag_f3*Ag_plasma_frq**2/Ag_frq3**2
        Ag_f4 = 0.840
        Ag_frq4 = 9.083*eV_from_um_factor      # 0.137 um
        Ag_gam4 = 0.916*eV_from_um_factor
        Ag_sig4 = Ag_f4*Ag_plasma_frq**2/Ag_frq4**2
        Ag_f5 = 5.646
        Ag_frq5 = 20.29*eV_from_um_factor      # 0.061 um
        Ag_gam5 = 2.419*eV_from_um_factor
        Ag_sig5 = Ag_f5*Ag_plasma_frq**2/Ag_frq5**2
        
        Ag_susc = [mp.DrudeSusceptibility(frequency=Ag_frq0, gamma=Ag_gam0, sigma=Ag_sig0),
                   mp.LorentzianSusceptibility(frequency=Ag_frq1, gamma=Ag_gam1, sigma=Ag_sig1),
                   mp.LorentzianSusceptibility(frequency=Ag_frq2, gamma=Ag_gam2, sigma=Ag_sig2),
                   mp.LorentzianSusceptibility(frequency=Ag_frq3, gamma=Ag_gam3, sigma=Ag_sig3),
                   mp.LorentzianSusceptibility(frequency=Ag_frq4, gamma=Ag_gam4, sigma=Ag_sig4),
                   mp.LorentzianSusceptibility(frequency=Ag_frq5, gamma=Ag_gam5, sigma=Ag_sig5)]
        
        Ag = mp.Medium(epsilon=1.0, E_susceptibilities=Ag_susc, 
                       valid_freq_range=metal_range)
        
        return Ag
    
    elif name=="Ag" and paper=="P":

    #------------------------------------------------------------------
    # Metals from D. Barchiesi and T. Grosges, J. Nanophotonics, Vol. 8, 08996, 2015
    # Wavelength range: 0.4 - 0.8 um
    ## WARNING: unstable; field divergence may occur
    # Silver (Au)
    # Fit to E.D. Palik, Handbook of Optical Constants, Academic Press, 1985 

        metal_visible_range = mp.FreqRange(min=from_um_factor*1e3/800, 
                                           max=from_um_factor*1e3/400)
        
        Ag_visible_frq0 = 1*from_um_factor/(0.142050162130618)
        Ag_visible_gam0 = 1*from_um_factor/(18.0357292925015)
        Ag_visible_sig0 = 1
        
        Ag_visible_frq1 = 1*from_um_factor/(0.115692151792108)
        Ag_visible_gam1 = 1*from_um_factor/(0.257794324096575)
        Ag_visible_sig1 = 3.74465275944019
        
        Ag_visible_susc = [mp.DrudeSusceptibility(frequency=Ag_visible_frq0, gamma=Ag_visible_gam0, sigma=Ag_visible_sig0),
                           mp.LorentzianSusceptibility(frequency=Ag_visible_frq1, gamma=Ag_visible_gam1, sigma=Ag_visible_sig1)]
        
        Ag_visible = mp.Medium(epsilon=0.0067526, 
                               E_susceptibilities=Ag_visible_susc, 
                               valid_freq_range=metal_visible_range)
        
        return Ag_visible

    elif name=="Ag":    
        raise ValueError("No source found for Ag with that name")
    
    else:
        raise ValueError("No source found for that material")

#%%

def verify_stability_freq_res(medium, resolution, courant=0.5):
    """Verifies stability via temporal resolution and resonant frequencies.
    
    Parameters
    ----------
    medium : mp.Medium
        The mp.Medium instance of the material.
    resolution : int
        The resolution that defines spatial discretization dx = 1/resolution 
        in Meep units.
    Courant=0.5 : float
        The Courant factor that defines temporal discretization dt = Courant*dx 
        in Meep units.
    
    Returns
    -------
    stable : bool
        True if the simulation turns out stable for that medium.        
    """
    
    resonant_frequencies = [Es.frequency for Es in medium.E_susceptibilities]
    dt = courant/resolution
    stable = True
    error = []
    for i, f in enumerate(resonant_frequencies):
        if f >= 1 / (np.pi * dt): 
            stable = False
            error.append(i)
    if stable:
        answer = "Medium is stable."
        answer += " All resonant frequencies are small enough for this resolution."
        print(answer)
    else:
        answer = [str(i) + vu.counting_sufix(i) for i in error]
        if len(error)>1: 
            answer = vu.enumerate_string(answer) + " frequencies are"
        else:
            answer = answer + " frequency is"
        print(f"Medium is not stable: {answer} too large.")
    return stable

#%%

def verify_stability_dim_index(medium, freq, ndims=3, courant=0.5):
    """Verifies stability via dimensions, refractive index and Courant factor.
    
    Parameters
    ----------
    medium : The mp.Medium instance of the material.
        The mp.Medium instance of the material.
    freq : float, array of floats
        Frequency in Meep units.
    ndims=3 : int, optional
        Number of dimensions of simulation.
    courant=0.5 : float, optional
        Courant factor that defines temporal discretization from spatial 
        discretization as dt = Courant * dx.

    Returns
    -------
    stable : bool
        True if the simulation turns out to be stable for that medium.
    """
    
    try:
        freq = [*freq]
    except:
        freq = [freq]
    
    index = np.array([ np.sqrt(medium.epsilon(f)[0,0]*medium.mu(f)[0,0]) for f in freq ])
    
    min_index = np.min(np.real(index))
    
    stable = ( courant < min_index / np.sqrt(ndims) )
    
    return stable

#%%

def max_stable_courant_freq_res(medium, resolution):
    """Maximum stable Courant via temporal resolution and resonant frequencies.
    
    Parameters
    ----------
    medium : mp.Medium
        The mp.Medium instance of the material.
    resolution : int
        The resolution that defines spatial discretization dx = 1/resolution 
        in Meep units.
    
    Returns
    -------
    max_courant : float
        Maximum value of Courant factor for the FDTD method to be stable.
    """
    
    resonant_frequencies = [Es.frequency for Es in medium.E_susceptibilities]
    max_courant = resolution / (np.pi * np.max(resonant_frequencies))
    
    return max_courant

#%%

def max_stable_courant_dim_index(medium, freq, ndims=3):
    """Maximum stable Courant via dimensions and refractive index condition.
    
    Parameters
    ----------
    medium : The mp.Medium instance of the material.
        The mp.Medium instance of the material.
    freq : float, array of floats
        Frequency in Meep units.
    ndims=3 : int, optional
        Number of dimensions of simulation.
    method="abs" : str, optional
        Method applied to epsilon * mu product to obtain refractive index.

    Returns
    -------
    max_courant : float
        Maximum value of Courant factor for the FDTD method to be stable.
    """
    
    try:
        freq = [*freq]
    except:
        freq = [freq]
    
    index = np.array([ np.sqrt(medium.epsilon(f)[0,0]*medium.mu(f)[0,0]) for f in freq ])
    
    min_index = np.min(np.real(index))
    
    max_courant = min_index / np.sqrt(ndims)
    
    return max_courant

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

#%%

class MeepUnitsManager:
    """Depricated class to manage units in Meep"""
    
    def __init__(self, from_um_factor=1):
        
        self._from_um_factor = from_um_factor
        self._a = from_um_factor * 1e-6 # Meep length unit [m]
        
        self.constants = vu.DottableWrapper(**dict(
            c = 299792458 , # Speed of light in vacuum c [m/s]
            e = 1.6021892 * 10e-19 , # Electron charge e [C]
            me = 9.109534 * 10e-31 , # Electron rest mass [kg]
            mp = 1.6726485 * 10e-27 # Proton rest mass [kg]
            ))
        
        self.constants.add(**dict(
            # Vacuum Permitivity ε0 [F/m]
            epsilon0 = 1/(4*np.pi*self.constants.c**2) * 10e7 , 
            # Vacuum Permeability μ0 [H/m])
            mu0 = 4*np.pi * 10e-7 
            ))
        
        self.Meep_to_SI = vu.DottableWrapper()
        
        self.SI_to_Meep = vu.DottableWrapper()
    
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
