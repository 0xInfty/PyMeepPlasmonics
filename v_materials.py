#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The `v_materials` module contains tools for materials from different sources.

It holds tools to work with...
    
    - Meep's medium classes and functions.
    - Complex dielectric constant epsilon and complex refractive index N.
    
Some of its most useful tools are...    

import_module : function
    Returns Medium instance from string with specified length scale
epsilon_function : function
    Generates an interpolation function for epsilon from experimental data.

It's widely based on Meep Materials Library.

@author: vall
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import interp1d
try:
    import meep as mp
except:
    print("Meep functions not available. Don't worry! You can still use everything else")
import v_save as vs
import v_theory as vt
import v_utilities as vu

syshome = vs.get_sys_home()
home = vs.get_home()

#%% MEEP MEDIUM IMPORT

def import_medium(material, paper="R", from_um_factor=1):
    
    """Returns Medium instance from string with specified length scale
    
    It's widely based on Meep Materials Library, merely adapted to change 
    according to the length unit scale used.
    
    Parameters
    ----------
    material: str
        Name of the desired material.
    paper="R": str
        Name of desired source for experimental input data of medium.
    from_um_factor=1 : int, float, optional
        Factor to transform from SI μm to the chosen length unit. For example, 
        to work with 100 nm as 1 Meep unit, from_um_factor=.1 must be specified.
    
    Returns
    -------
    medium : mp.Medium
        The mp.Medium instance of the desired material.
    
    Raises
    ------
    "No media called ... is available" : SyntaxError
        If no material is found whose name matches the string given.
    """
    
    if "Ag" not in material and "Au" not in material:
        raise SyntaxError("No media called {} is available".format(material))
        return
           
    # Default unit length is 1 um
    eV_from_um_factor = from_um_factor/1.23984193 # Conversion factor: eV to 1/um [=1/hc]
    # from_um_factor used to be um_scale (but that's a shady name)
    
    ############ GOLD #################################################
    
    if material=="Au" and paper=="R":
        
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
        Au.from_um_factor = from_um_factor
        
        return Au

    elif material=="Au" and paper=="JC":
        
    #------------------------------------------------------------------
    # Metals from D. Barchiesi and T. Grosges, J. Nanophotonics, Vol. 8, 08996, 2015
    # Wavelength range: 0.4 - 0.8 um
    # Gold (Au)
    # Fit to P.B. Johnson and R.W. Christy, Physical Review B, Vol. 6, pp. 4370-9, 1972
        
        # metal_visible_range = mp.FreqRange(min=from_um_factor*1e3/800, 
        #                                    max=from_um_factor*1e3/400)
    
        # Au_JC_visible_frq0 = 1*from_um_factor/0.139779231751333
        # Au_JC_visible_gam0 = 1*from_um_factor/26.1269913352870
        # Au_JC_visible_sig0 = 1
        
        # Au_JC_visible_frq1 = 1*from_um_factor/0.404064525036786
        # Au_JC_visible_gam1 = 1*from_um_factor/1.12834046202759
        # Au_JC_visible_sig1 = 2.07118534879440
        
        # Au_JC_visible_susc = [mp.DrudeSusceptibility(frequency=Au_JC_visible_frq0, gamma=Au_JC_visible_gam0, sigma=Au_JC_visible_sig0),
        #                       mp.LorentzianSusceptibility(frequency=Au_JC_visible_frq1, gamma=Au_JC_visible_gam1, sigma=Au_JC_visible_sig1)]
        
        # Au_JC_visible = mp.Medium(epsilon=6.1599, 
        #                           E_susceptibilities=Au_JC_visible_susc, 
        #                           valid_freq_range=metal_visible_range)
        # Au_JC_visible.from_um_factor = from_um_factor
        
        # return Au_JC_visible
        
    #------------------------------------------------------------------
    # Metal from my own fit
    # Wavelength range: 0.1879 - 1.937 um
    # Gold (Au)
    # Fit to P.B. Johnson and R.W. Christy, Physical Review B, Vol. 6, pp. 4370-9, 1972
    
        Au_JC_range = mp.FreqRange(min=from_um_factor/1.937, 
                                   max=from_um_factor/.1879)
        
        freq_0 = 1.000000082740371e-10 * from_um_factor
        gamma_0 = 4.4142682842363e-09 * from_um_factor
        sigma_0 = 3.3002903009040977e+21
        
        freq_1 = 0.3260039379724786 * from_um_factor
        gamma_1 = 0.03601307014052124 * from_um_factor
        sigma_1 = 103.74591029640469
        
        freq_2 = 0.47387215165339414 * from_um_factor
        gamma_2 = 0.34294093699162054 * from_um_factor
        sigma_2 = 14.168079504545002
        
        freq_3 = 2.3910144662345445 * from_um_factor
        gamma_3 = 0.5900378254265015 * from_um_factor
        sigma_3 = 0.8155991478264435
        
        freq_4 = 3.3577076082530537 * from_um_factor
        gamma_4 = 1.6689250252226686 * from_um_factor
        sigma_4 = 2.038193481751111
        
        freq_5 = 8.915719663759013 * from_um_factor
        gamma_5 = 7.539763092679264 * from_um_factor
        sigma_5 = 3.74409654935571
        
        Au_JC_susc = [mp.DrudeSusceptibility(frequency=freq_0, gamma=gamma_0, sigma=sigma_0),
                      mp.LorentzianSusceptibility(frequency=freq_1, gamma=gamma_1, sigma=sigma_1),
                      mp.LorentzianSusceptibility(frequency=freq_2, gamma=gamma_2, sigma=sigma_2),
                      mp.LorentzianSusceptibility(frequency=freq_3, gamma=gamma_3, sigma=sigma_3),
                      mp.LorentzianSusceptibility(frequency=freq_4, gamma=gamma_4, sigma=sigma_4),
                      mp.LorentzianSusceptibility(frequency=freq_5, gamma=gamma_5, sigma=sigma_5)]
        
        Au_JC = mp.Medium(epsilon=1.0,#6.1599, 
                          E_susceptibilities=Au_JC_susc, 
                          valid_freq_range=Au_JC_range)
        Au_JC.from_um_factor = from_um_factor
        
        return Au_JC

    elif material=="Au" and paper=="P":
        
    #------------------------------------------------------------------
    # Metals from D. Barchiesi and T. Grosges, J. Nanophotonics, Vol. 8, 08996, 2015
    # Wavelength range: 0.4 - 0.8 um
    # Gold (Au)
    # Fit to E.D. Palik, Handbook of Optical Constants, Academic Press, 1985 

        metal_visible_range = mp.FreqRange(min=from_um_factor*1e3/800, 
                                           max=from_um_factor*1e3/400)
        
        Au_visible_frq0 = 1*from_um_factor/0.0473629248511456
        Au_visible_gam0 = 1*from_um_factor/0.255476199605166
        Au_visible_sig0 = 1
        
        Au_visible_frq1 = 1*from_um_factor/0.800619321082804
        Au_visible_gam1 = 1*from_um_factor/0.381870287531951
        Au_visible_sig1 = -169.060953137985
        
        Au_visible_susc = [mp.DrudeSusceptibility(frequency=Au_visible_frq0, gamma=Au_visible_gam0, sigma=Au_visible_sig0),
                           mp.LorentzianSusceptibility(frequency=Au_visible_frq1, gamma=Au_visible_gam1, sigma=Au_visible_sig1)]
        
        Au_visible = mp.Medium(epsilon=0.6888, E_susceptibilities=Au_visible_susc, 
                               valid_freq_range=metal_visible_range)
        Au_visible.from_um_factor = from_um_factor
        
        return Au_visible
    
    elif material=="Au":
        raise ValueError("No source found for Au with that name")

    ############ SILVER ###############################################

    if material=="Ag" and paper=="R":
        
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
        Ag.from_um_factor = from_um_factor
        
        return Ag
    
    elif material=="Ag" and paper=="JC":
        
    #------------------------------------------------------------------
    # Metal from my own fit
    # Wavelength range: 0.1879 - 0.8211 um
    # Gold (Au)
    # Fit to P.B. Johnson and R.W. Christy, Physical Review B, Vol. 6, pp. 4370-9, 1972
    # Reduced range to improve convergence
    
        Ag_JC_range = mp.FreqRange(min=from_um_factor*1e3/821.1, 
                                   max=from_um_factor*1e3/187.9)
        
        freq_0 = 1.000000082740371e-10 * from_um_factor
        gamma_0 = 0.008487871792800084 * from_um_factor
        sigma_0 = 5.545340096443978e+21
        
        freq_1 = 0.0008857703799738381 * from_um_factor
        gamma_1 = 5.681495075323141 * from_um_factor
        sigma_1 = 7.73865735861863
        
        freq_2 = 3.5889483054274587 * from_um_factor
        gamma_2 = 0.5142171316339426 * from_um_factor
        sigma_2 = 0.3602917089945181
        
        freq_3 = 96.93018042700993 * from_um_factor
        gamma_3 = 0.0002454108091400897 * from_um_factor
        sigma_3 = 2.2055854266028954
        
        freq_4 = 4.243182517437894 * from_um_factor
        gamma_4 = 1.0115197559416669 * from_um_factor
        sigma_4 = 0.5560036781232447
        
        freq_5 = 5.375891811139136 * from_um_factor
        gamma_5 = 1.6462280921732821 * from_um_factor
        sigma_5 = 0.7492696272872168
        
        Ag_JC_susc = [mp.DrudeSusceptibility(frequency=freq_0, gamma=gamma_0, sigma=sigma_0),
                      mp.LorentzianSusceptibility(frequency=freq_1, gamma=gamma_1, sigma=sigma_1),
                      mp.LorentzianSusceptibility(frequency=freq_2, gamma=gamma_2, sigma=sigma_2),
                      mp.LorentzianSusceptibility(frequency=freq_3, gamma=gamma_3, sigma=sigma_3),
                      mp.LorentzianSusceptibility(frequency=freq_4, gamma=gamma_4, sigma=sigma_4),
                      mp.LorentzianSusceptibility(frequency=freq_5, gamma=gamma_5, sigma=sigma_5)]
        
        Ag_JC = mp.Medium(epsilon=1.0,#6.1599, 
                          E_susceptibilities=Ag_JC_susc, 
                          valid_freq_range=Ag_JC_range)
        Ag_JC.from_um_factor = from_um_factor
        
        return Ag_JC
    
    elif material=="Ag" and paper=="P":

    #------------------------------------------------------------------
    # Metals from D. Barchiesi and T. Grosges, J. Nanophotonics, Vol. 8, 08996, 2015
    # Wavelength range: 0.4 - 0.8 um
    ## WARNING: unstable; field divergence may occur
    # Silver (Au)
    # Fit to E.D. Palik, Handbook of Optical Constants, Academic Press, 1985 

        metal_visible_range = mp.FreqRange(min=from_um_factor*1e3/800, 
                                           max=from_um_factor*1e3/400)
        
        Ag_visible_frq0 = 1*from_um_factor/0.142050162130618
        Ag_visible_gam0 = 1*from_um_factor/18.0357292925015
        Ag_visible_sig0 = 1
        
        Ag_visible_frq1 = 1*from_um_factor/0.115692151792108
        Ag_visible_gam1 = 1*from_um_factor/0.257794324096575
        Ag_visible_sig1 = 3.74465275944019
        
        Ag_visible_susc = [mp.DrudeSusceptibility(frequency=Ag_visible_frq0, gamma=Ag_visible_gam0, sigma=Ag_visible_sig0),
                           mp.LorentzianSusceptibility(frequency=Ag_visible_frq1, gamma=Ag_visible_gam1, sigma=Ag_visible_sig1)]
        
        Ag_visible = mp.Medium(epsilon=0.0067526, 
                               E_susceptibilities=Ag_visible_susc, 
                               valid_freq_range=metal_visible_range)
        Ag_visible.from_um_factor = from_um_factor
        
        return Ag_visible

    elif material=="Ag":    
        raise ValueError("No source found for Ag with that name")
    
    else:
        raise ValueError("No source found for that material")

#%%
    
def recognize_material(material_or_index, english=True):
    
    materials_dict = {"Vacuum": 1,
                      "Water": 1.33,
                      "Glass": 1.54}
    materials_short_dict = {"vac": "Vacuum",
                            "wat": "Water",
                            "gl": "Glass"}
    materials_spanish_dict = {"Vacuum": "Vacío",
                              "Water": "Agua",
                              "Glass": "Vidrio"}
    materials_keys = list(materials_dict.keys())
    materials_values = list(materials_dict.values())
    materials_spanish = list(materials_spanish_dict.values())
    
    if isinstance(material_or_index, str):
        for short_key, material in materials_short_dict.items():
            if short_key in material_or_index.lower():
                return materials_dict[material]
        raise ValueError(f"Unrecognized material: must be in {materials_keys}")
        
    elif isinstance(material_or_index, float) or isinstance(material_or_index, int):
        try:
            if english:
                return materials_keys[materials_values.index(material_or_index)]
            else:
                return materials_spanish[materials_values.index(material_or_index)]
        except:
            raise ValueError(f"Unrecognized material's index': must be in {materials_values}")
            
    else:
        raise ValueError(f"Unrecognized format for material: must be in {materials_dict}")

#%% EPSILON INTERPOLATION

def epsilon_interpoler_from_n(wlen, complex_n):
    """
    Generates an interpolation function for epsilon from experimental N data.

    Parameters
    ----------
    wlen : np.array, list
        Wavelength in nm.
    complex_n : np.array, list
        Complex refractive index N = n + ik, dimensionless.

    Returns
    -------
    epsilon_function : function
        Epsilon interpoler that takes wavelength in nm as argument and returns 
        complex dielectric constant or relative permitivitty 
        epsilon = epsilon' + i epsilon'', dimensionless.
    """

    n_function = interp1d(wlen, np.real(complex_n), kind="cubic")
    k_function = interp1d(wlen, np.imag(complex_n), kind="cubic")
    N_function = lambda wl : n_function(wl) + 1j * k_function(wl)
    epsilon_function = lambda wl : np.power(N_function(wl), 2)
    
    return epsilon_function

def epsilon_interpoler_from_epsilon(wlen, complex_epsilon):
    """
    Generates an interpolation function for epsilon from experimental epsilon data.

    Parameters
    ----------
    wlen : np.array, list
        Wavelength in nm.
    complex_epsilon : np.array, list
        Complex dielectric constant or relative permitivitty 
        epsilon = epsilon' + i epsilon'', dimensionless.

    Returns
    -------
    epsilon_function : function
        Epsilon interpoler that takes wavelength in nm as argument and returns 
        complex dielectric constant or relative permitivitty 
        epsilon = epsilon' + i epsilon'', dimensionless.
    """

    real_function = interp1d(wlen, np.real(complex_epsilon), kind="cubic")
    imag_function = interp1d(wlen, np.imag(complex_epsilon), kind="cubic")
    epsilon_function = lambda wl : real_function(wl) + 1j * imag_function(wl)
    
    return epsilon_function

def epsilon_function_from_meep(material="Au", paper="JC", from_um_factor=1e-3):
    
    """
    Generates a function for isotropic epsilon from Meep Drude-Lorentz fit data.
    
    Parameters
    ----------
    material="Au" : str
        Material's chemical symbol. Available: 'Au' for gold.
    paper="JC" : str
        Paper source of experimental data. Available: 'JC' for Johnson 
        and Christy, 'R' for Rakic, 'P' for Palik.
    from_um_factor=1e-3 : float, optional
        Meep factor of length scale implying 1 Meep length unit is 
        from_um_factor length units in μm. If provided, the function takes 
        wavelength in Meep units instead of nm.
        
    Returns
    -------
    epsilon_function : function
        Epsilon function that takes wavelength in nm or Meep units as argument 
        and returns complex dielectric constant or relative permitivitty 
        epsilon = epsilon' + i epsilon'', dimensionless.
        
    Raises
    ------
    ValueError : "Material should either be..."
        When the desired material isn't available.
    ValueError : "Reference paper should either be..."
        When the desired paper reference of experimental data isn't available.
    ValueError : "Data should either be from..."
        When the desired data source isn't available.
    ValueError : "Experimental data couldn't be found. Sorry!"
        When the combination of parameters causes the data not to be found.
    """

    
    available_materials = {"Au": "gold", "Ag": "silver"}
    available_papers = {"JC": "Johnson & Christy", "R": "Rakic", "P": "Palik"}
    
    if material not in available_materials.keys():
        error = "Data should either be from "
        error += vu.enumerate_string(vu.join_strings_dict(available_materials), 
                                     "or", True)
        raise ValueError(error)
    if paper not in available_papers.keys():
        error = "Reference paper for experimental data should either be "
        error += vu.enumerate_string(vu.join_strings_dict(available_papers), 
                                     "or", True)
        raise ValueError(error)
        
    medium = import_medium(material, 
                           paper=paper) # This one has from_um_factor=1
    
    print(f"Data loaded using Meep and '{paper}'")
    epsilon_function = lambda wlen : medium.epsilon(1/(wlen*from_um_factor))[0,0]
    # To pass it to the medium, I transform wavelength from nm (or Meep units) to um
    
    wlen_range = 1/(np.flip(np.array([*medium.valid_freq_range]))*from_um_factor)
    epsilon_function._wlen_range_ = wlen_range
    epsilon_function._from_um_factor_ = from_um_factor
    
    return epsilon_function

def epsilon_data_from_file(material="Au", paper="JC", reference="RIinfo"):
    """
    Loads experimental data for epsilon from file.
    
    Parameters
    ----------
    material="Au" : str
        Material's chemical symbol. Available: 'Au' for gold.
    paper="JC" : str
        Paper source of experimental data. Available: 'JC' for Johnson 
        and Christy, 'R' for Rakic.
    reference="RIinfo" : str
        Reference from which the data was extracted, for example a web page. 
        Available: 'RIinfo' for 'www.refractiveindex.info'
        
    Returns
    -------
    wavelength : np.array
        Wavelength values in nm.
    epsilon : np.array
        Complex epsilon data, epsilon = epsilon' + i epsilon'', dimensionless.
    
    Raises
    ------
    ValueError : "Material should either be..."
        When the desired material isn't available.
    ValueError : "Reference paper should either be..."
        When the desired paper reference of experimental data isn't available.
    ValueError : "Data should either be from..."
        When the desired data source isn't available.
    ValueError : "Experimental data couldn't be found. Sorry!"
        When the combination of parameters causes the data not to be found.
    """
    
    available_materials = {"Au": "gold", "Ag": "silver"}
    available_papers = {"JC": "Johnson & Christy",
                        "R": "Rakic"}
    available_references = {"RIinfo": "www.refractiveindex.info"}
    
    if material not in available_materials.keys():
        error = "Data should either be from "
        error += vu.enumerate_string(vu.join_strings_dict(available_materials), 
                                     "or", True)
        raise ValueError(error)
    if paper not in available_papers.keys():
        error = "Reference paper for experimental data should either be "
        error += vu.enumerate_string(vu.join_strings_dict(available_papers), 
                                     "or", True)
        raise ValueError(error)
    if reference not in available_references.keys():
        error = "Experimental data should either be extracted from "
        error += vu.enumerate_string(vu.join_strings_dict(available_references), 
                                     "or", True)
        raise ValueError(error)
    
    data_series = os.listdir(os.path.join(syshome, 'MaterialsData'))
    
    try:
        data_files = []
        for df in data_series:
            if (f"_{paper}_") in df and material in df and reference in df:
                data_files.append( os.path.join(syshome, 'MaterialsData', df) )
    except:
        raise ValueError("Experimental data couldn't be found. Sorry!")
    
    file = data_files[0]
    data = np.loadtxt(file)
    
    wavelength = data[:,0]
        
    if 'N' in file:
        epsilon = np.power(data[:,1] + 1j*data[:,2], 2)
        print(f"Refractive index data loaded from '{file}'")
    elif "eps" in file.lower():
        epsilon = data[:,1] + 1j*data[:,2]
        print(f"Epsilon data loaded from '{file}'")
    else:
        raise ValueError("Experimental data couldn't be recognized. Sorry!")
    
    return wavelength, epsilon

def epsilon_function_from_file(material="Au", paper="JC", reference="RIinfo", 
                               from_um_factor=1e-3, plot=False):
    """
    Generates an interpolation function for epsilon from experimental data.
    
    Parameters
    ----------
    material="Au" : str
        Material's chemical symbol. Available: 'Au' for gold.
    paper="JC" : str
        Paper source of experimental data. Available: 'JC' for Johnson 
        and Christy, 'R' for Rakic.
    reference="RIinfo" : str
        Reference from which the data was extracted, for example a web page. 
        Available: 'RIinfo' for 'www.refractiveindex.info'
    from_um_factor=1e-3 : float, optional
        Meep factor of length scale implying 1 Meep length unit is 
        from_um_factor length units in μm. If provided, the function takes 
        wavelength in Meep units instead of nm.
    plot=False : bool
        Parameter that enables a plot of the interpolation and the data used.
        
    Returns
    -------
    epsilon_function : function
        Epsilon function that takes wavelength in nm or Meep units as argument 
        and returns complex dielectric constant or relative permitivitty 
        epsilon = epsilon' + i epsilon'', dimensionless.
    
    Raises
    ------
    ValueError : "Material should either be..."
        When the desired material isn't available.
    ValueError : "Reference paper should either be..."
        When the desired paper reference of experimental data isn't available.
    ValueError : "Data should either be from..."
        When the desired data source isn't available.
    ValueError : "Experimental data couldn't be found. Sorry!"
        When the combination of parameters causes the data not to be found.
    """
    
    wavelength, epsilon_data = epsilon_data_from_file(material, paper, reference)
    
    meep_wlen = wavelength / (1e3 * from_um_factor)
    # Change wavelength units if necessary
    # Going from nm to Meep units
    
    epsilon_function = epsilon_interpoler_from_epsilon(meep_wlen, epsilon_data)
    
    meep_wlen_range = [min(meep_wlen), max(meep_wlen)]
    epsilon_function._wlen_range_ = np.array(meep_wlen_range)
    epsilon_function._from_um_factor_ = from_um_factor
            
    if plot:
        wlen_range = [min(wavelength), max(wavelength)]
        wlen_long = np.linspace(*wlen_range, 500)
        
        epsilon_interpolated = epsilon_function(wlen_long)
                
        functions = [np.abs, np.real, np.imag]
        titles = ["Absolute value", "Real part", "Imaginary part"]
        ylabels = [r"|$\epsilon$| [nm$^3$]", r"Re($\epsilon$) [nm$^3$]", 
                   r"Im($\epsilon$) [nm$^3$]"]
        
        nplots = len(functions)
        fig = plt.figure(figsize=(nplots*6.4, 6.4))
        axes = fig.subplots(ncols=nplots)
        
        max_value = []
        min_value = []
        for ax, f, t, y in zip(axes, functions, titles, ylabels):
            ax.set_title(t)
            ax.plot(wavelength, f(epsilon_data), "ob", label="Data")
            ax.plot(wlen_long, f(epsilon_interpolated), "-r", label="Interpolation")
            ax.xaxis.set_label_text(r"Wavelength $\lambda$ [nm]")
            ax.yaxis.set_label_text(y)
            ax.legend()
            ax.set_xlim(*wlen_range)
            max_value.append(max([max(f(epsilon_data)), max(f(epsilon_interpolated))]))
            min_value.append(min([min(f(epsilon_data)), min(f(epsilon_interpolated))]))
                
        for ax in axes: ax.set_ylim([min(min_value)-.1*(max(max_value)-min(min_value)), 
                                     max(max_value)+.1*(max(max_value)-min(min_value))])
        axes[0].text(-.1, -.13, f"{material}{paper}{reference}", 
                     transform=axes[0].transAxes)
    
    return epsilon_function

def epsilon_function(material="Au", paper="JC", reference="RIinfo",
                     from_um_factor=1e-3):
    """
    Generates an interpolation function for epsilon from experimental data.
    
    Parameters
    ----------
    material="Au" : str
        Material's chemical symbol. Available: 'Au' for gold.
    paper="JC" : str
        Reference paper of experimental data. Available: 'JC' for Johnson and 
        Christy, 'R' for Rakic, 'P' for Palik.
    reference="RIinfo" : str
        Reference from which the data was extracted, for example a web page. 
        Available: 'RIinfo' for 'www.refractiveindex.info' and 'Meep' for Meep 
        materials library that uses a Drude-Lorentz model to fit data.
    from_um_factor=1e-3 : float, optional
        Meep factor of length scale implying 1 Meep length unit is 
        from_um_factor length units in μm. If provided, the function takes 
        wavelength in Meep units instead of nm.
        
    Returns
    -------
    epsilon_function : function
        Epsilon function that takes wavelength in nm or Meep units as argument 
        and returns complex dielectric constant or relative permitivitty 
        epsilon = epsilon' + i epsilon'', dimensionless.
        
    Raises
    ------
    ValueError : "Material should either be..."
        When the desired material isn't available.
    ValueError : "Reference paper should either be..."
        When the desired paper reference of experimental data isn't available.
    ValueError : "Data should either be from..."
        When the desired data source isn't available.
    ValueError : "Experimental data couldn't be found. Sorry!"
        When the combination of parameters causes the data not to be found.
    """
    
    available_references = {"RIinfo": "www.refractiveindex.info",
                           "Meep": "Meep materials library & Drude-Lorentz fit"}
    
    if reference not in available_references.keys():
        error = "Data should either be from"
        error += vu.enumerate_string(vu.join_strings_dict(available_references), 
                                     "or", True)
        raise ValueError(error)
    
    if reference=="Meep":
        epsilon_function = epsilon_function_from_meep(material, paper, 
                                                      from_um_factor)
    else:
        epsilon_function = epsilon_function_from_file(material, paper, 
                                                      reference,
                                                      from_um_factor)
        
    
    return epsilon_function

#%% REFRACTIVE INDEX INTERPOLATION

def n_interpoler_from_n(wlen, complex_n):
    """
    Generates an interpolation function for N from experimental N data.

    Parameters
    ----------
    wlen : np.array, list
        Wavelength in nm.
    complex_n : np.array, list
        Complex refractive index N = n + ik, dimensionless.

    Returns
    -------
    N_function : function
        Refractive index N interpoler that takes wavelength in nm as argument 
        and returns complex refractive index N = n + i k, dimensionless.
    """

    n_function = interp1d(wlen, np.real(complex_n), kind="cubic")
    k_function = interp1d(wlen, np.imag(complex_n), kind="cubic")
    N_function = lambda wl : n_function(wl) + 1j * k_function(wl)
    
    return N_function

def n_interpoler_from_epsilon(wlen, complex_epsilon):
    """
    Generates an interpolation function for N from experimental epsilon data.

    Parameters
    ----------
    wlen : np.array, list
        Wavelength in nm.
    complex_epsilon : np.array, list
        Complex dielectric constant or relative permitivitty 
        epsilon = epsilon' + i epsilon'', dimensionless.

    Returns
    -------
    N_function : function
        Refractive index N interpoler that takes wavelength in nm as argument 
        and returns complex refractive index N = n + i k, dimensionless.
    """

    real_function = interp1d(wlen, np.real(complex_epsilon), kind="cubic")
    imag_function = interp1d(wlen, np.imag(complex_epsilon), kind="cubic")
    epsilon_function = lambda wl : real_function(wl) + 1j * imag_function(wl)
    N_function = lambda wl : np.sqrt(epsilon_function(wl))
    
    return N_function

def n_function_from_meep(material="Au", paper="JC", from_um_factor=1e-3):
    
    """
    Generates a function for isotropic N from Meep Drude-Lorentz fit data.
    
    Parameters
    ----------
    material="Au" : str
        Material's chemical symbol. Available: 'Au' for gold.
    paper="JC" : str
        Paper source of experimental data. Available: 'JC' for Johnson 
        and Christy, 'R' for Rakic, 'P' for Palik.
    from_um_factor=1e-3 : float, optional
        Meep factor of length scale implying 1 Meep length unit is 
        from_um_factor length units in μm. If provided, the function takes 
        wavelength in Meep units instead of nm.
        
    Returns
    -------
    N_function : function
        Refractive index N interpoler that takes wavelength in nm as argument 
        and returns complex refractive index N = n + i k, dimensionless.
        
    Raises
    ------
    ValueError : "Material should either be..."
        When the desired material isn't available.
    ValueError : "Reference paper should either be..."
        When the desired paper reference of experimental data isn't available.
    ValueError : "Data should either be from..."
        When the desired data source isn't available.
    ValueError : "Experimental data couldn't be found. Sorry!"
        When the combination of parameters causes the data not to be found.
    """

    
    available_materials = {"Au": "gold", "Ag": "silver"}
    available_papers = {"JC": "Johnson & Christy", "R": "Rakic", "P": "Palik"}
    
    if material not in available_materials.keys():
        error = "Data should either be from "
        error += vu.enumerate_string(vu.join_strings_dict(available_materials), 
                                     "or", True)
        raise ValueError(error)
    if paper not in available_papers.keys():
        error = "Reference paper for experimental data should either be "
        error += vu.enumerate_string(vu.join_strings_dict(available_papers), 
                                     "or", True)
        raise ValueError(error)
        
    medium = import_medium(material, 
                           paper=paper) # This one has from_um_factor=1
    
    print(f"Data loaded using Meep and '{paper}'")
    epsilon_function = lambda wlen : medium.epsilon(1/(wlen*from_um_factor))[0,0]
    # To pass it to the medium, I transform wavelength from nm (or Meep units) to um
    N_function = lambda wlen : np.sqrt(epsilon_function(wlen))
    
    wlen_range = 1/(np.flip(np.array([*medium.valid_freq_range]))*from_um_factor)
    N_function._wlen_range_ = wlen_range
    N_function._from_um_factor_ = from_um_factor
    
    return N_function

def n_data_from_file(material="Au", paper="JC", reference="RIinfo"):
    """
    Loads experimental data for epsilon from file.
    
    Parameters
    ----------
    material="Au" : str
        Material's chemical symbol. Available: 'Au' for gold.
    paper="JC" : str
        Paper source of experimental data. Available: 'JC' for Johnson 
        and Christy, 'R' for Rakic.
    reference="RIinfo" : str
        Reference from which the data was extracted, for example a web page. 
        Available: 'RIinfo' for 'www.refractiveindex.info'
        
    Returns
    -------
    wavelength : np.array
        Wavelength values in nm.
    N : np.array
        Refractive index N data N = n + i k, dimensionless.
    
    Raises
    ------
    ValueError : "Material should either be..."
        When the desired material isn't available.
    ValueError : "Reference paper should either be..."
        When the desired paper reference of experimental data isn't available.
    ValueError : "Data should either be from..."
        When the desired data source isn't available.
    ValueError : "Experimental data couldn't be found. Sorry!"
        When the combination of parameters causes the data not to be found.
    """
    
    available_materials = {"Au": "gold", "Ag": "silver"}
    available_papers = {"JC": "Johnson & Christy",
                        "R": "Rakic"}
    available_references = {"RIinfo": "www.refractiveindex.info"}
    
    if material not in available_materials.keys():
        error = "Data should either be from "
        error += vu.enumerate_string(vu.join_strings_dict(available_materials), 
                                     "or", True)
        raise ValueError(error)
    if paper not in available_papers.keys():
        error = "Reference paper for experimental data should either be "
        error += vu.enumerate_string(vu.join_strings_dict(available_papers), 
                                     "or", True)
        raise ValueError(error)
    if reference not in available_references.keys():
        error = "Experimental data should either be extracted from "
        error += vu.enumerate_string(vu.join_strings_dict(available_references), 
                                     "or", True)
        raise ValueError(error)
    
    data_series = os.listdir(os.path.join(syshome, 'MaterialsData'))
    
    try:
        data_files = []
        for df in data_series:
            if (f"_{paper}_") in df and material in df and reference in df:
                data_files.append( os.path.join(syshome, 'MaterialsData', df) )
    except:
        raise ValueError("Experimental data couldn't be found. Sorry!")
    
    file = data_files[0]
    data = np.loadtxt(file)
    
    wavelength = data[:,0]
        
    if 'N' in file:
        N = data[:,1] + 1j*data[:,2]
        print(f"Refractive index data loaded from '{file}'")
    elif "eps" in file.lower():
        N = np.sqrt(data[:,1] + 1j*data[:,2])
        print(f"Epsilon data loaded from '{file}'")
    else:
        raise ValueError("Experimental data couldn't be recognized. Sorry!")
    
    return wavelength, N

def n_function_from_file(material="Au", paper="JC", reference="RIinfo", 
                         from_um_factor=1e-3, plot=False):
    """
    Generates an interpolation function for N from experimental data.
    
    Parameters
    ----------
    material="Au" : str
        Material's chemical symbol. Available: 'Au' for gold.
    paper="JC" : str
        Paper source of experimental data. Available: 'JC' for Johnson 
        and Christy, 'R' for Rakic.
    reference="RIinfo" : str
        Reference from which the data was extracted, for example a web page. 
        Available: 'RIinfo' for 'www.refractiveindex.info'
    from_um_factor=1e-3 : float, optional
        Meep factor of length scale implying 1 Meep length unit is 
        from_um_factor length units in μm. If provided, the function takes 
        wavelength in Meep units instead of nm.
    plot=False : bool
        Parameter that enables a plot of the interpolation and the data used.
        
    Returns
    -------
    N_function : function
        Refractive index N interpoler that takes wavelength in nm or Meep units 
        as argument  and returns complex refractive index N = n + i k, 
        dimensionless.
    
    Raises
    ------
    ValueError : "Material should either be..."
        When the desired material isn't available.
    ValueError : "Reference paper should either be..."
        When the desired paper reference of experimental data isn't available.
    ValueError : "Data should either be from..."
        When the desired data source isn't available.
    ValueError : "Experimental data couldn't be found. Sorry!"
        When the combination of parameters causes the data not to be found.
    """
    
    wavelength, N_data = n_data_from_file(material, paper, reference)
    
    meep_wlen = wavelength / (1e3 * from_um_factor)
    # Change wavelength units if necessary
    # Going from nm to Meep units
    
    N_function = n_interpoler_from_n(meep_wlen, N_data)
    
    meep_wlen_range = [min(meep_wlen), max(meep_wlen)]
    N_function._wlen_range_ = np.array(meep_wlen_range)
    N_function._from_um_factor_ = from_um_factor
            
    if plot:
        wlen_range = [min(wavelength), max(wavelength)]
        wlen_long = np.linspace(*wlen_range, 500)
        
        N_interpolated = N_function(wlen_long)
                
        functions = [np.abs, np.real, np.imag]
        titles = ["Absolute value", "Real part", "Imaginary part"]
        ylabels = [r"|N| [nm$^3$]", r"Re(N) [nm$^3$]", r"Im(N) [nm$^3$]"]
        
        nplots = len(functions)
        fig = plt.figure(figsize=(nplots*6.4, 6.4))
        axes = fig.subplots(ncols=nplots)
        
        max_value = []
        min_value = []
        for ax, f, t, y in zip(axes, functions, titles, ylabels):
            ax.set_title(t)
            ax.plot(wavelength, f(N_data), "ob", label="Data")
            ax.plot(wlen_long, f(N_interpolated), "-r", label="Interpolation")
            ax.xaxis.set_label_text(r"Wavelength $\lambda$ [nm]")
            ax.yaxis.set_label_text(y)
            ax.legend()
            ax.set_xlim(*wlen_range)
            max_value.append(max([max(f(N_data)), max(f(N_data))]))
            min_value.append(min([min(f(N_data)), min(f(N_data))]))
                
        for ax in axes: ax.set_ylim([min(min_value)-.1*(max(max_value)-min(min_value)), 
                                      max(max_value)+.1*(max(max_value)-min(min_value))])
        axes[0].text(-.1, -.13, f"{material}{paper}{reference}", 
                      transform=axes[0].transAxes)
    
    return N_function

def n_function(material="Au", paper="JC", reference="RIinfo", 
               from_um_factor=1e-3):
    """
    Generates an interpolation function for N from experimental data.
    
    Parameters
    ----------
    material="Au" : str
        Material's chemical symbol. Available: 'Au' for gold.
    paper="JC" : str
        Reference paper of experimental data. Available: 'JC' for Johnson and 
        Christy, 'R' for Rakic, 'P' for Palik.
    reference="RIinfo" : str
        Reference from which the data was extracted, for example a web page. 
        Available: 'RIinfo' for 'www.refractiveindex.info' and 'Meep' for Meep 
        materials library that uses a Drude-Lorentz model to fit data.
    from_um_factor=1e-3 : float, optional
        Meep factor of length scale implying 1 Meep length unit is 
        from_um_factor length units in μm. If provided, the function takes 
        wavelength in Meep units instead of nm.
        
    Returns
    -------
    N_function : function
        Refractive index N interpoler that takes wavelength in nm or Meep units 
        as argument  and returns complex refractive index N = n + i k, 
        dimensionless.
        
    Raises
    ------
    ValueError : "Material should either be..."
        When the desired material isn't available.
    ValueError : "Reference paper should either be..."
        When the desired paper reference of experimental data isn't available.
    ValueError : "Data should either be from..."
        When the desired data source isn't available.
    ValueError : "Experimental data couldn't be found. Sorry!"
        When the combination of parameters causes the data not to be found.
    """
    
    available_references = {"RIinfo": "www.refractiveindex.info",
                           "Meep": "Meep materials library & Drude-Lorentz fit"}
    
    if reference not in available_references.keys():
        error = "Data should either be from"
        error += vu.enumerate_string(vu.join_strings_dict(available_references), 
                                     "or", True)
        raise ValueError(error)
    
    if reference=="Meep":
        N_function = n_function_from_meep(material, paper, from_um_factor)
    else:
        N_function = n_function_from_file(material, paper, 
                                          reference, from_um_factor)
        
    
    return N_function

#%% SCATTERING CALCULATED USING MEEP

def sigma_scatt_meep(r, material, paper, wlen, 
                     surrounding_index=1, asEfficiency=False):
    
    """
    Calculates scattering cross section using Mie theory for a spherical NP.

    Parameters
    ----------
    r : float
        Spherical NP radius. Measured in nm.
    material: str
        Name of the desired material.
    paper="R": str
        Name of desired source for experimental input data of medium.
    wlen : float, list, np.array
        Incident light wavelength. Measured in nm.
    inner_N=1.458 : float, list, np.array
        Spherical NP inner medium's complex refractive index. The default is 
        1.458 for fused-silica.
    surrounding_index=1 : float, list, np.array
        Surrounding medium's complex refractive index. The default is 
        1 for vacuum.
    asEfficiency : bool, optional
        If false, scattering cross section sigma is returned, measured in nm^2. 
        If true, scattering effienciency Q is returned, dimensionless and 
        related to scattering cross section by sigma = pi r^2 Q 
        for a spherical NP. The default is False.

    Returns
    -------
    scattering : float, np.array
        The scattering cross section calculated using Mie theory and measured 
        in nm^2. In case `asEfficiency=True`, scattering effiency is returned 
        instead, dimensionless.
        
    Raises
    ------
    ValueError : "Must have as many surrounding refractive index values as..."
        If the length of `wlen` and `surrounding_N` differ.

    """
    
    try:
        wlen = np.array([*wlen])
    except:
        wlen = np.array([wlen])
    
    try:
        len(surrounding_index)
    except TypeError:
        surrounding_index = [*[surrounding_index]*len(wlen)]
    
    medium = import_medium(material, paper=paper, from_um_factor=1e-3)
    # Meep unit chosen to be 1 nm
    
    epsilon = np.array([ medium.epsilon(1 / wl)[0,0] for wl in wlen ])
    N = vt.N_from_epsilon( epsilon )
    
    scattering = vt.sigma_scatt_Mie(r,  wlen,  inner_N=N, 
                                    surrounding_N=surrounding_index,
                                    asEfficiency=asEfficiency)
    
    return scattering

def max_scatt_meep(r, material, paper, wlen_range, wlen_delta=0.5,
                   surrounding_index=1, asEfficiency=False):
            
    """
    Returns maximum scattering cross section using Mie theory for a nanosphere.

    Parameters
    ----------
    r : float
        Spherical NP radius. Measured in nm.
    material: str
        Name of the desired material.
    paper="R": str
        Name of desired source for experimental input data of medium.
    wlen_range : tuple
        Wavelength range to check. Measured in nm.
    wlen_delta : float
        Wavelength desired precision. Measured in nm.
    surrounding_index=1 : float, list, np.array
        Surrounding medium's complex refractive index. The default is 
        1 for vacuum.
    asEfficiency : bool, optional
        If false, scattering cross section sigma is returned, measured in nm^2. 
        If true, scattering effienciency Q is returned, dimensionless and 
        related to scattering cross section by sigma = pi r^2 Q 
        for a spherical NP. The default is False.

    Returns
    -------
    max_wlen : float
        The wavelength of maximum scattering cross secion, measured in nm.
    max_scatt : float
        The maximum scattering cross section calculated using Mie theory and 
        measured in nm^2. In case `asEfficiency=True`, scattering effiency is 
        returned instead, dimensionless.
        
    Raises
    ------
    ValueError : "Must have as many surrounding refractive index values as..."
        If the length of `wlen` and `surrounding_N` differ.

    """
    
    wavelength = np.arange(*wlen_range, wlen_delta)
    
    scattering = sigma_scatt_meep(r, material, paper, wavelength, 
                                  surrounding_index=surrounding_index, 
                                  asEfficiency=asEfficiency)
    
    max_index = np.argmax(scattering)
    max_wlen = wavelength[max_index]
    max_scatt = scattering[max_index]
    
    return max_wlen, max_scatt

#%% ABSORPTION CALCULATED USING MEEP

def sigma_abs_meep(r, material, paper, wlen, 
                   surrounding_index=1, asEfficiency=False):
    
    """
    Calculates absorption cross section using Mie theory for a spherical NP.

    Parameters
    ----------
    r : float
        Spherical NP radius. Measured in nm.
    material: str
        Name of the desired material.
    paper="R": str
        Name of desired source for experimental input data of medium.
    wlen : float, list, np.array
        Incident light wavelength. Measured in nm.
    inner_N=1.458 : float, list, np.array
        Spherical NP inner medium's complex refractive index. The default is 
        1.458 for fused-silica.
    surrounding_index=1 : float, list, np.array
        Surrounding medium's complex refractive index. The default is 
        1 for vacuum.
    asEfficiency : bool, optional
        If false, absorption cross section sigma is returned, measured in nm^2. 
        If true, absorption effienciency Q is returned, dimensionless and 
        related to absorption cross section by sigma = pi r^2 Q 
        for a spherical NP. The default is False.

    Returns
    -------
    absorption : float, np.array
        The absorption cross section calculated using Mie theory and measured 
        in nm^2. In case `asEfficiency=True`, scattering effiency is returned 
        instead, dimensionless.
        
    Raises
    ------
    ValueError : "Must have as many surrounding refractive index values as..."
        If the length of `wlen` and `surrounding_N` differ.

    """
    
    try:
        wlen = np.array([*wlen])
    except:
        wlen = np.array([wlen])
    
    try:
        len(surrounding_index)
    except TypeError:
        surrounding_index = [*[surrounding_index]*len(wlen)]
    
    medium = import_medium(material, paper=paper, from_um_factor=1e-3)
    # Meep unit chosen to be 1 nm
    
    epsilon = np.array([ medium.epsilon(1 / wl)[0,0] for wl in wlen ])
    N = vt.N_from_epsilon( epsilon )
    
    absorption = vt.sigma_abs_Mie(r,  wlen,  inner_N=N, 
                                  surrounding_N=surrounding_index,
                                  asEfficiency=asEfficiency)
    
    return absorption

def max_abs_meep(r, material, paper, wlen_range, wlen_delta=0.5,
                 surrounding_index=1, asEfficiency=False):
            
    """
    Returns maximum absorption cross section using Mie theory for a nanosphere.

    Parameters
    ----------
    r : float
        Spherical NP radius. Measured in nm.
    material: str
        Name of the desired material.
    paper="R": str
        Name of desired source for experimental input data of medium.
    wlen_range : tuple
        Wavelength range to check. Measured in nm.
    wlen_delta : float
        Wavelength desired precision. Measured in nm.
    surrounding_index=1 : float, list, np.array
        Surrounding medium's complex refractive index. The default is 
        1 for vacuum.
    asEfficiency : bool, optional
        If false, absorption cross section sigma is returned, measured in nm^2. 
        If true, absorption effienciency Q is returned, dimensionless and 
        related to absorption cross section by sigma = pi r^2 Q 
        for a spherical NP. The default is False.

    Returns
    -------
    max_wlen : float
        The wavelength of maximum absorption cross secion, measured in nm.
    max_abs : float
        The maximum absorption cross section calculated using Mie theory and 
        measured in nm^2. In case `asEfficiency=True`, scattering effiency is 
        returned instead, dimensionless.
        
    Raises
    ------
    ValueError : "Must have as many surrounding refractive index values as..."
        If the length of `wlen` and `surrounding_N` differ.

    """
    
    wavelength = np.arange(*wlen_range, wlen_delta)
    
    absorption = sigma_abs_meep(r, material, paper, wavelength, 
                                surrounding_index=surrounding_index, 
                                asEfficiency=asEfficiency)
    
    max_index = np.argmax(absorption)
    max_wlen = wavelength[max_index]
    max_abs = absorption[max_index]
    
    return max_wlen, max_abs

#%% EXTINCTION CALCULATED USING MEEP

def sigma_ext_meep(r, material, paper, wlen, 
                   surrounding_index=1, asEfficiency=False):
    
    """
    Calculates extinction cross section using Mie theory for a spherical NP.

    Parameters
    ----------
    r : float
        Spherical NP radius. Measured in nm.
    material: str
        Name of the desired material.
    paper="R": str
        Name of desired source for experimental input data of medium.
    wlen : float, list, np.array
        Incident light wavelength. Measured in nm.
    inner_N=1.458 : float, list, np.array
        Spherical NP inner medium's complex refractive index. The default is 
        1.458 for fused-silica.
    surrounding_index=1 : float, list, np.array
        Surrounding medium's complex refractive index. The default is 
        1 for vacuum.
    asEfficiency : bool, optional
        If false, extinction cross section sigma is returned, measured in nm^2. 
        If true, extinction effienciency Q is returned, dimensionless and 
        related to extinction cross section by sigma = pi r^2 Q 
        for a spherical NP. The default is False.

    Returns
    -------
    extinction : float, np.array
        The extinction cross section calculated using Mie theory and measured 
        in nm^2. In case `asEfficiency=True`, extinction effiency is returned 
        instead, dimensionless.
        
    Raises
    ------
    ValueError : "Must have as many surrounding refractive index values as..."
        If the length of `wlen` and `surrounding_N` differ.

    """
    
    try:
        wlen = np.array([*wlen])
    except:
        wlen = np.array([wlen])
    
    try:
        len(surrounding_index)
    except TypeError:
        surrounding_index = [*[surrounding_index]*len(wlen)]
    
    medium = import_medium(material, paper=paper, from_um_factor=1e-3)
    # Meep unit chosen to be 1 nm
    
    epsilon = np.array([ medium.epsilon(1 / wl)[0,0] for wl in wlen ])
    N = vt.N_from_epsilon( epsilon )
    
    extinction = vt.sigma_ext_Mie(r,  wlen,  inner_N=N, 
                                 surrounding_N=surrounding_index,
                                 asEfficiency=asEfficiency)
    
    return extinction

def max_ext_meep(r, material, paper, wlen_range, wlen_delta=0.5,
                 surrounding_index=1, asEfficiency=False):
            
    """
    Returns maximum extinction cross section using Mie theory for a nanosphere.

    Parameters
    ----------
    r : float
        Spherical NP radius. Measured in nm.
    material: str
        Name of the desired material.
    paper="R": str
        Name of desired source for experimental input data of medium.
    wlen_range : tuple
        Wavelength range to check. Measured in nm.
    wlen_delta : float
        Wavelength desired precision. Measured in nm.
    surrounding_index=1 : float, list, np.array
        Surrounding medium's complex refractive index. The default is 
        1 for vacuum.
    asEfficiency : bool, optional
        If false, extinction cross section sigma is returned, measured in nm^2. 
        If true, extinction effienciency Q is returned, dimensionless and 
        related to extinction cross section by sigma = pi r^2 Q 
        for a spherical NP. The default is False.

    Returns
    -------
    max_wlen : float
        The wavelength of maximum absorption cross secion, measured in nm.
    max_abs : float
        The maximum extinction cross section calculated using Mie theory and 
        measured in nm^2. In case `asEfficiency=True`, extinction effiency is 
        returned instead, dimensionless.
        
    Raises
    ------
    ValueError : "Must have as many surrounding refractive index values as..."
        If the length of `wlen` and `surrounding_N` differ.

    """
    
    wavelength = np.arange(*wlen_range, wlen_delta)
    
    extinction = sigma_ext_meep(r, material, paper, wavelength, 
                               surrounding_index=surrounding_index, 
                               asEfficiency=asEfficiency)
    
    max_index = np.argmax(extinction)
    max_wlen = wavelength[max_index]
    max_ext = extinction[max_index]
    
    return max_wlen, max_ext

#%% MEEP MEDIUM THAT TAKES FUNCTION

class MediumFromFunction(mp.Medium):
    
    """A Meep Medium with isotropic complex epsilon and mu as functions of wavelength in nm"""
    
    def __init__(self, 
                 epsilon_diag=mp.Vector3(1, 1, 1),
                 epsilon_offdiag=mp.Vector3(),
                 mu_diag=mp.Vector3(1, 1, 1),
                 mu_offdiag=mp.Vector3(),
                 E_susceptibilities=[],
                 H_susceptibilities=[],
                 E_chi2_diag=mp.Vector3(),
                 E_chi3_diag=mp.Vector3(),
                 H_chi2_diag=mp.Vector3(),
                 H_chi3_diag=mp.Vector3(),
                 D_conductivity_diag=mp.Vector3(),
                 D_conductivity_offdiag=mp.Vector3(),
                 B_conductivity_diag=mp.Vector3(),
                 B_conductivity_offdiag=mp.Vector3(),
                 epsilon=None,
                 index=None,
                 mu=None,
                 chi2=None,
                 chi3=None,
                 D_conductivity=None,
                 B_conductivity=None,
                 E_chi2=None,
                 E_chi3=None,
                 H_chi2=None,
                 H_chi3=None,
                 valid_freq_range=mp.FreqRange(min=-mp.inf, max=mp.inf),
                 epsilon_function=lambda wlen:1,
                 mu_function=lambda wlen:1,
                 have_logger=False):
        
        super().__init__(epsilon_diag=epsilon_diag,
                 epsilon_offdiag=epsilon_offdiag,
                 mu_diag=mu_diag,
                 mu_offdiag=mu_offdiag,
                 E_susceptibilities=[],
                 H_susceptibilities=[],
                 E_chi2_diag=E_chi2_diag,
                 E_chi3_diag=E_chi3_diag,
                 H_chi2_diag=H_chi2_diag,
                 H_chi3_diag=H_chi3_diag,
                 D_conductivity_diag=D_conductivity_diag,
                 D_conductivity_offdiag=D_conductivity_offdiag,
                 B_conductivity_diag=B_conductivity_diag,
                 B_conductivity_offdiag=B_conductivity_offdiag,
                 epsilon=epsilon,
                 index=index,
                 mu=mu,
                 chi2=chi2,
                 chi3=chi3,
                 D_conductivity=D_conductivity,
                 B_conductivity=B_conductivity,
                 E_chi2=E_chi2,
                 E_chi3=E_chi3,
                 H_chi2=H_chi2,
                 H_chi3=H_chi3,
                 valid_freq_range=valid_freq_range)
        
        self.epsilon_function = epsilon_function
        self.mu_function = mu_function
                
        if len(E_susceptibilities)>0 or len(H_susceptibilities)>0:
            raise ValueError("This class doesn't take susceptibilities!")

        try:
            eps_wlen_range = epsilon_function._wlen_range_
        except:
            eps_wlen_range = np.array([-mp.inf, mp.inf])

        try:
            mu_wlen_range = mu_function._wlen_range_
        except:
            mu_wlen_range = np.array([-mp.inf, mp.inf])
            
        if self.valid_freq_range.min == -mp.inf:
            max_current_wlen = np.inf
        else:
            max_current_wlen = 1/self.valid_freq_range.min
        if self.valid_freq_range.max == mp.inf:
            min_current_wlen = -np.inf
        else:
            min_current_wlen = 1/self.valid_freq_range.max
        
        self.valid_wlen_range = np.array([max([eps_wlen_range[0],
                                               mu_wlen_range[0],
                                               min_current_wlen]),
                                          min([eps_wlen_range[1],
                                               mu_wlen_range[1],
                                               max_current_wlen])])
        
        if self.valid_wlen_range[1] == np.inf:
            min_freq = -mp.inf
        elif 1/self.valid_wlen_range[1] < mp.inf:
            min_freq = 1/self.valid_wlen_range[1]
        else:
            min_freq = -mp.inf
        if self.valid_wlen_range[0] == -np.inf:
            max_freq = mp.inf
        elif 1/self.valid_wlen_range[0] > -mp.inf:
            max_freq = 1/self.valid_wlen_range[0]
        else:
            max_freq = -mp.inf
        
        self.valid_freq_range = mp.FreqRange(min_freq, max_freq)
        
        self._logger_list = []
        self._have_logger = have_logger
        
        # self.__class__ = mp.Medium
    
    def _log_string(self, string):
        
        if self._have_logger and not string in self._logger_list: self._logger_list.append(string)
    
    def _log_print(self):
        
        print(self._logger_list)
        
    def epsilon(self,freq):
        """
        Returns the medium's permittivity tensor as a 3x3 Numpy array at the specified
        frequency `freq` which can be either a scalar, list, or Numpy array. In the case
        of a list/array of N frequency points, a Numpy array of size Nx3x3 is returned.
        """
        self._log_string("Outside eps")

        return self._get_eps(self.epsilon_diag, self.epsilon_offdiag, self.E_susceptibilities, self.D_conductivity_diag, self.D_conductivity_offdiag, freq)

    def mu(self,freq):
        """
        Returns the medium's permeability tensor as a 3x3 Numpy array at the specified
        frequency `freq` which can be either a scalar, list, or Numpy array. In the case
        of a list/array of N frequency points, a Numpy array of size Nx3x3 is returned.
        """
        self._log_string("Outside mu")
        
        return self._get_mu(self.mu_diag, self.mu_offdiag, self.H_susceptibilities, self.B_conductivity_diag, self.B_conductivity_offdiag, freq)
 
    def _get_epsmu(self, diag, offdiag, susceptibilities, conductivity_diag, conductivity_offdiag, freq):
        
        self._log_string("Inside epsmu")
        
        return self._get_eps(self, diag, offdiag, susceptibilities, conductivity_diag, conductivity_offdiag, freq)
 
    def _get_eps(self, diag, offdiag, susceptibilities, conductivity_diag, conductivity_offdiag, freq):
        
        self._log_string("Inside eps")
        
        # Clean the input
        if np.isscalar(freq):
            freqs = np.array(freq)[np.newaxis, np.newaxis, np.newaxis]
        else:
            freqs = np.squeeze(freq)
            freqs = freqs[:, np.newaxis, np.newaxis]

        # Check for values outside of allowed ranges
        if np.min(np.squeeze(freqs)) < self.valid_freq_range.min:
            raise ValueError('User specified frequency {} is below the Medium\'s limit, {}.'.format(np.min(np.squeeze(freqs)),self.valid_freq_range.min))
        if np.max(np.squeeze(freqs)) > self.valid_freq_range.max:
            raise ValueError('User specified frequency {} is above the Medium\'s limit, {}.'.format(np.max(np.squeeze(freqs)),self.valid_freq_range.max))

        # Initialize with instantaneous dielectric tensor
        epsmu = np.expand_dims(mp.Matrix(diag=diag,offdiag=offdiag),axis=0)
        
        # Use function for epsilon
        if freq==0:
            epsmu = epsmu * self.epsilon_function(mp.inf)
        else:
            epsmu = epsmu * self.epsilon_function(1/freq)

        # Account for conductivity term (only multiply if nonzero to avoid unnecessary complex numbers)
        conductivity = np.expand_dims(mp.Matrix(diag=conductivity_diag,offdiag=conductivity_offdiag),axis=0)
        if np.count_nonzero(conductivity) > 0:
            epsmu = (1 + 1j/freqs * conductivity) * epsmu

        # Convert list matrix to 3D numpy array size [freqs,3,3]
        return np.squeeze(epsmu)
    
    def _get_mu(self, diag, offdiag, susceptibilities, conductivity_diag, conductivity_offdiag, freq):
 
        self._log_string("Inside mu")
        
        # Clean the input
        if np.isscalar(freq):
            freqs = np.array(freq)[np.newaxis, np.newaxis, np.newaxis]
        else:
            freqs = np.squeeze(freq)
            freqs = freqs[:, np.newaxis, np.newaxis]

        # Check for values outside of allowed ranges
        if np.min(np.squeeze(freqs)) < self.valid_freq_range.min:
            raise ValueError('User specified frequency {} is below the Medium\'s limit, {}.'.format(np.min(np.squeeze(freqs)),self.valid_freq_range.min))
        if np.max(np.squeeze(freqs)) > self.valid_freq_range.max:
            raise ValueError('User specified frequency {} is above the Medium\'s limit, {}.'.format(np.max(np.squeeze(freqs)),self.valid_freq_range.max))

        # Initialize with instantaneous dielectric tensor
        epsmu = np.expand_dims(mp.Matrix(diag=diag,offdiag=offdiag),axis=0)
        
        # Use function for mu
        if freq==0:
            epsmu = epsmu * self.mu_function(mp.inf)
        else:
            epsmu = epsmu * self.mu_function(1/freq)

        # Account for conductivity term (only multiply if nonzero to avoid unnecessary complex numbers)
        conductivity = np.expand_dims(mp.Matrix(diag=conductivity_diag,offdiag=conductivity_offdiag),axis=0)
        if np.count_nonzero(conductivity) > 0:
            epsmu = (1 + 1j/freqs * conductivity) * epsmu

        # Convert list matrix to 3D numpy array size [freqs,3,3]
        return np.squeeze(epsmu)
    
#%% MEEP MEDIUM WITH INTERPOLER

class MediumFromFile(MediumFromFunction):
    
    """A Meep Medium subclass that loads experimental data and interpoles"""
    
    def __init__(self, 
                  epsilon_diag=mp.Vector3(1, 1, 1),
                  epsilon_offdiag=mp.Vector3(),
                  mu_diag=mp.Vector3(1, 1, 1),
                  mu_offdiag=mp.Vector3(),
                  E_susceptibilities=[],
                  H_susceptibilities=[],
                  E_chi2_diag=mp.Vector3(),
                  E_chi3_diag=mp.Vector3(),
                  H_chi2_diag=mp.Vector3(),
                  H_chi3_diag=mp.Vector3(),
                  D_conductivity_diag=mp.Vector3(),
                  D_conductivity_offdiag=mp.Vector3(),
                  B_conductivity_diag=mp.Vector3(),
                  B_conductivity_offdiag=mp.Vector3(),
                  epsilon=None,
                  index=None,
                  mu=None,
                  chi2=None,
                  chi3=None,
                  D_conductivity=None,
                  B_conductivity=None,
                  E_chi2=None,
                  E_chi3=None,
                  H_chi2=None,
                  H_chi3=None,
                  valid_freq_range=mp.FreqRange(min=-mp.inf, max=mp.inf),
                  material="Au",
                  paper="JC",
                  reference="RIinfo",
                  from_um_factor=1e-3,
                  have_logger=False):
        
        self.material = material
        self.paper = paper
        self.reference = reference
        self.from_um_factor = from_um_factor
        
        epsilon_function = epsilon_function_from_file(material=material, 
                                                      paper=paper, 
                                                      reference=reference, 
                                                      from_um_factor=from_um_factor)
        
        super().__init__(
                  epsilon_diag=epsilon_diag,
                  epsilon_offdiag=epsilon_offdiag,
                  mu_diag=mu_diag,
                  mu_offdiag=mu_offdiag,
                  E_susceptibilities=[],
                  H_susceptibilities=[],
                  E_chi2_diag=E_chi2_diag,
                  E_chi3_diag=E_chi3_diag,
                  H_chi2_diag=H_chi2_diag,
                  H_chi3_diag=H_chi3_diag,
                  D_conductivity_diag=D_conductivity_diag,
                  D_conductivity_offdiag=D_conductivity_offdiag,
                  B_conductivity_diag=B_conductivity_diag,
                  B_conductivity_offdiag=B_conductivity_offdiag,
                  epsilon=epsilon,
                  index=index,
                  mu=mu,
                  chi2=chi2,
                  chi3=chi3,
                  D_conductivity=D_conductivity,
                  B_conductivity=B_conductivity,
                  E_chi2=E_chi2,
                  E_chi3=E_chi3,
                  H_chi2=H_chi2,
                  H_chi3=H_chi3,
                  valid_freq_range=valid_freq_range,
                  epsilon_function=epsilon_function,
                  mu_function=lambda wlen:1,
                  have_logger=have_logger)
        
        if len(E_susceptibilities)>0 or len(H_susceptibilities)>0:
            raise ValueError("This class doesn't take susceptibilities!")

#%% MEEP SUSCEPTIBILITY THAT TAKES AN INTERPOLATION

class FromFileSusceptibility(mp.Susceptibility):
    """
    Specifies a susceptibility which is an interpolation of data loaded from a file.
    """
    def __init__(self, 
                  material="Au",
                  paper="JC",
                  reference="RIinfo",
                  from_um_factor=1e-3,
                  have_logger=False):
        
        super().__init__(sigma_diag=mp.Vector3(1,1,1), sigma_offdiag=mp.Vector3(), sigma=None)
        
        self.material = material
        self.paper = paper
        self.reference = reference
        self.from_um_factor = from_um_factor
        
        epsilon_function = epsilon_function_from_file(material=material, 
                                                      paper=paper, 
                                                      reference=reference, 
                                                      from_um_factor=from_um_factor)
        
        self.epsilon_function = epsilon_function
        
        self._have_logger = have_logger
        self._logger_list = []

    def _log_string(self, string):
        
        if self._have_logger and not string in self._logger_list: self._logger_list.append(string)
    
    def _log_print(self):
        
        print(self._logger_list)

    def eval_susceptibility(self,freq):
        self._log_string("Called eval susceptibility")
        sigma = np.expand_dims(mp.Matrix(diag=self.sigma_diag,offdiag=self.sigma_offdiag),axis=0)
        return self.epsilon_function(1/freq) * sigma
        
class MediumFromFileSusceptibility(mp.Medium):
    
    """A Meep Medium subclass that loads experimental data and interpoles"""
    
    def __init__(self, 
                  epsilon_diag=mp.Vector3(0,0,0),
                  epsilon_offdiag=mp.Vector3(),
                  mu_diag=mp.Vector3(1,1,1),
                  mu_offdiag=mp.Vector3(),
                  E_susceptibilities=[],
                  H_susceptibilities=[],
                  E_chi2_diag=mp.Vector3(),
                  E_chi3_diag=mp.Vector3(),
                  H_chi2_diag=mp.Vector3(),
                  H_chi3_diag=mp.Vector3(),
                  D_conductivity_diag=mp.Vector3(),
                  D_conductivity_offdiag=mp.Vector3(),
                  B_conductivity_diag=mp.Vector3(),
                  B_conductivity_offdiag=mp.Vector3(),
                  epsilon=None,
                  index=None,
                  mu=None,
                  chi2=None,
                  chi3=None,
                  D_conductivity=None,
                  B_conductivity=None,
                  E_chi2=None,
                  E_chi3=None,
                  H_chi2=None,
                  H_chi3=None,
                  valid_freq_range=mp.FreqRange(min=-mp.inf, max=mp.inf),
                  material="Au",
                  paper="JC",
                  reference="RIinfo",
                  from_um_factor=1e-3,
                  have_logger=False):
        
        if len(E_susceptibilities)>0 or len(H_susceptibilities)>0:
            raise ValueError("This class doesn't take external susceptibilities!")
        
        self.material = material
        self.paper = paper
        self.reference = reference
        self.from_um_factor = from_um_factor
        
        interpolated_susceptibility = FromFileSusceptibility(
            material=material, 
            paper=paper, 
            reference=reference, 
            from_um_factor=from_um_factor,
            have_logger=have_logger)
        
        try:
            eps_wlen_range = interpolated_susceptibility.epsilon_function._wlen_range_
        except:
            eps_wlen_range = np.array([-mp.inf, mp.inf])
        
        self.valid_wlen_range = eps_wlen_range
        
        if self.valid_wlen_range[1] == np.inf:
            min_freq = -mp.inf
        elif 1/self.valid_wlen_range[1] < mp.inf:
            min_freq = 1/self.valid_wlen_range[1]
        else:
            min_freq = -mp.inf
        if self.valid_wlen_range[0] == -np.inf:
            max_freq = mp.inf
        elif 1/self.valid_wlen_range[0] > -mp.inf:
            max_freq = 1/self.valid_wlen_range[0]
        else:
            max_freq = -mp.inf

        valid_freq_range = mp.FreqRange(min_freq, max_freq)

        super().__init__(
                  epsilon_diag=epsilon_diag,
                  epsilon_offdiag=epsilon_offdiag,
                  mu_diag=mu_diag,
                  mu_offdiag=mu_offdiag,
                  E_susceptibilities=[interpolated_susceptibility],
                  H_susceptibilities=[],
                  E_chi2_diag=E_chi2_diag,
                  E_chi3_diag=E_chi3_diag,
                  H_chi2_diag=H_chi2_diag,
                  H_chi3_diag=H_chi3_diag,
                  D_conductivity_diag=D_conductivity_diag,
                  D_conductivity_offdiag=D_conductivity_offdiag,
                  B_conductivity_diag=B_conductivity_diag,
                  B_conductivity_offdiag=B_conductivity_offdiag,
                  epsilon=epsilon,
                  index=index,
                  mu=mu,
                  chi2=chi2,
                  chi3=chi3,
                  D_conductivity=D_conductivity,
                  B_conductivity=B_conductivity,
                  E_chi2=E_chi2,
                  E_chi3=E_chi3,
                  H_chi2=H_chi2,
                  H_chi3=H_chi3,
                  valid_freq_range=valid_freq_range)

