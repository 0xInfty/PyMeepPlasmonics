#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The `v_meep` module contains tools used to complement Meep.

Some of its most useful tools are...

verify_stability_freq_res : function
    Verifies stability via temporal resolution and resonant frequencies.
verify_stability_dim_index : function
    Verifies stability via dimensions, refractive index and Courant factor.
MeepUnitsManager : class
    Incomplete class to manage units in Meep.

It's widely based on Meep Materials Library.

@author: vall
"""

import meep as mp
import numpy as np
import os
import v_save as vs
import v_utilities as vu

sysname = vs.get_sys_name()
syshome = vs.get_sys_home()
home = vs.get_home()

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

def save_midflux(sim, box_x1, box_x2, box_y1, box_y2, box_z1, box_z2, params, path):
    
    
    dir_file = os.path.join(home, "FluxData/FluxDataDirectory.txt")
    dir_backup = os.path.join(home, f"FluxData/FluxDataDir{sysname}Backup.txt")
    new_flux_path = vs.new_dir(vs.datetime_dir(os.path.join(home, "FluxData/MidFlux")))

    os.chdir(new_flux_path)
    sim.save_flux("MidFluxX1", box_x1)
    sim.save_flux("MidFluxX2", box_x2)
    sim.save_flux("MidFluxY1", box_y1)
    sim.save_flux("MidFluxY2", box_y2)
    sim.save_flux("MidFluxZ1", box_z1)
    sim.save_flux("MidFluxZ2", box_z2)
    os.chdir(syshome)
        
    database = vs.retrieve_footer(dir_file)
    vs.savetxt(dir_backup, np.array([]), footer=database, overwrite=True)
    key_params = ["from_um_factor", "resolution", "courant", 
                 "wlen_range", "cutoff", "nfreq", 
                 "submerged_index", "surface_index", "displacement",
                 "cell_width", "pml_width", "source_center",
                 "until_after_sources", 
                 "parallel", "n_processes"]
    
    database["flux_path"].append( os.path.split(new_flux_path)[-1] )
    database["path"].append(path)
    for key in key_params:
        try:
            if isinstance(params[key], np.ndarray):
                database[key].append(list(params[key]))
            else:
                database[key].append(params[key])
        except:
            raise ValueError(f"Missing key parameter: {key}")
    
    vs.savetxt(dir_file, np.array([]), footer=database, overwrite=True)
    
    return new_flux_path

#%%

def check_midflux(params):
    
    dir_file = os.path.join(home, "FluxData/FluxDataDirectory.txt")
    
    database = vs.retrieve_footer(dir_file)
    key_params = ["from_um_factor", "resolution", "courant", 
                 "wlen_range", "cutoff", "nfreq", 
                 "submerged_index", "surface_index", "displacement",
                 "cell_width", "pml_width", "source_center",
                 "until_after_sources", 
                 "parallel", "n_processes"]
    
    database_array = []
    for key in key_params:
        if isinstance(database[key][0], bool):
            aux_data = [int(data) for data in database[key]]
            database_array.append(aux_data)
        else:
            try:
                if len(list(database[key][0])) > 1:
                    for i in range( len(list( database[key][0] )) ):
                        aux_data = [data[i] for data in database[key]]
                        database_array.append(aux_data)
                else:
                    database_array.append(database[key])
            except:
                database_array.append(database[key])
    database_array = np.array(database_array)
    
    desired_array = []
    for key in key_params:
        if isinstance(params[key], bool):
            desired_array.append(int(params[key]))
        else:
            try:
                if len(list(params[key])) > 1:
                    for i in range( len(list( params[key] )) ):
                        desired_array.append(params[key][i])
                else:
                    desired_array.append(params[key])
            except:
                desired_array.append(params[key])
    desired_array = np.array(desired_array)
    
    boolean_array = []
    for array in database_array.T:
        boolean_array.append( np.all( array - desired_array.T == np.zeros(desired_array.T.shape) ) )
    index = [i for i, boolean in enumerate(boolean_array) if boolean]
    
    if len(index) == 0:
        print("No coincidences where found at the midflux database!")
    elif len(index) == 1:
        right_index = index[0]
        print(f"You could use data from '{database['path'][right_index]}'")
    else:
        right_index = index[0]
        print("More than one coincidence was found at the midflux database!")
        print(f"You could use data from '{database['path'][right_index]}'")
        
    try:
        flux_path = os.path.join(home, "FluxData", database['flux_path'][right_index])
    except:
        flux_path = None
    
    return flux_path

#%%

def load_midflux(sim, box_x1, box_x2, box_y1, box_y2, box_z1, box_z2, flux_path):
    
    print(f"Loading flux from '{flux_path}'")
    
    os.chdir(flux_path)
    sim.load_flux("MidFluxX1", box_x1)
    sim.load_flux("MidFluxX2", box_x2)
    sim.load_flux("MidFluxY1", box_y1)
    sim.load_flux("MidFluxY2", box_y2)
    sim.load_flux("MidFluxZ1", box_z1)
    sim.load_flux("MidFluxZ2", box_z2)
    os.chdir(syshome)
    
    return

#%%

def parallel_assign(process_number, process_total_number, parallel=True):
    
    if parallel and process_total_number > 1:
        if process_number == 0:
            return mp.am_master()
        else:
            return mp.my_rank() == 1
    else:
        return True

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
