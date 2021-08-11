#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 13:17:36 2021

@author: vall
"""

from copy import deepcopy
import numpy as np
import meep as mp
import v_materials as vmt

#%%

class SimpleNanoparticle:
    
    """Representation of a single simple plasmonic nanoparticle.
    
    This class represents a single simple plasmonic nanoparticle. It can either 
    be a sphere, a spheroid or a cylindrical rod. It can be made of gold or 
    silver, via a Drude-Lorentz fit of Jhonson & Christy bulk data or Rakic's 
    model.
    
    Parameters
    ----------
    r=51.5 : float, optional
        Radius of a spherical or cylindrical nanoparticle, expressed in nm.
    d=103 : float, optional
        Diameter of a spherical or cylindrical nanoparticle, expressed in nm.
    h=None : float, optional
        Height of a cylindrical nanoparticle, expressed in nm.
    a=None : float, optional
        One of the characteristic size parameters of an ellipsoidal 
        nanoparticle, expressed in nm.
    b=None : float, optional
        One of the characteristic size parameters of an ellipsoidal 
        nanoparticle, expressed in nm.
    c=None : float, optional
        One of the characteristic size parameters of an ellipsoidal 
        nanoparticle, expressed in nm.
    material="Au" : string, optional
        Plasmonic metal. Currently supported options: "Au" for gold, "Ag" for 
        silver.
    paper="JC" : string, optional
        Plasmonic metal's material source. Currently supported options: "R" for 
        Rakic's data for Drude-Lorentz model and "JC" for a fit on Jhonson & 
        Christy's experimental bulk data.
    standing=True : bool, optional
        Nanoparticle's preferred orientation in the X axis, normal to the 
        surface of the substrate, if there's one. When set to true, the NP is 
        standing, with it's principal axis alligned with the surface's normal 
        direction.
    transversal=False : bool, optional.
        Nanoparticle's preferred orientation in the ZY plane, parallel to the 
        surface of the substrate, if there's one. When set to false, the NP 
        mostly lives in the incidence plane, meaning its principal axis or its 
        next more prominent axis belongs to the incidence plane. When set to 
        true, the NP has a mostly transversal orientation, meaning its 
        principal axis or its next more prominent axis is alligned with the 
        incidence plane's normal direction.
        
    Attributes
    ----------
    shape : string
        Nanoparticle's shape, which can be a spherical nanoparticle, an
        ellipsoidal nanoparticle or a cylindrical nanorod.
    size : mp.Vector3
        Nanoparticle's dimensions if it's encapsulated inside a box, all 
        expressed in nm; i.e. (100, 50, 50) could be the size of a cylindrical 
        nano rod with a 100 nm  height and a 50 nm diamenter, standing normal 
        to a sustrate and parallel to a planewave with normal incidence.
        
    Methods
    -------
    get_info()
        Prints information on the NP.
    get_geometry(from_um_factor)
        Returns the Meep NP instance, scaled using from_um_factor, which 
        specifies Meep's length unit in um, i.e. from_um_factor=10e-3 stands 
        for a Meep length unit of 0.01 um, which is equivalent to 10 nm.
    
    See Also
    --------
    mp.Vector3
    mp.Sphere
    mp.Ellipsoid
    mp.Cylinder
    mp.Block
    vmt.import_material
    """
    
    def __init__(self, 
                 r=51.5, d=103, h=None, a=None, b=None, c=None,
                 material="Au", paper="R",
                 standing=True, transversal=False):
        
        # Nanoparticle's dimensions
        self._r = r # sphere, cylinder
        self._d = d # sphere, cylinder
        self._h = h # cylinder
        self._a = a # spheroid
        self._b = b # spheroid
        self._c = c # spheroid
        
        # Nanoparticle's material
        self._material = material
        self._paper = paper
        self._material_options = ["Au", "Ag"]
        self._paper_options = ["R", "JC"]
        
        # Nanoparticle's orientation
        self._standing = standing
        self._transversal = transversal
        self._standing_direction = "Z"
        self._transversal_direction = "Y"
        
        # Nanoparticle's structure
        self._size = None
        self._shape = None
        self._structure = None
        self._set_structure()
    
    @property
    def r(self):
        return self._r
    
    @r.setter
    def r(self, value):
        self._r = value
        self._d = 2*value
        self._a = None
        self._b = None
        self._c = None
        self._set_structure()

    @property
    def d(self):
        return self._d
    
    @d.setter
    def d(self, value):
        self._r = value / 2
        self._d = value
        self._a = None
        self._b = None
        self._c = None
        self._set_structure()
        
    @property
    def h(self):
        return self._h
    
    @h.setter
    def h(self, value):
        self._h = value
        if self._r is None:
            self._r = value
            raise Warning("Radius wasn't set, so it's now equal to height.")
        self._a = None
        self._b = None
        self._c = None
        self._set_structure()
    
    @property
    def a(self):
        return self._a
    
    @a.setter
    def a(self, value):
        self._a = value
        if self._b is None:
            self._b = value
        if self._c is None:
            self._c = value
        self._r = None
        self._h = None
        self._set_structure()
        
    @property
    def b(self):
        return self._a
    
    @b.setter
    def b(self, value):
        if self._a is None:
            self._a = value
        self._b = value
        if self._c is None:
            self._c = value
        self._r = None
        self._h = None
        self._set_structure()
        
    @property
    def c(self):
        return self._c
    
    @c.setter
    def c(self, value):
        if self._a is None:
            self._a = value
        if self._b is None:
            self._b = value
        self._c = value
        self._r = None
        self._h = None
        self._set_structure()
    
    @property
    def material(self):
        return self._material
    
    @material.setter
    def material(self, value):
        if value in self._material_options:
            self._material = value
        else:
            raise ValueError(f"Material not recognized.\
                             Options are: f{self._material_options}")
    
    @property
    def paper(self):
        return self._paper
    
    @paper.setter
    def paper(self, value):
        if value in self._paper_options:
            self._paper = value
        else:
            raise ValueError(f"Material not recognized.\
                             Options are: f{self._material_options}")
    
    @property
    def standing(self):
        return self._standing
    
    @standing.setter
    def standing(self, value):
        if isinstance(value, bool):
            self._standing = value
            self._set_structure()
        else:
            raise ValueError("This property must be boolean.")
            
    @property
    def transversal(self):
        return self._transversal
    
    @transversal.setter
    def transversal(self, value):
        if isinstance(value, bool):
            self._transversal = value
            self._set_structure()
        else:
            raise ValueError("This property must be boolean.")

    @property
    def shape(self):
        return self._shape
    
    @property
    def size(self):
        return self._size
            
    @shape.setter
    @size.setter
    def negator(self, value):
        raise ValueError("This property cannot be set manually! Run geometry.")
    
    def _set_structure(self):
        
        if self.r is not None and self.h is None:
            self._shape = "spherical nanoparticle"
            self._size = mp.Vector3( self.d, self.d, self.d )
            self._structure = mp.Sphere(radius=self.r)
        
        elif self.a is not None:
            self._shape = "ellipsoidal nanoparticle"
            minor_to_major = [self.a, self.b, self.c]
            minor_to_major.sort()
            if self.standing:
                # Mostly in x direction: parallel to surface's normal
                size_x = minor_to_major[-1]
                minor_to_major.remove(size_x)
                if not self.transversal:
                    # Next, mostly in z direction: included in incidence plane.
                    size_z = minor_to_major[-1]
                    size_y = minor_to_major[0]
                else:
                    # Next, mostly in y direction: transversal to incidence plane.
                    size_y = minor_to_major[-1]
                    size_z = minor_to_major[0]                    
            else:
                # Mostly in zy plane: parallel to the surface.
                if not self.transversal:
                    # Mostly in z direction: included in incidence plane.
                    size_z = minor_to_major[-1]
                    minor_to_major.remove(size_y)
                    size_x = minor_to_major[-1]
                    size_y = minor_to_major[0]
                else:
                    # Mostly in y direction: transversal to incidence plane.
                    size_y = minor_to_major[-1]
                    minor_to_major.remove(self.size_x)
                    size_x = minor_to_major[-1]
                    size_z = minor_to_major[0]
            self._size = mp.Vector3( size_x, size_y, size_z )
            self._structure = mp.Ellipsoid(size=self.size)
    
        elif self.h is not None:
            self._shape = "cylindrical nanorod"
            if self.standing:
                # Principal axis in x direction: parallel to surface's normal
                size_x = self.h
                size_z = self.d
                size_y = self.d
                orientation = mp.Vector3(1,0,0)
            else:
                if not self.transversal:
                    # Principal axis in z direction: included in incidence plane.
                    size_z = self.h
                    size_x = self.d
                    size_y = self.d
                    orientation = mp.Vector3(0,0,1)
                else:
                    # Principal axis in y direction: transversal to incidence plane.
                    size_y = self.h
                    size_x = self.d
                    size_z = self.d
                    orientation = mp.Vector3(0,1,0)
            self._size = mp.Vector3( size_x, size_y, size_z )
            self._structure =  mp.Cylinder(radius=self.r, height=self.h,
                               axis=orientation)
    
    def get_size(self, from_um_factor):
        
        return self.size / (1e3 * from_um_factor) # Meep units
    
    def get_geometry(self, from_um_factor):
                
        geometry = deepcopy(self._structure)
        if self.r is not None and self.h is None:
            geometry.radius = geometry.radius / (1e3 * from_um_factor) # Meep Units
        elif self.a is not None:
            geometry.size = geometry.size / ( 1e3 * from_um_factor ) # Meep Units
        elif self.h is not None:
            geometry.radius = geometry.radius / (1e3 * from_um_factor) # Meep Units
            geometry.height = geometry.height / (1e3 * from_um_factor) # Meep Units
        
        medium = vmt.import_medium(self.material,
                                   paper=self.paper,
                                   from_um_factor=from_um_factor) # Meep units
        geometry.material = medium
        
        return geometry # Meep units
    
    def get_info(self):
        
        print(f"{self.material} {self.paper} {self.shape}")
        orientation = ""
        if self.standing:
            orientation += "standing, "
        else:
            orientation += "not standing, "
        if self.transversal:
            orientation += "transversal"
        else:
            orientation += "not transversal"
        print(orientation)
        self._structure.info()
        print(f"size = ({str(list(self.size))[1:-1]})")
        print("all expressed in nm")
        print("with x normal to possible sustrate and zx incidence plane")
        
#%%

class NoNanoparticle:
    
    """Representation of what happens if there is no nanoparticle.
    
    This class allows to use plasmonic nanoparticles routines without the 
    nanoparticle.
    
    See Also
    --------
    SimpleNanoparticle
    """
    
    def __init__(self):
                
        self.size = mp.Vector3()
        
        return
            
    def get_geometry(self, from_um_factor):
                
        return # Meep units
    
    def get_info(self):
        
        print("No nanoparticle")
        
#%%

class PlanePulseSource:
    
    """Representation of a planewave pulse with Gaussian frequency profile
    
    For the time being, only wavefront parallel to a side of the cell box is 
    available
    """
    
    def __init__(self,
                 wlen_range=np.array([450, 600]),
                 polarization=mp.Ez,
                 normal=mp.Vector3(1,0,0),
                 side=-1,
                 cutoff=3.2):
        
        self._wlen_range = wlen_range
        self._polarization = polarization
        self._normal = normal
        self._side = side
        self._cutoff = cutoff
    
    @property
    def wlen_range(self):
        return self._wlen_range
    
    @wlen_range.setter
    def wlen_range(self, value):
        try:
            if len(value) == 2:
                self._wlen_range = np.array(*value)
            else:
                raise ValueError("Hey! Must be iterable of length 2.")
        except:
            raise ValueError("Hey! Must be iterable of length 2.")
        
    @property
    def polarization(self):
        return self._polarization
    
    @polarization.setter
    def polarization(self, value):
        if isinstance(value, int):
            if value < 5:
                self._polarization = value
            else:
                raise ValueError("Hey! If int, must be either \
                                 0 (EX), 1 (EY), 4 (EZ), \
                                 5 (HX), 6 (HY), 9 (HZ)")
        elif isinstance(value, str):
            if "H" in value.upper():
                if "x" in value.lower():
                    self._polarization = mp.Hx
                elif "y" in value.lower():
                    self._polarization = mp.Hy
                elif "z" in value.lower():
                    self._polarization = mp.Hz
                else:
                    raise ValueError("Hey! If string, must have length 0 and \
                                     refer to one of the axis")
            else:
                if "x" in value.lower():
                    self._polarization = mp.Ex
                elif "y" in value.lower():
                    self._polarization = mp.Ey
                elif "z" in value.lower():
                    self._polarization = mp.Ez
                else:
                    raise ValueError("Hey! If string, must have length 0 and \
                                     refer to one of the axis")
        else:
            raise ValueError("Hey! Must be an integer; i.e. mp.Ez; \
                             or a string referring to one of the axis")
              
    @property
    def normal(self):
        return self._normal
    
    @normal.setter
    def normal(self, value):            
        if isinstance(value, int):
            if value < 2:
                self._normal = value
            else:
                raise ValueError("Hey! If int, must be either \
                                 0 (X), 1 (Y) or 2 (Z)")
        elif isinstance(value, str):
            if "x" == value.lower():
                self._normal = mp.X
            elif "y" == value.lower():
                self._normal = mp.Y
            elif "z" == value.lower():
                self._normal = mp.Z
            else:
                raise ValueError("Hey! If string, must have length 0 and \
                                 refer to one of the axis")
        else:
            raise ValueError("Hey! Must be an integer; i.e. mp.Z; \
                             or a string referring to one of the axis")
    
    @property
    def side(self):
        return self._side
    
    @side.setter
    def side(self, value):
        if abs(value) == 1:
            self._side = int(value)
        else:
            raise ValueError("Hey! Must be either 1 or -1.")
    
    @property
    def cutoff(self):
        return self._cutoff
    
    @cutoff.setter
    def cutoff(self, value):
        self._cutoff = value    
    
    def get_wavelength(self, from_um_factor):
        
        return self.wlen_range / (1e3 * from_um_factor) # Meep Units, from lowest to highest
    
    def get_frequency(self, from_um_factor):
        
        return  1 / self.get_wavelength(from_um_factor) # Meep Units, from highest to lowest
        
    def get_center(self, cell, from_um_factor):
        
        cell_size = cell.get_cell_size(from_um_factor) # Meep Units
        pml_width = cell.get_pml_width(from_um_factor) # Meep Units
        
        if self.normal == mp.X:
            center = self.side * (0.5 * cell_size.x - pml_width) * mp.Vector3(x=1)
        elif self.normal == mp.Y:
            center = self.side * (0.5 * cell_size.y - pml_width) * mp.Vector3(y=1)
        elif self.normal == mp.Z:
            center = self.side * (0.5 * cell_size.z - pml_width) * mp.Vector3(z=1)
            
        return center # Meep Units
        
    def get_size(self, cell, from_um_factor):
        
        cell_size = cell.get_cell_size(from_um_factor) # Meep Units
        
        if self.normal == mp.X:
            size = cell_size * mp.Vector3(0,1,1)
        elif self.normal == mp.Y:
            size = cell_size * mp.Vector3(1,0,1)
        elif self.normal == mp.Z:
            size = cell_size * mp.Vector3(1,1,0)
            
        return size # Meep Units
    
    def get_source(self, cell, from_um_factor):
            
        freq_range = self.get_frequency(from_um_factor) # Meep Units
        freq_center = np.mean(freq_range) # Meep Units
        freq_width = max(freq_range) - min(freq_range) # Meep Units
        
        gaussian_source = mp.GaussianSource(freq_center,
                                            fwidth=freq_width,
                                            is_integrated=True,
                                            cutoff=self.cutoff)
        
        return mp.Source(gaussian_source,
                         center=self.get_center(cell, from_um_factor),
                         size=self.get_size(cell, from_um_factor),
                         component=self.polarization)
    
#%%

class NoSource:
    
    """Representation of what happens if there is no source.
    
    See Also
    --------
    PlanePulseSource
    """
    
    def __init__(self):
            
        self.wlen_range = np.array([350, 800])
        self.polarization = mp.Ez
        self.normal = mp.Vector3(1,0,0)
        self.side = -1
        self.cutoff = 3.2
    
    def get_wavelength(self, from_um_factor):
        
        return np.array([250, 850]) / (1e3 * from_um_factor) # Meep Units, visible spectrum
    
    def get_frequency(self, from_um_factor):
        
        return  1 / self.get_wavelength(from_um_factor) # Meep Units, from highest to lowest
        
    def get_center(self, cell, from_um_factor):
        
        return mp.Vector3() # Meep Units
        
    def get_size(self, cell, from_um_factor):
        
        return mp.Vector3() # Meep Units
    
    def get_source(self, cell, from_um_factor):
        
        return
    
#%%

class Surroundings:
    
    """Representation of the sustrate and the surrounding medium"""
    
    def __init__(self,
                 nanoparticle=NoNanoparticle(),
                 submerged_index=0,
                 surface_index=0,
                 overlap=0,
                 displacement=0,
                 normal=mp.X,
                 side=-1):
        
        self._nanoparticle = None
        
        if submerged_index != 0:
            self.submerged_index = submerged_index
        else:
            self.submerged_index = 1
        
        if surface_index != 0:
            self.surface_index = surface_index
        else:
            self.surface_index = submerged_index
        
        if overlap != 0 and displacement != 0:
            raise ValueError("Either overlap or displacement is accepted. Not both!")
        elif overlap != 0:
            self._overlap = overlap
            self._displacement = overlap - self._nanoparticle.size.x / 2
        else:
            self._displacement = displacement
            self._overlap = displacement + self._nanoparticle.size.x / 2
        
        self._normal = normal
        self._side = 1
       
       # ESTOY CON NANOPARTICLE SIZE 
       
    @property
    def overlap(self):
        return self._overlap
    
    @overlap.setter
    def overlap(self, value):
        self._overlap = value
        self._displacement = value - self._nanoparticle.size.x / 2
        
    @property
    def displacement(self):
        return self._displacement
    
    @displacement.setter
    def displacement(self, value):
        self._displacement = value
        self._overlap = value + self._nanoparticle.size.x / 2
        
    @property
    def normal(self):
        return self._normal
    
    @normal.setter
    def normal(self, value):            
        if isinstance(value, int):
            if value < 2:
                self._normal = value
            else:
                raise ValueError("Hey! If int, must be either \
                                 0 (X), 1 (Y) or 2 (Z)")
        elif isinstance(value, str):
            if "x" == value.lower():
                self._normal = mp.X
            elif "y" == value.lower():
                self._normal = mp.Y
            elif "z" == value.lower():
                self._normal = mp.Z
            else:
                raise ValueError("Hey! If string, must have length 0 and \
                                 refer to one of the axis")
        else:
            raise ValueError("Hey! Must be an integer; i.e. mp.Z; \
                             or a string referring to one of the axis")
                             
    @property
    def side(self):
        return self._side
    
    @side.setter
    def side(self, value):
        if abs(value) == 1:
            self._side = int(value)
        else:
            raise ValueError("Hey! Must be either 1 or -1.")
        
    def get_center(self, cell, from_um_factor):
        
        cell_size = cell.get_cell_size(from_um_factor) # Meep Units
        nanoparticle_size = self._nanoparticle.get_size() # Meep Units
        overlap = self.overlap / (1e3 * from_um_factor) # Meep Units
        
        if self.normal == mp.X:
            surface_center = nanoparticle_size.x / 2 - overlap / 2 + cell_size.x / 4
            surface_center = self.side * surface_center * mp.Vector3(x=1)
        elif self.normal == mp.Y:
            surface_center = nanoparticle_size.y / 2 - overlap / 2 + cell_size.y / 4
            surface_center = self.side * surface_center * mp.Vector3(y=1)
        elif self.normal == mp.Z:
            surface_center = nanoparticle_size.z / 2 - overlap / 2 + cell_size.z / 4
            surface_center = self.side * surface_center * mp.Vector3(z=1)
            
        return surface_center # Meep Units
        
    def get_size(self, cell, from_um_factor):
        
        cell_size = cell.get_cell_size(from_um_factor) # Meep Units
        nanoparticle_size = self._nanoparticle.get_size() # Meep Units
        overlap = self.overlap / (1e3 * from_um_factor) # Meep Units
        
        if self.normal == mp.X:
            surface_size = cell_size.x / 2 - nanoparticle_size.x + overlap
            surface_size = surface_size * mp.Vector3(1,0,0) + cell_size * mp.Vector3(0,1,1)
        elif self.normal == mp.Y:
            surface_size = cell_size.y / 2 - nanoparticle_size.y + overlap
            surface_size = surface_size * mp.Vector3(0,1,0) + cell_size * mp.Vector3(1,0,1)
        elif self.normal == mp.Z:
            surface_size = cell_size.z / 2 - nanoparticle_size.z + overlap
            surface_size = surface_size * mp.Vector3(0,0,1) + cell_size * mp.Vector3(1,1,0)
            
        return surface_size # Meep Units
    
    def get_medium(self):
        
        return mp.Medium(index=self.submerged_index)
    
    def get_geometry(self, cell, from_um_factor):
        
        if self.surface_index != self.submerged_index:
        
            surface_center = self.get_center(cell, from_um_factor) # Meep Units
            surface_size = self.get_size(cell, from_um_factor) # Meep Units
            
            return mp.Block(material=mp.Medium(index=self.surface_index),
                            center=surface_center,
                            size=surface_size)
        
        else: return
    
#%%

class SingleParticleCell:
    
    """Representation of a simulation cell centered on a single nanoparticle"""
    
    def __init__(self,
                 nanoparticle=NoNanoparticle(),
                 surroundings=Surroundings(),
                 source=NoSource(),
                 pml_wlen_factor=0.38,
                 empty_r_factor=0.5,
                 flux_padd_points=5):
        
        self._nanoparticle = nanoparticle
        self._surroundings = surroundings
        self._source = source
        self._pml_wlen_factor = pml_wlen_factor
        self._empty_r_factor = empty_r_factor
        self._flux_padd_points = flux_padd_points
        
    
    @property
    def pml_wlen_factor(self):
        return self._pml_wlen_factor
    
    @pml_wlen_factor.setter
    def pml_wlen_factor(self, value):
        self._pml_wlen_factor = value
    
    @property
    def empty_r_factor(self):
        return self._empty_r_factor
    
    @empty_r_factor.setter
    def empty_r_factor(self, value):
        self._empty_r_factor = value
        
    @property
    def flux_padd_points(self):
        return self._flux_padd_points
    
    @flux_padd_points.setter
    def flux_padd_points(self, value):
        self._flux_padd_points = value
        
    def get_pml_width(self, from_um_factor):
        wavelength = self._source.get_wavelength(from_um_factor)
        try:
            max_wavelength = max(wavelength) / (1e3 * from_um_factor) # Meep units
        except:
            max_wavelength = wavelength / (1e3 * from_um_factor) # Meep units
        return self.pml_wlen_factor * max_wavelength # Meep units
    
    def get_empty_width(self, from_um_factor):
        
        nanoparticle_size = self._nanoparticle.get_size() # Meep units
        return self.empty_r_factor * max(nanoparticle_size) # Meep units

    def get_cell_size(self, from_um_factor):
        
        pml = self.get_pml_width(from_um_factor) * mp.Vector3(1,1,1) # Meep units
        empty = self.get_empty_width(from_um_factor) * mp.Vector3(1,1,1) # Meep units
        nanoparticle_size = self._nanoparticle.get_size() # Meep units
        return 2 * (pml + empty + nanoparticle_size) # Meep units
    
    # def get_pml(self, from_um_factor):