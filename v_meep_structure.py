#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 13:17:36 2021

@author: vall
"""

from copy import deepcopy
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
    
    def get_geometry(self, from_um_factor):
                
        geometry = deepcopy(self._structure)
        if self.r is not None and self.h is None:
            geometry.radius = geometry.radius / (1e3 * from_um_factor)
        elif self.a is not None:
            geometry.size = geometry.size / ( 1e3 * from_um_factor )
        elif self.h is not None:
            geometry.radius = geometry.radius / (1e3 * from_um_factor)
            geometry.height = geometry.height / (1e3 * from_um_factor)
        
        medium = vmt.import_medium(self.material,
                                   paper=self.paper,
                                   from_um_factor=from_um_factor)
        geometry.material = medium
        
        return geometry
    
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
        
    