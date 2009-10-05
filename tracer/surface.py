# Define some basic surfaces for use with the ray tracer. From this minimal 
# hierarchy other surfaces should be derived to implement actual geometric
# operations.
#
# References:
# [1] John J. Craig, Introduction to Robotics, 3rd ed., 2005. 

# All surfaces are grey.

import numpy as N
from has_frame import HasFrame

class Surface(HasFrame):
    """
    Defines the base of surfaces that interact with rays.
    """
    def __init__(self, geometry, optics, location=None, rotation=None):
        """
        Arguments:
        geometry - a GeometryManager object responsible for finding ray 
            intersections with the surface.
        optics - an OpticsManager object responsible for reflections and
            refractions.
        location, rotation - passed directly to the HasFrame constructor.
        """
        HasFrame.__init__(self, location, rotation)
        self._geom = geometry
        self._opt = optics

    def set_parent_object(self, object):
        """Describes which object the surface is in """
        self.parent_object = object

    def set_inner_n(self, n):
        """Arbitrarily describes one side of the surface as the inner side and the
        other a the outer side.  Then, assigns a refractive index to one of these sides."""
        self._inner_n = n

    def set_outer_n(self, n):
        self._outer_n = n

    def get_inner_n(self):
        return self._inner_n

    def get_outer_n(self):
        return self._outer_n
    
    def get_ref_index(self, n):
        """                                                                                  
        Determines which refractive index to use based on the refractive index               
        the ray is currently travelling through.
         
        Arguments: 
        n - an array of the refractive indices of the materials each of the
            rays in a ray bundle is travelling through.
        
        Returns:
        An array of length(n) with the index to use for each ray.
        """
        return N.where(n == self.get_inner_n(), \
            self.get_outer_n(), self.get_inner_n())
    
    def register_incoming(self, ray_bundle):
        """
        Records the incoming ray bundle, and uses the geometry manager to
        return the parametric positions of intersection with the surface along
        the ray.
        
        Arguments:
        ray_bundle - a RayBundle object with at-least its vertices and
            directions specified.
        
        Returns
        A 1D array with the parametric position of intersection along each of
            the rays. Rays that missed the surface return +infinity.
        """
        self._current_bundle = ray_bundle
        return self._geom.find_intersections(frame, ray_bundle)
    
class UniformSurface(Surface):
    """Implements an abstract surface whose material properties are independent of
    location.
    Currently only absorptivity is tracked.
    Private attributes:
    self.mirror - whether the surface is mirrored or not
    self._absorp - the absorptivity of the surface
    """
    def __init__(self,  location=None,  rotation=None,  absorptivity=0., mirror=True):
        Surface.__init__(self,  location,  rotation)
        self.set_absorptivity(absorptivity)
        self.mirror = mirror
    
    def get_absorptivity(self):
        return self._absorpt
    
    def set_absorptivity(self,  absorptivity):
        if  not 0 <= absorptivity <= 1:
            raise ValueError("Absorptivity must be in [0,1]")
        self._absorpt = absorptivity
