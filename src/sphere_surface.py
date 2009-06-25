# Implements spherical mirrored surface 

from surface import UniformSurface
import optics
from ray_bundle import RayBundle
from boundary_shape import BoundarySphere
import numpy as N

class SphereSurface(UniformSurface):
    """
    Implements the geometry of a spherical mirror surface.  
    """
    def __init__(self, center=None, absorptivity=0., radius=1., boundary=None):
        """
        Arguments:
        location of center, rotation, absorptivity - passed along to the base class.
        boundary - boundary shape defining the surface
        Private attributes:
        _rad - radius of the sphere
        _center - center of the sphere
        _boundary - boundary shape defining the surface
        """
        UniformSurface.__init__(self, center, None,  absorptivity)
        self.set_radius(radius)
        self._center = center
        self._boundary = boundary

    def get_radius(self):
        return self._rad

    def get_center(self):
        return self._center
    
    def set_radius(self, rad):
        if rad <= 0:
            raise ValuError("Radius must be positive")
        self._rad = rad

    # Ray handling protocol:
    def register_incoming(self, ray_bundle):
        """
        Deals wih a ray bundle intersecting with a sphere
        Arguments:
        ray_bundle - the incoming bundle 
        Returns a 1D array with the parametric position of intersection along
        each ray.  Rays that miss the surface return +infinity
        """
        d = ray_bundle.get_directions()
        v = ray_bundle.get_vertices()
        n = ray_bundle.get_num_rays()
        c = self.get_center()
        params = []
        vertices = []
        norm = []

        for ray in xrange(n):
            # Solve the equations to find the intersection point:
            A = d[0,ray]**2 + d[1,ray]**2 + d[2,ray]**2 
            B = 2*(d[0,ray]*(v[0,ray] - c[0])
                   +d[1,ray]*(v[1,ray] - c[1])
                   +d[2,ray]*(v[2,ray] - c[2]))
            C = ((v[0,ray] - c[0])**2 
               +(v[1,ray] - c[1])**2
               +(v[2,ray] - c[2])**2 - self.get_radius()**2)

            vertex = v[:,ray]

            # If the discriminant is less than zero, the solution is not real,
            # and it misses the surface, so the parametric position of that
            # ray is returned as +infinity
            if (B**2 - 4*A*C) < 0:
                params.append(N.inf)

            # If the discriminant is not less than zero, we must use the quadratic
            # equation to solve for the possible parametric solutions, t0 and t1
            else:
                t0 = (-B - N.sqrt(B**2 - 4*A*C))/(2*A)
                t1 = (-B + N.sqrt(B**2 - 4*A*C))/(2*A)
                coords = N.c_[v[:,ray] + d[:,ray]*t0, v[:,ray] + d[:,ray]*t1]
                hits = N.r_[[t0,t1]]

                is_positive = N.where(hits > 0)
            
                # If both are negative, it is a miss
                if N.shape(is_positive)[1] == 0:
                    params.append(N.inf)
                    vertices.append(N.empty([3,1]))
                
                else:
                    # If both are positive, use the smaller one (where the ray first intersected)
                    if N.shape(is_positive)[1] == 2:
                        param = N.argmin(hits)
                                                
                    # If either one is negative, use the positive one
                    else:
                        param = is_positive[0][0]
                        
                    verts = N.c_[coords[:,param]]
                    
                    # Define normal based on whether it is hitting an inner or
                    # an outer surface of the sphere
                    dot = N.vdot(c-coords[:,param], N.c_[coords[:,param]-vertex])
                    if dot <= 0:  # it hits an inner surface
                        normal = N.c_[c-coords[:,param]]
                    else:  # it hits an outer surface
                        normal = N.c_[coords[:,param]-c]
                       
                    # Check if it is hitting within the boundary
                    selector = self._boundary.in_bounds(verts)
                    if selector[0]:
                        params.append(hits[param])
                        vertices.append(verts)
                        norm.append(normal)
                    else:
                        params.append(N.inf)
                        vertices.append(N.empty([3,1]))
                    
        # Storage for later reference:
        n = len(vertices)
        self._vertices = N.array(vertices).reshape(-1,3).T  
        self._current_bundle = ray_bundle
        self._norm = N.array(norm).reshape(-1,3).T
    
        return params

    def get_outgoing(self, selector):
        """
        Generates a new ray bundle, which is the reflection of the user selected rays out of
        the incoming ray bundle that was previously registered.
        Arguments:
        selector - a boolean array specifying which rays of the incoming bundle are still relevant
        Returns: a new RayBundle object with the new bundle, with vertices where it intersected with the surface, and directions according to the optic laws
        """
        dirs = optics.reflections(self._current_bundle.get_directions()[:,selector],
                                  self._norm)
        new_parent = parent[selector]

        outg = RayBundle()
        outg.set_vertices(self._vertices[:,selector])
        outg.set_directions(dirs)
        outg.set_energy(energy[:,selector])
        outg.set_parent(new_parent)

        return outg

