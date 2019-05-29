"""
This module contains functions to create some frequently-used light sources.
Each function returns a RayBundle instance that represents the distribution of
rays expected from that source.

References:
.. [1] Monte Carlo Ray Tracing, Siggraph 2003 Course 44


Ye changed def(solar_rect_bundel)
Line189 vertices_local = N.vstack((ys, xs, N.zeros(num_rays)))
to      vertices_local = N.vstack((xs, ys, N.zeros(num_rays)))
28 Oct 2016
"""

from numpy import random, linalg as LA
import numpy as N
from tracer.ray_bundle import RayBundle
from tracer.spatial_geometry import *

def single_ray_source(position, direction, flux=None):
    '''
    Establishes a single ray source originating from a definned point on a defined exact 
    direction for the purpose of testing single ray behviours.

    Arguments:
    position - column 3-array with the ray's starting position.
    direction - a 1D 3-array with the unit average direction vector for the
                bundle.
    flux - if not None, the energy transported by the ray.

    Returns:
    A Raybundle object with the corresponding characteristics.
    '''
    directions = N.tile(direction[:,None],1)
    directions /= N.sqrt(N.sum(directions**2, axis=0))
    singray = RayBundle(vertices = position, directions = directions)
    singray.set_energy(flux*N.ones(1))
    return singray

def pillbox_sunshape_directions(num_rays, ang_rang):
    """
    Calculates directions for a ray bundles with ``num_rays`` rays, distributed
    as a pillbox sunshape shining toward the +Z axis, and deviating from it by
    at most ang_range, such that if all rays have the same energy, the flux
    distribution comes out right.
    
    Arguments:
    num_rays - number of rays to generate directions for.
    ang_range - in radians, the maximum deviation from +Z.
    
    Returns:
    A (3, num_rays) array whose each column is a unit direction vector for one
        ray, distributed to match a pillbox sunshape.
    """

    x1=N.random.uniform(size=num_rays)
    x2=N.random.uniform(size=num_rays)

    theta=N.sqrt(x1)*ang_rang
    phi=x2*2.*N.pi

    a = N.vstack((N.cos(phi)*N.sin(theta), N.sin(phi)*N.sin(theta) , N.cos(theta)))

    return a

def gaussian_sunshape_directions(num_rays, sigma, simplified=True):
    """
    Calculates directions for a ray bundles with ``num_rays`` rays, distributed
    as a Gaussian sunshape shining toward the +Z axis, and deviating from it by
    sigma, such that if all rays have the same energy, the flux
    distribution comes out right.
    
    Arguments:
    num_rays - number of rays to generate directions for.
    sigma- in radians, the standard deviation of the Gaussian distribution
    
    Returns:
    A (3, num_rays) array whose each column is a unit direction vector for one
        ray, distributed to match a Gaussian distribution.
    """
    if simplified:
        x = N.random.normal(scale=sigma, size=num_rays)
        y = N.random.normal(scale=sigma, size=num_rays)        
        z = N.ones(num_rays)
        a = N.vstack((x, y, z))
    else:
        phi = N.random.uniform(low=0., high=2.*N.pi, size=num_rays)

        x= N.random.uniform(low=0., high=1., size=num_rays)
        theta=sigma*N.sqrt(-2.*N.log(x))

        a = N.vstack((N.cos(phi)*N.sin(theta), N.sin(phi)*N.sin(theta) , N.cos(theta)))
    return a

def collimated_directions(num_rays):

    """
    Calculates directions for a ray bundles with ``num_rays`` rays, 
    shining toward the +Z axis. Assist theoretical tests.

    Arguments:
    num_rays - number of rays to generate

    Returns:
    A (3, num_rays) array whose each column is a unit direction vector for one
        ray.
    """
    x=N.zeros(num_rays)
    y=N.zeros(num_rays)
    z=N.ones(num_rays)

    a = N.vstack((x, y,z))

    return a

class buie_integration:

    def __init__(self, CSR, preproc_CSR):
        self.CSR=CSR
        self.theta_1=4.65e-3
        self.theta_2=43.6e-3

        if preproc_CSR=='CA':
            self.CSR=self.CSR_calibration('CA')
        elif preproc_CSR=='tonatiuh':
            self.CSR=self.CSR_calibration('tonatiuh')    

        self.sample, self.FI=self.int_sun_region()


    def CSR_calibration(self, source):
        '''
        pre proceed CSR to true value
        source - 'CA' from Charles Charles-Alexis Asselineau at ANU; or 'tonatiuh' from Manuel Blanco at CyI
        '''
        csr=self.CSR
        if source=='CA':
            if csr<=0.1:
                CSR = -2.245e+03*csr**4.+5.207e+02*csr**3.-3.939e+01*csr**2.+1.891e+00*csr+8e-03
            else:
                CSR = 1.973*csr**4.-2.481*csr**3.+0.607*csr**2.+1.151*csr-0.020   

        elif source=='tonatiuh':

            if csr>0.145:
                CSR = -0.04419909985804843 + csr * (1.401323894233574 + csr * (-0.3639746714505299 + csr * (-0.9579768560161194 + 1.1550475450828657 * csr)))
		
            elif csr>0.035 and csr <=0.145:
                CSR=0.022652077593662934 + csr * (0.5252380349996234 + (2.5484334534423887 - 0.8763755326550412 * csr)) * csr
		
            else:
                CSR = 0.004733749294807862 + csr* (4.716738065192151 + csr * (-463.506669149804 + csr * ( 24745.88727411664+csr * (-606122.7511711778 + 5521693.445014727 * csr) ) ) )

        return CSR



    def int_csr_region(self, theta):

        self.kappa=0.9*N.log(13.5*self.CSR)*N.power(self.CSR,-0.3)
        self.gamma=2.2*N.log(0.52*self.CSR)*N.power(self.CSR,0.43)-0.1

        # the circuim solar region
        I2=10.**(self.gamma*3.)*N.exp(self.kappa)/(self.gamma+2.)*(theta**(self.gamma+2.)-self.theta_1**(self.gamma+2.))
        return I2 
   
    def func(self, theta):
        '''
        theta is in radian
        '''
        f=theta*N.cos(theta*326.)/N.cos(theta*308.)

        return f 

    def numerical_int(self, x1, x2, n):

        dx=(x2-x1)/float(n)
        I=0.
        for k in xrange(n):
            I+=dx*self.func(x1+float(k+0.5)*dx)
        return I

    def int_disk_region(self, initial=False):
        I1=self.numerical_int(0., self.theta_1, 100)

        theta=N.linspace(0., self.theta_1, 10000)
        I=N.array([])

        i=1
        for t in theta:
            Ii=self.numerical_int(0, t, 100)  
            #I0+=Ii
            I=N.append(I, Ii)
            i+=1
        PI=I/I1

        return I1, theta, PI


    def int_sun_region(self):

        I1=self.numerical_int(0., self.theta_1, 100)
        I2=self.int_csr_region(self.theta_2)

        theta=N.linspace(0., self.theta_1, 10000)
        theta2=N.linspace(self.theta_1, self.theta_2, 1000)
        I=N.array([])

        for t in theta:
            Ii=self.numerical_int(0, t, 100)  
            I=N.append(I, Ii)

        for t2 in theta2:
            Ii=self.int_csr_region(t2)
            I=N.append(I, Ii+I1)

        PI=I/(I1+I2)

        THETA=N.append(theta, theta2) 

        return THETA, PI


def buie_sunshape_directions(num_rays, sample, FI):

    """
    Calculates directions for a ray bundles with ``num_rays`` rays, distributed 	
    according to Buie et al.: "Sunshape distributions for terrestrial simulations." Solar Energy 74 (2003) 113-122 (DOI: 10.1016/S0038-092X(03)00125-7).

    Arguments:
    num_rays - number of rays to generate directions for.
    CSR - Circumsolar ratio, fraction of the incoming solar energy which incident angle is greater than the angle subtended by the solar disc.
    
    Returns:
    A (3, num_rays) array whose each column is a unit direction vector for one
        ray, distributed to match buie sunshape.
    """
    
    x1=N.random.uniform(low=0., high=1., size=num_rays) 
    theta=N.interp(x1, FI, sample)

    phi = N.random.uniform(low=0., high=2.*N.pi, size=num_rays)

    a = N.vstack((N.cos(phi)*N.sin(theta), N.sin(phi)*N.sin(theta) , N.cos(theta)))
   
    return a


def rect_ray_bundle(num_rays, center, direction, x, y, sunshape, ang_rang, flux=None, BUIE=None):
    '''
    generate a rectancular ray bundle for different sunshape options

    Arguments:
    num_rays - number of rays to generate
    center - a column 3-array with the 3D coordinate of the ray bundle's center 
    direction - a 1D 3-array with the unit average direction vector for the bundle.
    x - width of the rectangular ray bundle
    y - height of the rectangular ray bundle
    sunshape - str, 'pillbox', 'Gaussian','Buie' or 'collimated'
    ang_rang - the angular width of the solar disk (pillbox), sigma (gaussian) or CSR (Buie sunshape)
    flux - if not None, the ray bundle's energy is set such that each ray has
        an equal amount of energy, and the total energy is flux*pi*radius**2

    Returns:
    A RayBundle object with the above characteristics set.
    '''
    if sunshape=='pillbox':
        a= pillbox_sunshape_directions(num_rays, ang_rang)
    elif sunshape=='gaussian': 
        a= gaussian_sunshape_directions(num_rays, ang_rang)
    elif sunshape=='buie':   
        a=buie_sunshape_directions(num_rays, BUIE.sample, BUIE.FI)
    elif sunshape=='collimated':
        a= collimated_directions(num_rays)

    # making sure the rect bundle can cover the whole region of interest
    x*=1.2
    y*=1.2

    # Rotate to a frame in which <direction> is Z:
    perp_rot = rotation_to_z(direction)
    directions = N.sum(perp_rot[...,None] * a[None,...], axis=1)

	# Locations:
	# See [1]
    xs = N.random.uniform(low=-x/2., high=x/2., size=num_rays)
    ys = N.random.uniform(low=-y/2., high=y/2., size=num_rays)


    # Rotate locations to the plane defined by <direction>:
    vertices_local = N.vstack((xs, ys, N.zeros(num_rays)))
    vertices_global = N.dot(perp_rot, vertices_local)

    rayb = RayBundle(vertices=vertices_global + center, directions=directions)

    if flux != None:
        rayb.set_energy(x*y/num_rays*flux*N.ones(num_rays))

    return rayb


def bivariate_directions(num_rays, ang_range_hor, ang_range_vert):
    """
    Calculates directions for a ray bundles with ``num_rays`` rays, distributed
    as uniform bi-variate distribution shining toward the +Z axis and deviating from it by
    at most ang_range_hor on the zx plane and ang_range_vert on the yz plane such that if all rays have the same energy, the flux
    distribution comes out right.
    
    Arguments:
    num_rays - number of rays to generate directions for.
    ang_range_hor - in radians, the maximum deviation from +Z on the zx plane.
    ang_range_vert - in radians, the maximum deviation from +Z on the yz plane.
    
    Returns:
    A (3, num_rays) array whose each column is a unit direction vector for one
        ray, distributed to match a uniform bi-variate distribution.
    """
    # Diffuse divergence from +Z:
    # development based on eq. 2.12  from [1]
    '''
    xi1 = N.random.uniform(low=-1., high=1., size=num_rays)
    xi2 = N.random.uniform(low=-1., high=1., size=num_rays)
    
    theta_hor = N.arcsin(N.sin(ang_range_hor)*N.sqrt(xi1))
    theta_vert = N.arcsin(N.sin(ang_range_vert)*N.sqrt(xi2))

    xa = N.sin(theta_hor)
    ya = N.sin(theta_vert)
    za = N.sqrt(1.-N.sin(theta_hor)**2.-N.sin(theta_vert)**2.)

    a = N.vstack((xa, ya, za))
    '''
    return a

def edge_rays_directions(num_rays, ang_range):
    """
    Calculates directions for a ray bundles with ``num_rays`` rays, distributed
    as a pillbox sunshape shining toward the +Z axis, and deviating from it by
    at most ang_range, such that if all rays have the same energy, the flux
    distribution comes out right.
    
    Arguments:
    num_rays - number of rays to generate directions for.
    ang_range - in radians, the maximum deviation from +Z.
    
    Returns:
    A (3, num_rays) array whose each column is a unit direction vector for one
        ray, distributed to match a pillbox sunshape.
    """
    # Diffuse divergence from +Z:
    # development based on eq. 2.12  from [1]
    xi1 = random.uniform(high=2.*N.pi, size=num_rays)
    sin_th = N.ones(num_rays)*N.sin(ang_range)
    a = N.vstack((N.cos(xi1)*sin_th, N.sin(xi1)*sin_th , N.cos(N.ones(num_rays)*ang_range)))

    return a

def solar_disk_bundle(num_rays,  center,  direction,  radius, ang_range, flux=None, radius_in=0., angular_span=[0.,2.*N.pi], procs=1):
    """
    Generates a ray bundle emanating from a disk, with each surface element of 
    the disk having the same ray density. The rays all point at directions uniformly 
    distributed between a given angle range from a given direction.
    Setting of the bundle's energy is left to the caller.
    
    Arguments:
    num_rays - number of rays to generate.
    center - a column 3-array with the 3D coordinate of the disk's center
    direction - a 1D 3-array with the unit average direction vector for the
        bundle.
    radius - of the disk.
    ang_range - in radians, the maximum deviation from <direction>.
    flux - if not None, the ray bundle's energy is set such that each ray has
        an equal amount of energy, and the total energy is flux*pi*radius**2
    radius_in - Inner radius if the disc is pierced
    angular_span - wedge of the disc to consider
    
    Returns: 
    A RayBundle object with the above characteristics set.
    """

	# FIXME why should 'center' be a column vector... that's just annoying.

    radius = float(radius)
    radius_in = float(radius_in)
    a = pillbox_sunshape_directions(num_rays, ang_range)
        
    # Rotate to a frame in which <direction> is Z:
    perp_rot = rotation_to_z(direction)
    directions = N.sum(perp_rot[...,None] * a[None,...], axis=1)
    # Locations:
    # See [1]
    xi1 = random.uniform(size=num_rays)
    thetas = random.uniform(low=angular_span[0], high=angular_span[1], size=num_rays)
    rs = N.sqrt(radius_in**2.+xi1*(radius**2.-radius_in**2.))
    xs = rs * N.cos(thetas)
    ys = rs * N.sin(thetas)

    # Rotate locations to the plane defined by <direction>:
    vertices_local = N.vstack((xs, ys, N.zeros(num_rays)))
    vertices_global = N.dot(perp_rot, vertices_local)

    rayb = RayBundle(vertices=vertices_global + center, directions=directions)
    if flux != None:
        rayb.set_energy(N.pi*(radius**2.-radius_in**2.)/num_rays*flux*N.ones(num_rays))
    else:
        rayb.set_energy(N.ones(num_rays)/num_rays/procs)
    return rayb

def solar_rect_bundle(num_rays, center, direction, x, y, ang_range, flux=None):

    a = pillbox_sunshape_directions(num_rays, ang_range)

    # Rotate to a frame in which <direction> is Z:
    perp_rot = rotation_to_z(direction)
    directions = N.sum(perp_rot[...,None] * a[None,...], axis=1)

    xs = random.uniform(low=-x/2., high=x/2., size=num_rays)
    ys = random.uniform(low=-y/2., high=y/2., size=num_rays)

    if (direction == N.array([0,0,-1])).all():
        xs, ys = ys, xs

    # Rotate locations to the plane defined by <direction>:
    vertices_local = N.vstack((xs, ys, N.zeros(num_rays)))
    vertices_global = N.dot(perp_rot, vertices_local)

    rayb = RayBundle(vertices=vertices_global + center, directions=directions)
    if flux != None:
        rayb.set_energy(x*y/num_rays*flux*N.ones(num_rays))
    return rayb

#def bivariate_rect_bundle(num_rays, center, direction, x, y, ang_range_vert, ang_range_hor, flux=None):


def edge_rays_bundle(num_rays,  center,  direction,  radius, ang_range, flux=None, radius_in=0.):

    radius = float(radius)
    radius_in = float(radius_in)
    a = edge_rays_directions(num_rays, ang_range)
        
    # Rotate to a frame in which <direction> is Z:
    perp_rot = rotation_to_z(direction)
    directions = N.sum(perp_rot[...,None] * a[None,...], axis=1)
    # Locations:
    # See [1]
    xi1 = random.uniform(size=num_rays)
    thetas = random.uniform(high=2.*N.pi, size=num_rays)
    rs = N.sqrt(radius_in**2.+xi1*(radius**2.-radius_in**2.))
    xs = rs * N.cos(thetas)
    ys = rs * N.sin(thetas)

    # Rotate locations to the plane defined by <direction>:
    vertices_local = N.vstack((xs, ys, N.zeros(num_rays)))
    vertices_global = N.dot(perp_rot, vertices_local)

    rayb = RayBundle(vertices=vertices_global + center, directions=directions)
    if flux != None:
        rayb.set_energy(N.pi*(radius**2.-radius_in**2.)/num_rays*flux*N.ones(num_rays))
    return rayb



def square_bundle(num_rays, center, direction, width):
    """
    Generate a ray bundles whose rays are equally spaced along a square grid,
    and all pointing in the same direction.
    
    Arguments:
    num_rays - number of rays to generate.
    center - a column 3-array with the 3D coordinate of the disk's center
    direction - a 1D 3-array with the unit direction vector for the bundle.
    width - of the square of starting points.
    
    Returns: 
    A RayBundle object with the above charachteristics set.
    """
    rot = rotation_to_z(direction)
    directions = N.tile(direction[:,None], (1, num_rays))
    range1 = N.s_[-width:width:float(2*width)/N.sqrt(num_rays)]
    xs, ys = N.mgrid[range1, range1]
    vertices_local = N.array([xs.flatten(),  ys.flatten(),  N.zeros(len(xs.flatten()))])
    vertices_global = N.dot(rot,  vertices_local)

    rayb = RayBundle()
    rayb.set_vertices(vertices_global + center)
    rayb.set_directions(directions)
    return rayb

def vf_frustum_bundle(num_rays, r0, r1, depth, center, direction, flux=None , rays_in=True, procs=1, angular_span=[0.,2.*N.pi], angular_range=N.pi/2.):
    '''
    Generate a frustum shaped lambertian source with randomly situated rays to compute view factors. The overall energy of the bundle is 1.

    Arguments:
    num_rays - number of rays to generate.
    center - a column 3-array with the 3D coordinate of the center of one of the bases.
    r0 - The radius of the frustum base which center coordinate has been given.
    r1 - the radius of the frustum at the other base location.
    depth - the depth of the frustum.
    direction - The orientation of the overall bundle. 
               Positive if in the same direction as the depth.
    rays_in - True if rays are fired towards the axis of the frustum.
    angular_span - wedge of the shape to consider.

    Returns:
    A raybundle object with the above characteristics set.
    '''
    r0 = float(r0)
    r1 = float(r1)
    depth = float(depth)
    if r0>r1:
         raise AttributeError('wrong radii') 
    if depth <0.:
         raise AttributeError('wrong depth')
    num_rays = float(num_rays)

    dir_flat = pillbox_sunshape_directions(num_rays, angular_range)

    c = (r1-r0)/depth

    R = random.uniform(size=num_rays)

    zs = (-r0+N.sqrt(r0**2.+R*(r1**2.-r0**2.)))/c

    phi_s = random.uniform(low=angular_span[0], high=angular_span[1], size=num_rays)
    rs = r0+c*zs
    xs = rs * N.cos(phi_s)
    ys = rs * N.sin(phi_s)

    theta_s = N.arctan(c)
    theta_rot = -N.pi/2.+theta_s
    yrot = roty(theta_rot)[:3,:3]
    local_unit = N.zeros((N.shape(dir_flat)))
    for t in xrange(N.shape(dir_flat)[1]):
        zrot = rotz(phi_s[t])[:3,:3]
        rot = N.dot(zrot, yrot)
        local_unit[:,t] = N.dot(rot, dir_flat[:,t])

    if rays_in == False:
        local_unit = -local_unit

    vertices_local = N.vstack((xs, ys, zs))

    perp_rot = rotation_to_z(direction)
    vertices_global = N.dot(perp_rot, vertices_local)
    directions = N.dot(perp_rot, local_unit)

    if flux == None:
        energy = N.ones(num_rays)/num_rays/procs
    else:
        area = (angular_span[1]-angular_span[0])*(r1+r0)/2.*N.sqrt(abs(r1-r0)**2.+depth**2.)
        energy = N.ones(num_rays)*flux*area/num_rays/procs

    rayb = RayBundle(vertices = vertices_global+center, directions = directions, energy = energy)

    return rayb

def vf_cylinder_bundle(num_rays, rc, lc, center, direction, flux=None, rays_in=True, procs=1, angular_span=[0.,2.*N.pi]):
    '''
    Generate a cylinder shaped lambertian source with randomly situated rays to compute view factors. The overall energy of the bundle is 1.

    Arguments:
    num_rays - number of rays to generate.
    center - a column 3-array with the 3D coordinate of the center of one of the bases.
    rc - The radius of the cylinder.
    lc - the length of the cylinder.
    direction - the direction of outgoing rays as projected on the cylinder axis. 
               Positive if in the same direction as lc.
    rays_in - True if rays are fired towards the axis of the frustum.
    angular_span - wedge of the shape to consider.

    Returns:
    A raybundle object with the above characteristics set.
    '''
    rc = float(rc)
    lc = float(lc)
    num_rays = float(num_rays)

    zs = lc*random.uniform(size=num_rays)

    phi_s = random.uniform(low=angular_span[0], high=angular_span[1], size=num_rays)

    xs = rc * N.cos(phi_s)
    ys = rc * N.sin(phi_s)

    dir_flat = pillbox_sunshape_directions(num_rays, N.pi/2.)

    yrot = roty(-N.pi/2.)[:3,:3]
    local_unit = N.zeros((N.shape(dir_flat)))
    for t in range(N.shape(dir_flat)[1]):
        zrot = rotz(phi_s[t])[:3,:3]
        rot = N.dot(zrot, yrot)
        local_unit[:,t] = N.dot(rot, dir_flat[:,t])

    if rays_in == False:
        local_unit = -local_unit

    vertices_local = N.vstack((xs, ys, zs))
    perp_rot = rotation_to_z(direction)
    vertices_global = N.dot(perp_rot, vertices_local)
    directions = N.dot(perp_rot, local_unit)
    '''
    plt.hist(vertices_local[2,:]/(N.sqrt(vertices_local[0,:]**2.+vertices_local[1,:]**2.)))
    plt.show()
    '''
    if flux == None:
        energy = N.ones(num_rays)/num_rays/procs
    else:
        area = rc*(angular_span[1]-angular_span[0])*lc
        energy = N.ones(num_rays)*flux*area/num_rays/procs

    rayb = RayBundle(vertices = vertices_global+center, directions = directions, energy = energy)

    return rayb


# vim: et:ts=4
