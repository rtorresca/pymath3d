#coding=utf-8
"""
Module for Plane class.
"""

__author__ = "Morten Lind"
__copyright__ = "Morten Lind 2013"
__credits__ = ["Morten Lind"]
__license__ = "GPLv3"
__maintainer__ = "Morten Lind"
__email__ = "morten@lind.dyndns.dk"
__status__ = "Development"

import math3d as m3d
import numpy as np

class Plane(object):
    def __init__(self, **kwargs):
        """Create a plane representation by one of the following named
        arguments:

        * 'plane_vector': A normalized plane vector. The normal will
          be pointing away from the origo. If kw-argument
          'origo_inside' is given, this will determine the direction
          of the plane normal; otherwise origo will be set inside.
        
        * 'pn_pair': An ordered sequence for creating a reference
        point and a normal vector. The normal

        * 'points': A set of at least three points for fitting a
        plane.

        The internal representation is point and normal. If given as a
        pn_pair, A boolean, 'origo_inside', is held to decide the
        direction of the normal vector, such that the origo of the
        defining coordinate system is on the inside when true."""

        self._origo_inside = kwargs.get('origo_inside', True)
        if 'plane_vector' in kwargs:
            pv = m3d.Vector(kwargs['plane_vector'])
            (self._p, self._n) = self.pv_to_pn(pv)
        elif 'pn_pair' in kwargs:
            (self._p, self._n) = [m3d.Vector(e) for e in kwargs['pn_pair']]
            # # Override a given origo inside.
            self._origo_inside = (self._p * self._n) > 0
            # # Make point a 'minimal' point on the plane, i.e. the
            # # projection of origo in the plane.
            self._p = (self._p * self._n) * self._n
        elif 'points' in kwargs:
            self.fit_plane(kwargs['points'])
        else:
            raise Exception('Plane.__init__ : Must have either of constructor '
                            + 'kw-arguments: "plane_vector", "pn_pair", or ' +
                            '"points". Neither given!')
    def copy(self):
        return Plane(pn_pair=(self._p, self._n))

    def __repr__(self):
        return '<Plane: [{:.5f}, {:.5f}, {:.5f}]>'.format(
            *tuple(self.plane_vector.array))

    def __rmul__(self, transf):
        """Support transformation of this plane to another coordinate
        system by multiplication of an m3d.Transform from left."""
        if type(transf) != m3d.Transform:
            return NotImplemented
        tnormal = transf.orient * self._n
        tpoint = transf * self._p
        return Plane(pn_pair=(tpoint, tnormal))
    
    def dist(self, p):
        """Signed distance to a point, measured positive along the
        normal vector direction."""
        return (m3d.Vector(p) - self._p) * self._n

    @property
    def plane_vector(self):
        return self.pn_to_pv(self._p, self._n)
    @plane_vector.setter
    def plane_vector(self, pv):
        (self._p, self._n) = self.pv_to_pn(pv)
        
    @property
    def point_normal(self):
        return (self._p, self._n)

    @property
    def point(self):
        return self._p

    @property
    def normal(self):
        return self._n

    def fit_plane(self, points):
        """Compute the plane vector from a set of points. 'points'
        must be an array of row position vectors, such that
        points.T[i] is a position vector array."""
        centre = np.sum(points, axis=0)/len(points)
        eigen = np.linalg.eig(np.cov(points.T))
        min_ev_i = np.where(eigen[0] == min(eigen[0]))[0][0]
        normal = eigen[1].T[min_ev_i]
        (self._p, self._n) = (m3d.Vector(centre), m3d.Vector(normal))
        

    def pn_to_pv(self, p, n):
        """Compute the plane vector of a plane represented by a point
        and normal."""
        if type(p) != m3d.Vector:
            p = m3d.Vector(p)
        if type(n) != m3d.Vector:
            n = m3d.Vector(n)
        # // Origo projection on plane
        p0 = (p * n) * n
        # // Square of offset from origo
        d2 = p0.length_squared
        # // return the plane vector
        return p0 / (d2)

    def pv_to_pn(self, pv):
        """Calculate a point-normal representation of the plane
        described by the given plane vector."""
        if type(pv) != m3d.Vector:
            pv = m3d.Vector(pv)
        d= pv.length
        n = pv / pv.length
        p = n / d
        if not self._origo_inside:
            n = -n
        return (p,n)
