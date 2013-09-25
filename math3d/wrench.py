# coding=utf-8
"""
Module implementing the Wrench class(es). A wrench is a spatial vector
composed of a force and a moment acting at a point. The two components
transform in a different way than two separate vectors.
"""

__author__ = "Morten Lind"
__copyright__ = "Morten Lind 2013"
__credits__ = ["Morten Lind"]
__license__ = "GPLv3"
__maintainer__ = "Morten Lind"
__email__ = "morten@lind.dyndns.dk"
__status__ = "Production"


import numpy as np
import math3d as m3d

class OrigoWrench(object):
    """An OrigoWrench is a wrench, i.e. a force and a moment vector,
    which acts around the origo of the defining coordinate
    system. When transformed, it changes to the origo of the target
    coordinate system.
    """

    def __init__(self, *args, **kwargs):
        """Args may contain 1) one iterable of two iterables of three
        floats. 2) six floats. 3) two iterables of three floats.
        """
        if 'f' in kwargs or 'm' in kwargs:
            self._force = m3d.Vector(kwargs.get('f', m3d.Vector()))
            self._moment = m3d.Vector(kwargs.get('m', m3d.Vector()))
        elif len(args) == 1 and len(args[0]) == 6:
            self._force = m3d.Vector(args[0][:3])
            self._moment = m3d.Vector(args[0][3:])
        elif len(args) == 1 and type(args[0]) == OrigoWrench:
            self._force = args[0].force
            self._moment = args[0].moment
        elif len(args) == 2 and len(args[0]) == 3 and len(args[1]) == 3:
            self._force = m3d.Vector(args[0])
            self._moment = m3d.Vector(args[1])
        else:
            raise Exception(self.__class__.__name__ + 'Could not construct on given arguments: *args=' + str(args) + ' *kwargs=' + str(kwargs))

    def __rmul__(self, left):
        """Mainly support transformation to another coordinate system,
        where the equivalent wrench is returned as acting at the
        target coordinate system origo. In case of a transform, the 
        """
        if type(left) == m3d.Transform:
            # The new force is the same as the old, but
            # reoriented. The new moment is the old one reoriented
            # plus the action of the force acting at the old origo
            m = left.orient * self._moment
            f_n = left.orient * self._force 
            m_n = m + left.pos.cross(f_n)
            return OrigoWrench(np.append(f_n.data, m_n.data))
            
    def get_moment(self):
        """Get the moment part."""
        return self._moment.copy()
    def set_moment(self, new_moment):
        """Set the moment part."""
        self._moment = m3d.Vector(new_moment)
    moment = property(get_moment, set_moment)

    def get_force(self):
        """Get the force part."""
        return self._force.copy()
    def set_force(self, new_force):
        """Set the force part."""
        self._force = m3d.Vector(new_force)
    force = property(get_force, set_force)

    def __add__(self, w_add):
        """Add two wrenches. Note that they are percieved as belonging
        to the same origo in the same coordinate system!.
        """
        return OrigoWrench(f=self._force+w_add._force, m=self._moment+w_add._moment)

    def __sub__(self, w_sub):
        """Subtract two wrenches. Note that they are percieved as belonging
        to the same origo in the same coordinate system!.
        """
        return OrigoWrench(f=self._force-w_sub._force, m=self._moment-w_sub._moment)

    def __neg__(self):
        """Return the negative wrench."""
        return OrigoWrench(f=-self._force, m=-self._moment.data)

    def __repr__(self):
        """String represenstation of the wrench."""
        return '<{} f=[{:.3f}, {:.3f}, {:.3f}] m=[{:.3f}, {:.3f}, {:.3f}]>'.format(
            *([self.__class__.__name__] + self._force.list + self._moment.list))

class FootedWrench(object):
    """A FootedWrench is a wrench that contains, in addition to the
    force-moment vectors, a position vector, the 'foot point', for
    holding the position of action of the wrench. Under coordinate
    changes, the force and moment vectors then transforms as free
    vectors and the foot point transforms as a position vector."""
    pass
