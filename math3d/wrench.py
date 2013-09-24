# coding=utf-8
"""
Module implementing the Wrench class. A wrench is a spatial vector
composed of a force and a torqu acting at a point. The two components
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
    """An OrigoWrench is a wrench, i.e. a force and a torque vector,
    which acts around the origo of the defining coordinate
    system. When transformed, it changes to the origo of the target
    coordinate system.
    """

    def __init__(self, *args, **kwargs):
        """Args may contain 1) one iterable of two iterables of three
        floats. 2) six floats. 3) two iterables of three floats.
        """
        self._force = m3d.Vector(args[:3])
        self._torque = m3d.Vector(args[3:])

    def __rmul__(self, left):
        if type(left) == m3d.Transform:
            ...

class FootedWrench(object):
    """A FootedWrench is a wrench that contains, in addition to the
    force-torque vectors, a position vector, the 'foot point', for
    holding the position of action of the wrench. Under coordinate
    changes, the force and torque vectors then transforms as free
    vectors and the foot point transforms as a position vector."""
    pass
