# coding=utf-8

"""
Module for line class
"""

__author__ = "Morten Lind"
__copyright__ = "Morten Lind 2016"
__credits__ = ["Morten Lind"]
__license__ = "GPLv3"
__maintainer__ = "Morten Lind"
__email__ = "morten@lind.dyndns.dk"
__status__ = "Development"


import math3d as m3d
import numpy as np


class Line(object):
    """A line class."""

    def __init__(self, **kwargs):
        """Supported, named constructor arguments:

        * 'point_direction': An ordered pair of vectors representing a
          point on the line and the line's direction.

        * 'point', 'direction': Separate vectors for point on line and
          direction of line in named arguments.

        * 'point0', 'point1': Two points defining the line, in
        separate named arguments.
        """

        if 'point_direction' in kwargs:
            self._p, self._d = [m3d.Vector(e)
                                for e in kwargs['point_direction']]
        elif 'point' in kwargs and 'direction' in kwargs:
            self._p = m3d.Vector(kwargs['point'])
            self._d = m3d.Vector(kwargs['direction'])
        elif 'point0' in kwargs and 'point1' in kwargs:
            self._p = m3d.Vector(kwargs['point0'])
            self._d = m3d.Vector(kwargs['point1']) - self._p
        else:
            raise Exception(
                'Can not create Line object on given arguments: "{}"'
                .format(kwargs))

    @property
    def point(self):
        return m3d.Vector(self._p)

    @property
    def direction(self):
        return m3d.Vector(self._d)
