"""
Module implementing the SO(3) interpolator class; Slerp.
"""

__author__ = "Morten Lind"
__copyright__ = "Morten Lind 2009-2012"
__credits__ = ["Morten Lind"]
__license__ = "GPLv3"
__maintainer__ = "Morten Lind"
__email__ = "morten@lind.no-ip.org"
__status__ = "Production"

import numpy as np

from ..orientation import Orientation
from ..quaternion import Quaternion

class SO3Interpolation(object):
    """A SLERP interpolator class in SO(3)."""
    
    class Error(Exception):
        """Exception class."""
        def __init__(self, message):
            self.message = 'SO3Interpolation Error: ' + message
            Exception.__init__(self, self.message)
        def __repr__(self):
            return self.message
        
    def __init__(self, start, end, shortest=True):
        """Initialise an SO(3) interpolation from orientation 'start'
        to orientation 'end'. If 'shortest' is true, the shortest
        rotation path is chosen, if false, it is indeterminate."""
        self._qstart = Quaternion(start) if type(start) == Orientation else start
        self._qend = Quaternion(end) if type(end) == Orientation else end
        self._qstart.normalize()
        self._qend.normalize()
        if shortest and self._qstart.dist(self._qend) > self._qstart.dist(-self._qend):
            self._qend = -self._qend
        self._qstartconj = self._qstart.conjugated.normalized
        self._qstartconjqend = (self._qstartconj * self._qend).normalized

    def __call__(self,t):
        return self.quat(t)
    
    def quat(self, time, checkrange=True):
        """Return the quaternion in the slerp at 'time'; in [0,1]."""
        if checkrange:
            time = np.float64(time)
            if time < 0.0 or time > 1.0:
                raise self.Error('"time" must be number in [0.0 ; 1.0]. Was %f' % time) 
        return self._qstart * (self._qstartconjqend) ** time

    def orient(self, time, checkrange=True):
        """Return the orientation in the slerp at 'time'; in [0,1]. """
        return self.quat(time, checkrange).orientation
    
SLERP = SO3Interpolation
OrientationInterpolation = SO3Interpolation

def _test():
    """Simple test function."""
    global o, o1, q, q1, osl, qsl
    from math import pi
    o = Orientation()
    o.set_to_x_rotation(pi / 2)
    o1 = Orientation()
    o1.set_to_z_rotation(pi / 2)
    q = Quaternion(o)
    q1 = Quaternion(o1)
    qsl = SO3Interpolation(q,q1)
    osl = SO3Interpolation(o,o1)

if __name__ == '__main__':
    import readline
    import rlcompleter
    readline.parse_and_bind("tab: complete")

