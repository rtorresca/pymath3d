"""
Module implementing a 3D homogenous Transform class. The transform is
represented internally by associated orientation and a vector objects.
"""

__author__ = "Morten Lind"
__copyright__ = "Morten Lind 2009-2012"
__credits__ = ["Morten Lind"]
__license__ = "GPL"
__maintainer__ = "Morten Lind"
__email__ = "morten@lind.no-ip.org"
__status__ = "Production"

import numpy as np

import math3d as m3d
from math3d.utils import isSequence, _eps

def isTransform(t):
    if __debug__: print('Deprecation warning: "isTransform(t)".'
          + ' Use "type(t) == math3d.Transform".')
    return type(t)==Transform

class Transform(object):
    """ A Transform is a member of SE(3), represented as a homogenous
    transformation matrix. It uses an Orientation in member '_o'
    (accessible through 'orient') to represent the orientation part
    and a Vector in member '_v' (accessible through 'pos') to
    represent the position part."""

    class Error(Exception):
        """ Exception class."""
        def __init__(self, message):
            self.message = 'Transform Error : ' + message
            Exception.__init__(self, self.message)
        def __repr__(self):
            return self.message

    def __create_on_sequence(self, arg):
        """Called from init when a single argument of sequence type
        was given the constructor."""
        # if len(arg) == 1 and isSequence(arg[0]):
        #     self.__createOnSequence(arg[0])
        if type(arg) in (tuple, list):
            self.__create_on_sequence(np.array(arg, dtype=np.float64))
        elif type(arg) == np.ndarray and arg.shape in ((4,4), (3,4)):
            self._o = m3d.Orientation(arg[:3,:3])
            self._v = m3d.Vector(arg[:3,3])
        elif type(arg) == np.ndarray and arg.shape==(6,):
                # # Assume a pose vector of 3 position vector and 3 rotation vector components
                self._v = m3d.Vector(arg[:3])
                self._o = m3d.Orientation(arg[3:])
        else:
            raise self.Error(
                'Could not create Transform on arguments : "' + str(arg) + '"')
        
    def __init__(self, *args):
        """A Transform is a homogeneous transform on SE(3), internally
        represented by an Orientation and a Vector. A Transform can be constructed on:
        * A Transform.
        * A numpy array, list or tuple of shape (4,4) or (3,4) giving direct data; as [orient | pos].
        * A --''-- of shape (6,) giving a pose vector; concatenated position and rotation vector.
        * Two --''--; the first for orientation and the second for position.
        * Four --''--; the first three for orientation and the fourth for position.
        * Twelve numbers, the first nine used for orientation and the last three for position.
        * An ordered pair of Orientation and Vector.
        """
        if len(args) == 0: # or (len(args) == 1 and type(args[0]) == type(None)):
            self._v = m3d.Vector()
            self._o = m3d.Orientation()
        elif len(args) == 1:
            arg = args[0]
            if type(arg) == Transform or \
                   hasattr(arg,'pos') and hasattr(arg,'orient'): # and Orientation.canCreateOn(arg.orient) and m3d.Vector.canCreateOn(arg.pos):
                self._v = m3d.Vector(arg.pos)
                self._o = m3d.Orientation(arg.orient)
            else:
                self.__create_on_sequence(arg)
        elif len(args) == 2:
            self._o = m3d.Orientation(args[0])
            self._v = m3d.Vector(args[1])
        elif len(args) == 4:
            self._o = m3d.Orientation(args[:3])
            self._v = m3d.Vector(args[3])
        elif len(args) == 12:
            # // 12 numbers are required
            args = np.array(args, dtype=float64)
            self._o = m3d.Orientation(args[:9])
            self._v = m3d.Vector(args[9:])
        else:
            raise self.Error(
                'Could not create Transform on arguments : "' + str(args) + '"')
        self._setFromOV(self._o, self._v)
        
    def _setFromOV(self, o, v):
        self._data = np.identity(4)
        ## First take over the data from Orientation and Vector
        self._data[:3,:3] = o._data
        self._data[:3,3] = v._data
        ## Then share data with Orientation and Vector.
        self._o._data = self._data[:3,:3]
        self._v._data = self._data[:3,3]
        
    def __getattr__(self, name):
        if name == 'data':
            return self._data.copy()
        elif name == 'orient':
            return self._o
        elif name == 'pos':
            return self._v
        else:
            raise AttributeError('Attribute "%s" not found in Transform'%name)
        
    def __setattr__(self, name, value):
        if name == 'orient':
            if type(value) == np.ndarray:
                self._data[:3,:3] = value
            elif type(value) == m3d.Orientation:
                self._data[:3,:3] = value._data
        elif name == 'pos':
            if type(value) == np.ndarray:
                self._data[:3,3] = value
            elif type(value) == m3d.Vector:
                self._data[:3,3] = value._data
        else:
            object.__setattr__(self, name, value)
            
    def __copy__(self):
        """Copy method for creating a (deep) copy of this Transform."""
        return Transform(self)
    
    def __deepcopy__(self, memo):
        return self.__copy__()

    def copy(self, other=None):
        """Copy data from other to self. """
        if other is None:
            return Transform(self)
        else:
            self._data[:,:] = other._data.copy()
        
    def __repr__(self):
        return '<Transform:\n' + repr(self.orient) + '\n' + repr(self.pos) + '>'

    def __str__(self):
        return self.__repr__()
    
    def __coerce__(self, other):
        print('!!!! Warning !!!!  : Coercion called on Transform!!!')
        if type(other) == Transform:
            return (self, other)
        elif type(other) == m3d.Vector:
            return (self.pos, other)
        elif type(other) == m3d.Orientation:
            return (self.orient, other)
        else:
            return None
        
    def __eq__(self,other):
        if type(other) == Transform:
            return np.sum((self._data-other._data)**2) < _eps
        else:
            raise self.Error('Could not compare to non-Transform!')

    def fromXYP(self, cx, cy, p):
        """Make this transform correspond to the orientation given by
        the given 'cx' and 'cy' directions and translation given by
        'p'."""
        self._o.fromXY(cx, cy)
        self._v.pos = p
        self._setFromOV(self._o, self._v)
        
    def fromXZP(self, cx, cz, p):
        """Make this transform correspond to the orientation given by
        the given 'cx' and 'cz' directions and translation given by 'p'."""
        self._o.fromXZ(cx, cz)
        self._v.pos = p
        self._setFromOV(self._o, self._v)

    def dist2(self, other):
        """Return the square of the metric distance, as unweighted
        combined linear and angular distance, to the 'other'
        transform. Note that the units and scale among linear and
        angular representations matters heavily."""
        return self._v.dist2(other._v) + self._o.angDist2(other._o)
    
    def dist(self, other):
        """Return the metric distance, as unweighted combined linear
        and angular distance, to the 'other' transform. Note that the
        units and scale among linear and angular representations
        matters heavily."""
        return np.sqrt(self.dist2(other))

    def inverse(self):
        """Return an inverse of this Transform."""
        #io = self._o.inverse()
        #iv = io * (-self._v)
        #return Transform(io, iv)
        return Transform(np.linalg.inv(self._data))
    
    def invert(self):
        """In-place invert this Transform."""
        #self._o.invert()
        # self._v = self._o * (-self._v)
        self._data[:,:]=np.linalg.inv(self._data)
        
    def __mul__(self, other):
        """Multiplication of self with another Transform or operate
        on a Vector given by 'other'."""
        if type(other) == Transform:
            #o = self._o * other._o
            #v = self._o * other._v + self._v
            #return Transform(o, v)
            return Transform(np.dot(self._data, other._data))
        elif type(other) == m3d.Vector:
            #return self._o * other + other._isPosition * self._v
            v = np.ones(4)
            v[:3] = other._data
            return m3d.Vector(np.dot(self._data, v)[:3])
        elif type(other) == np.ndarray and other.shape == (3,):
            return np.dot(self._o._data, other)+self._v._data
        elif isSequence(other):
            return list(map(self.__mul__,other))
        else:
            raise self.Error('Inadequate data type for multiplication in "other" : "'
                             + str(type(other)) + '"')
        
    def toArray(self):
        """Return a tuple pair of an 3x3 orientation array and
        position as 3-array."""
        #return (self._o._data.copy(),self._v._data.copy())
        return (self._data[:3,:3], self._data[:3,3])
    
    def toList(self):
        """Return a list with orientation and position in list form."""
        #return [self._o._data.tolist(),self._v._data.tolist()]
        return [self._data[:3,:3].tolist(), self._data[:3,3].tolist()]
    
def newTransFromXYP(cx, cy, p):
    """Create a transform corresponding to the orientation given by
    the given 'cx' and 'cy' directions and translation given by 'p'."""
    t = Transform()
    t.fromXYP(cx, cy, p)
    return t

def newTransFromXZP(cx, cz, p):
    """ Create a transform corresponding to the orientation given by
    the given 'cx' and 'cz' directions and translation given by 'p'."""
    t = Transform()
    t.fromXZP(cx, cz, p)
    return t

def _test():
    cx = m3d.Vector(2, 3, 0)
    cz = m3d.Vector.e2
    p = m3d.Vector(1, 2, 3)
    t = newTransFromXZP(cx, cz, p)
    print((t*cx))
    it = t.inverse()

if __name__ == '__main__':
    _test()
