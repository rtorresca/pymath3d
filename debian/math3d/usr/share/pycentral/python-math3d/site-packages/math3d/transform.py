"""
Copyright (C) 2011 Morten Lind
mailto: morten@lind.no-ip.org

This file is part of PyMath3D (Math3D for Python).

PyMath3D is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

PyMath3D is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PyMath3D.  If not, see <http://www.gnu.org/licenses/>.
"""
"""
Module implementing a 3D homogenous Transform class. The transform is
represented internally by associated orientation and a vector objects.
"""

import numpy as np

from math3d.orientation import Orientation
from math3d.vector import Vector
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
            self.message = message
        def __repr__(self):
            return self.__class__ + '.Error :' + self.message

    @classmethod
    def canCreateOn(cls, *args):
        """ Query whether a transform, syntactically and by type, can
        be constructed on the given arguments."""
        if len(args) == 0:
            return True
        elif len(args) == 1:
            arg = args[0]
            if type(arg) == type(None):
                return True
            elif type(arg) == Transform:
                return True
            elif type(arg) == np.ndarray and arg.shape==(4,4):
                return True
            elif hasattr(arg,'pos') and hasattr(arg,'orient') and \
                 Orientation.canCreateOn(arg.orient) and \
                 Vector.canCreateOn(arg.pos):
                return True
            elif isSequence(arg):
                return cls.canCreateOn(*arg)
            else:
                return False
        elif len(args) == 2:
            return \
                   Orientation.canCreateOn(args[0]) and \
                   Vector.canCreateOn(args[1]) \
                   or \
                   Orientation.canCreateOn(args[1]) and \
                   Vector.canCreateOn(args[0])
        elif len(args) == 4:
            return Orientation.canCreateOn(args[:3]) and \
                   Vector.canCreateOn(args[3])
        elif len(args) == 12:
            return Orientation.canCreateOn(args[:9]) and \
                   Vector.canCreateOn(args[9:])
        else:
            return False

    def __createOnSequence(self, args):
        if len(args) == 1 and isSequence(args[0]):
            self.__createOnSequence(args[0])
        elif len(args) == 2:
            if Orientation.canCreateOn(args[0]) \
                   and Vector.canCreateOn(args[1]):
                self._o = Orientation(args[0])
                self._v = Vector(args[1])
            elif Orientation.canCreateOn(args[1]) \
                   and Vector.canCreateOn(args[0]):
                self._o = Orientation(args[1])
                self._v = Vector(args[0])
            else:
                raise self.Error(
                    "Could not create Transform on arguments : " + str(args))
        elif len(args) == 4:
            self._o = Orientation(args[:3])
            self._v = Vector(args[3])
        elif len(args) == 12:
            self._o = Orientation(args[:9])
            self._v = Vector(args[9:])
        elif len(args) == 0 or len(args) == 1 and \
                 type(args[0]) == type(None):
            self._v = Vector()
            self._o = Orientation()
        else:
            raise self.Error(
                'Could not create Transform on arguments : "' + str(args) + '"')
        
    def __init__(self, *args):
        if len(args) == 0 or len(args) == 1 and \
               type(args[0]) == type(None):
            self._v = Vector()
            self._o = Orientation()
        elif len(args) == 1 :
            arg = args[0]
            if type(arg) == Transform or \
                   hasattr(arg,'pos') and hasattr(arg,'orient') and \
                   Orientation.canCreateOn(arg.orient) and \
                   Vector.canCreateOn(arg.pos):
                t = arg
                self._v = t.pos.copy()
                self._o = t.orient.copy()
            elif type(arg) == np.ndarray and arg.shape==(4,4):
                self._o = Orientation(arg[:3,:3].copy())
                self._v = Vector(arg[:3,3].copy())
            elif type(arg) == np.ndarray and arg.shape==(6,):
                # # Assume a pose vector of 3 position vector and 3 rotation vector components
                self._v = Vector(arg[:3])
                self._o = Orientation(arg[3:])
            else:
                raise self.Error(
                    'Could not create Transform on arguments : "'
                    + str(args) + '"')
        elif len(args) >= 1:
            self.__createOnSequence(args)
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
            elif type(value) == Orientation:
                self._data[:3,:3] = value._data
        elif name == 'pos':
            if type(value) == np.ndarray:
                self._data[:3,3] = value
            elif type(value) == Vector:
                self._data[:3,3] = value._data
        else:
            object.__setattr__(self, name, value)
            
    def __copy__(self):
        """ Copy method for creating a (deep) copy of this Transform."""
        return Transform(self)
    
    def __deepcopy__(self, memo):
        return self.__copy__()

    def copy(self, other=None):
        """ Copy data from other to self. """
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
        elif type(other) == Vector:
            return (self.pos, other)
        elif type(other) == Orientation:
            return (self.orient, other)
        else:
            return None
        
    def __eq__(self,other):
        if type(other) == Transform:
            return np.sum((self._data-other._data)**2) < utils._eps
        else:
            raise self.Error('Could not compare to non-Transform!')

    def fromXYP(self, cx, cy, p):
        """ Make this transform correspond to the orientation given by
        the given 'cx' and 'cy' directions and translation given by
        'p'."""
        self._o.fromXY(cx, cy)
        self._v.pos = p
        self._setFromOV(self._o, self._v)
        
    def fromXZP(self, cx, cz, p):
        """ Make this transform correspond to the orientation given by
        the given 'cx' and 'cz' directions and translation given by 'p'."""
        self._o.fromXZ(cx, cz)
        self._v.pos = p
        self._setFromOV(self._o, self._v)

    def dist2(self, other):
        """ Return the square of the metric distance, as unweighted
        combined linear and angular distance, to the 'other'
        transform. Note that the units and scale among linear and
        angular representations matters heavily."""
        return self._v.dist2(other._v) + self._o.angDist2(other._o)
    
    def dist(self, other):
        """ Return the metric distance, as unweighted combined linear
        and angular distance, to the 'other' transform. Note that the
        units and scale among linear and angular representations
        matters heavily."""
        return np.sqrt(self.dist2(other))

    def inverse(self):
        """ Return an inverse of this Transform."""
        #io = self._o.inverse()
        #iv = io * (-self._v)
        #return Transform(io, iv)
        return Transform(np.linalg.inv(self._data))
    
    def invert(self):
        """ In-place invert this Transform."""
        #self._o.invert()
        # self._v = self._o * (-self._v)
        self._data[:,:]=np.linalg.inv(self._data)
        
    def __mul__(self, other):
        """ Multiplication of self with another Transform or operate
        on a Vector given by 'other'."""
        if type(other) == Transform:
            #o = self._o * other._o
            #v = self._o * other._v + self._v
            #return Transform(o, v)
            return Transform(np.dot(self._data, other._data))
        elif type(other) == Vector:
            #return self._o * other + other._isPosition * self._v
            v = np.ones(4)
            v[:3] = other._data
            return Vector(np.dot(self._data, v)[:3])
        elif type(other) == np.ndarray and other.shape == (3,):
            return np.dot(self._o._data, other)+self._v._data
        elif isSequence(other):
            return list(map(self.__mul__,other))
        else:
            raise self.Error('Inadequate data type for multiplication in "other" : "'
                             + str(type(other)) + '"')
        
    def toArray(self):
        """ Return a tuple pair of an 3x3 orientation array and
        position as 3-array."""
        #return (self._o._data.copy(),self._v._data.copy())
        return (self._data[:3,:3], self._data[:3,3])
    
    def toList(self):
        """ Return a list with orientation and position in list form."""
        #return [self._o._data.tolist(),self._v._data.tolist()]
        return [self._data[:3,:3].tolist(), self._data[:3,3].tolist()]
    
def newTransFromXYP(cx, cy, p):
    """ Create a transform corresponding to the orientation given by
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
    cx = Vector(2, 3, 0)
    cz = Vector.e2
    p = Vector(1, 2, 3)
    t = newTransFromXZP(cx, cz, p)
    print((t*cx))
    it = t.inverse()

if __name__ == '__main__':
    _test()
