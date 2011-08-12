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
Module implementing the Vector class.
"""

import numpy

from math3d.utils import isNumTypes, isNumType, isSequence, isThreeSequence

def isVector(v):
    if __debug__: print('Deprecation warning: "isVector(v)". '
                        + 'Use "return type(v) == Vector"')
    return type(v) == Vector

class Vector(object):
    """ A Vector is a 3D vector (member of R3) with standard euclidian
    operations."""
    
    class Error(Exception):
        """ Exception class."""
        def __init__(self, message):
            self.message = message
        def __repr__(self):
            return self.__class__ + '.Error :' + self.message

    @classmethod
    def canCreateOn(cls, *arg):
        if type(arg) == cls:
            return True
        elif isSequence(arg):
            if len(arg) <= 3 and isNumTypes(arg):
                return True
            elif len(arg) == 1:
                return cls.canCreateOn(*arg[0])
            else:
                return False
        else:
            return False

    def __init__(self, *args, **kwargs):
        """ Constructor for Vector. If optional keyword argument
        'position' is evaluates to true, or is not given, the vector
        is represented as a position vector. Otherwise it is
        represented as a real vector."""
        if len(args) == 3 and isNumTypes(args):
            self._data=numpy.array(list(map(float, args)))
        elif len(args) == 2 and isNumTypes(args):
            self._data = numpy.array(list(map(float,[args[0], args[1], 0])))
        elif len(args) == 1:
            arg = args[0]
            if isThreeSequence(arg):
                self._data = numpy.array(list(map(float, arg)))
            elif isSequence(arg) and len(arg)  == 2:
                self._data = numpy.array(list(map(float, [arg[0], arg[1], 0])))
            elif type(arg) == Vector:
                self._data = arg._data.copy()
            else:
                raise self.Error('__init__ : could not create vector on argument : ' + str(args[0]))
        else:
            self._data = numpy.array([0.0,0.0,0.0])
        self._isPosition = 1
        if 'position' in kwargs:
            if kwargs['position']: self._isPosition = 1
            else: self._isPosition = 0
    
    def __copy__(self):
        """ Copy method for creating a copy of this Vector."""
        return Vector(self)
    
    def __deepcopy__(self, memo):
        return self.__copy__()
    
    def copy(self, other=None):
        """ Copy data from 'other' to self. """
        if other is None:
            return Vector(self)
        else:
            self._data[:] = other._data.copy()

    def __getattr__(self,name):
        if name == 'data':
            return self._data.copy()
        elif name == 'x':
            return self._data[0]
        elif name == 'y':
            return self._data[1]
        elif name == 'z':
            return self._data[2]
        else:
            raise AttributeError('Attribute "%s" not found in Vector'%name)
        
    def __setattr__(self,name,val):
        if name == 'x':
            self._data[0] = val
        elif name == 'y':
            self._data[1] = val
        elif name == 'z':
            self._data[2] = val
        elif name == '_data':
            self.__dict__[name] = val
        elif name == 'pos':
            if type(val) == Vector:
                self._data = copy(val._data)
            elif isThreeSequence(val):
                self._data = numpy.array(val)
        else:
            object.__setattr__(self, name, val)

    def __coerce__(self, other):
        return None

    def __getitem__(self, n):
        return self._data[n]

    def __setitem__(self,n,val):
        self._data[n] = val

    def __cmp__(self, other):
        if self.x == other.x and self.y == other.y and self.z == other.z: return 0
        else: return cmp(self.x,other.x)

    def __eq__(self,other):
        if type(other) == Vector:
            return numpy.sum((self._data-other._data)**2) < utils._eps
        else:
            raise self.Error('Could not compare to non-Vector!')

    def __repr__(self):
        return '<Vector: x=%f y=%f z=%f>'%(self.x,self.y,self.z)

    def __str__(self):
        return self.__repr__()
    
    def isPos(self):
        """ If the vector is a position vector, default, then it
        transforms differently than a real vector."""
        return self._isPosition

    def angle(self, other):
        """ Return the angle (radians) to the 'other' vector. This is the
        absolute, positive angle."""
        costheta = (self * other) / (self.length() * other.length())
        if costheta > 1:
            costheta = 1
        elif costheta < -1:
            costheta = -1
        return numpy.arccos(costheta)

    def sangle(self, other, refVec=None):
        """ With default reference rotation vector as Z-axis (if
        'refVec' == None), compute the signed angle of rotation from
        self to 'other'."""
        theta = self.angle(other)
        xprod = self.cross(other)
        if not refVec is None:
            if xprod*refVec < 0:
                theta = -theta
        else:
            if xprod.z < 0:
                theta = -theta
        return theta
    
    def length(self):
        """ Standard Euclidean length."""
        return numpy.sqrt(self.length2())

    def length2(self):
        """ Square of the standard Euclidean length."""
        return numpy.dot(self._data, self._data)

    def normalize(self):
        """ In-place normalization of this Vector."""
        l = self.length()
        if l != 1.0:
            self._data = self._data / l

    def normalized(self):
        """ Returns a normalized Vector with same direction as this
        one."""
        nv = Vector(self)
        nv.normalize()
        return nv

    def dist(self, other):
        """ Compute euclidean distance between points given by self
        and 'other'."""
        return numpy.sqrt(self.dist2(other))
    
    def dist2(self, other):
        """ Compute euclidean distance between points given by self
        and 'other'."""
        return (self - other).length2()


    def cross(self, other):
        return Vector(numpy.cross(self._data, other._data))

    def __sub__(self, other):
        if type(other) == Vector:
            return Vector(numpy.subtract(self._data, other._data))

    def __isub__(self, other):
        if type(other) == Vector:
            self._data -= other._data
        return self
    
    def __mul__(self, other):
        """ Multiplication with an 'other' Vector (inner product) or
        with a scalar."""
        if type(other) == Vector:
            return numpy.dot(self._data, other._data)
        elif isNumType(other):
            return Vector(numpy.dot(self._data, other))

    def __imul__(self, other):
        """ Inplace multiplication with a scalar, 'other'. """
        if isNumType(other):
            self._data *= other
        else:
            raise self.Error('__imul__ : Could not multiply by non-number')
        return self
    
    def __rmul__(self, other):
        """ Right multiplication with a scalar, 'other'. """
        if isNumType(other):
            return Vector(other * self._data)
        else:
            raise self.Error('__rmul__ : Could not multiply by non-number')
        
    def __div__(self, other):
        """ Division with a scalar, 'other'. """
        if isNumType(other):
            return Vector(1.0 / other * self._data)
        else:
            raise self.Error('__rdiv__ : Could not divide by non-number')
        
    def __add__(self, other):
        """ Return the sum of this and the 'other' vector."""
        if type(other) == Vector:
            return Vector(self._data + other._data)
        else:
            raise self.Error('__add__ : Could not add non-vector')

    def __iadd__(self, other):
        """ In-place add the 'other' vector to this vector."""
        if type(other) == Vector:
            self._data += other._data
        else:
            raise self.Error('__iadd__ : Could not add non-vector')
        return self

    def __neg__(self):
        return Vector(-self._data)

# Unit Vectors
Vector.e0 = Vector(1,0,0)
Vector.e1 = Vector(0,1,0)
Vector.e2 = Vector(0,0,1)
    
def _test():
    print((Vector.canCreateOn(1,2,3), Vector.canCreateOn((1,2,3)), Vector.canCreateOn(1,2)))
