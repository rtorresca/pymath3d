"""
Module implementing the Vector class.
"""

__author__ = "Morten Lind"
__copyright__ = "Morten Lind 2012"
__credits__ = ["Morten Lind"]
__license__ = "GPLv3"
__maintainer__ = "Morten Lind"
__email__ = "morten@lind.dyndns.dk"
__status__ = "Production"


import numpy as np

from . import utils

def isVector(v):
    utils._deprecation_warning('return type(v) == Vector')
    return type(v) == Vector

class Vector(object):
    """A Vector is a 3D vector (member of R3) with standard Euclidian
    operations."""
    
    class Error(Exception):
        """Exception class."""
        def __init__(self, message):
            self.message =  message
            Exception.__init__(self, self.message)
        def __repr__(self):
            return self.message

    @classmethod
    def canCreateOn(cls, *arg):
        if type(arg) == cls:
            return True
        elif utils.is_sequence(arg):
            if len(arg) <= 3 and utils.is_num_types(arg):
                return True
            elif len(arg) == 1:
                return cls.canCreateOn(*arg[0])
            else:
                return False
        else:
            return False

    def __init__(self, *args, **kwargs):
        """Constructor for Vector. If optional keyword argument
        'position' is evaluates to true, or is not given, the vector
        is represented as a position vector. Otherwise it is
        represented as a real vector."""
        if len(args) == 3 and utils.is_num_types(args):
            self._data=np.array(args)
        elif len(args) == 2 and utils.is_num_types(args):
            self._data = np.array((args[0], args[1], 0))
        elif len(args) == 1:
            arg = args[0]
            if utils.is_three_sequence(arg):
                self._data = np.array(arg)
            elif utils.is_sequence(arg) and len(arg)  == 2:
                self._data = np.array((arg[0], arg[1], 0))
            elif type(arg) == Vector:
                self._data = arg.data
            else:
                raise self.Error(
                    ('__init__ : could not create vector on argument : "{}"'
                    + ' of type "{}"').format(str(args[0]), str(type(args[0]))))
        else:
            self._data = np.array([0.0,0.0,0.0])
        self._isPosition = 1
        if 'position' in kwargs:
            if kwargs['position']: self._isPosition = 1
            else: self._isPosition = 0
        # Ensure data are float64
        self._data = self._data.astype(np.float64)

    def __copy__(self):
        """Copy method for creating a copy of this Vector."""
        return Vector(self)
    
    def __deepcopy__(self, memo):
        return self.__copy__()
    
    def copy(self, other=None):
        """Copy data from 'other' to self. If no argument given,
        i.e. 'other==None', return a copy of this Vector."""
        if other is None:
            return Vector(self)
        else:
            self._data[:] = other._data

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
        # elif name == '_data':
        #     # Important for initialization? Or would
        #     # object.__setattr__ take care of it?
        #     self.__dict__[name] = val
        elif name == 'pos':
            if type(val) == Vector:
                self._data[:] = val.data
            elif utils.is_three_sequence(val):
                self._data[:] = np.array(val)
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
            return np.sum((self._data-other._data)**2) < utils._eps
        else:
            raise self.Error('Could not compare to non-Vector!')

    def __repr__(self):
        return '<Vector: x=%f y=%f z=%f>'%(self.x,self.y,self.z)

    def __str__(self):
        return self.__repr__()
    
    def isPos(self):
        """If the vector is a position vector, default, then it
        transforms differently than a real vector."""
        return self._isPosition

    def angle(self, other):
        """Return the angle (radians) to the 'other' vector. This is the
        absolute, positive angle."""
        costheta = (self * other) / (self.length() * other.length())
        if costheta > 1:
            costheta = 1
        elif costheta < -1:
            costheta = -1
        return np.arccos(costheta)

    def sangle(self, other, refVec=None):
        """With default reference rotation vector as Z-axis (if
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
        """Standard Euclidean length."""
        return np.sqrt(self.length_sq)

    @property
    def length_sq(self):
        """Square of the standard Euclidean length."""
        return np.dot(self._data, self._data)
    def length2(self):
        utils._deprecation_warning('lenght2() -> [prop] length_sq')
        return self.length_sq
    
    def normalize(self):
        """In-place normalization of this Vector."""
        l = self.length()
        if l != 1.0:
            self._data = self._data / l

    def normalized(self):
        """Returns a normalized Vector with same direction as this
        one."""
        nv = Vector(self)
        nv.normalize()
        return nv

    def dist(self, other):
        """Compute euclidean distance between points given by self
        and 'other'."""
        return np.sqrt(self.dist2(other))
    
    def dist2(self, other):
        """Compute euclidean distance between points given by self
        and 'other'."""
        return (self - other).length_sq

    @property
    def cross_operator(self):
        """Return the cross product operator for this Vector. I.e. the
        skew-symmetric operator cross_op, such that cross_op * u == v x u, for any
        vector u."""
        cross_op = np.zeros((3,3))
        cross_op[0,1] = -self._data[2]
        cross_op[0,2] = self._data[1]
        cross_op[1,0] = self._data[2]
        cross_op[1,2] = -self._data[0]
        cross_op[2,0] = -self._data[1]
        cross_op[2,1] = self._data[0]
        return cross_op
    
    def cross(self, other):
        return Vector(np.cross(self._data, other._data))

    @property
    def array(self):
        """Return a copy of the ndarray which is the fundamental data
        of the Vector."""
        return self._data.copy()

    @property
    def list(self):
        """Return the fundamental data of the Vector as a list."""
        return self._data.tolist()
        
    @property
    def matrix(self):
        """Property for getting a single-column np-matrix with the data
        from the vector."""
        return np.matrix(self._data).T

    @property
    def column(self):
        """Property for getting a single-column array with the data
        from the vector."""
        return self._data.reshape((3,1))

    def __sub__(self, other):
        if type(other) == Vector:
            return Vector(np.subtract(self._data, other._data))

    def __isub__(self, other):
        if type(other) == Vector:
            self._data -= other._data
        return self
    
    def __mul__(self, other):
        """Multiplication with an 'other' Vector (inner product) or
        with a scalar."""
        if type(other) == Vector:
            return np.dot(self._data, other._data)
        elif utils.is_num_type(other):
            return Vector(np.dot(self._data, other))

    def __imul__(self, other):
        """In-place multiplication with a scalar, 'other'. """
        if utils.is_num_type(other):
            self._data *= other
        else:
            raise self.Error('__imul__ : Could not multiply by non-number')
        return self
    
    def __rmul__(self, other):
        """Right multiplication with a scalar, 'other'. """
        if utils.is_num_type(other):
            return Vector(other * self._data)
        else:
            raise self.Error('__rmul__ : Could not multiply by non-number')
        
    def __truediv__(self, other):
        """Division with a scalar, 'other'. """
        if utils.is_num_type(other):
            return Vector(1.0 / other * self._data)
        else:
            raise self.Error('__rdiv__ : Could not divide by non-number')
    __div__ = __truediv__
    
    def __add__(self, other):
        """Return the sum of this and the 'other' vector."""
        if type(other) == Vector:
            return Vector(self._data + other._data)
        else:
            raise self.Error('__add__ : Could not add non-vector')

    def __iadd__(self, other):
        """In-place add the 'other' vector to this vector."""
        if type(other) == Vector:
            self._data += other._data
        else:
            raise self.Error('__iadd__ : Could not add non-vector')
        return self

    def __neg__(self):
        return Vector(-self._data)

# Unit Vectors
Vector.ex = Vector.e0 = Vector(1,0,0)
Vector.ey = Vector.e1 = Vector(0,1,0)
Vector.ez = Vector.e2 = Vector(0,0,1)
    
def _test():
    print((Vector.canCreateOn(1,2,3), Vector.canCreateOn((1,2,3)), Vector.canCreateOn(1,2)))
