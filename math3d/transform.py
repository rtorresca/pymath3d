# coding=utf-8

"""
Module implementing a 3D homogenous Transform class. The transform is
represented internally by associated orientation and a vector objects.
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
from .vector import Vector
from .orientation import Orientation

def isTransform(t):
    utils._deprecation_warning('type(t) == math3d.Transform')
    return type(t) == Transform

class Transform(object):
    """A Transform is a member of SE(3), represented as a homogenous
    transformation matrix. It uses an Orientation in member '_o'
    (accessible through 'orient') to represent the orientation part
    and a Vector in member '_v' (accessible through 'pos') to
    represent the position part."""

    """A set of acceptable multi-value types for entering data."""
    __value_types = (np.ndarray, list, tuple)

    class Error(Exception):
        """Exception class."""
        def __init__(self, message):
            self.message = message
            Exception.__init__(self, self.message)
        def __repr__(self):
            return self.message

    def __create_on_sequence(self, arg):
        """Called from init when a single argument of sequence type
        was given the constructor."""
        # if len(arg) == 1 and utils.is_sequence(arg[0]):
        #     self.__createOnSequence(arg[0])
        if type(arg) in (tuple, list):
            self.__create_on_sequence(np.array(arg))
        elif type(arg) == np.ndarray and arg.shape in ((4,4), (3,4)):
            self._o = Orientation(arg[:3,:3])
            self._v = Vector(arg[:3,3])
        elif type(arg) == np.ndarray and arg.shape==(6,):
            # Assume a pose vector of 3 position vector and 3 rotation
            # vector components
            self._v = Vector(arg[:3])
            self._o = Orientation(arg[3:])
        else:
            raise self.Error(
                'Could not create Transform on arguments : "' + str(arg) + '"')
        
    def __init__(self, *args):
        """A Transform is a homogeneous transform on SE(3), internally
        represented by an Orientation and a Vector. A Transform can be
        constructed on:

        * A Transform.
        
        * A numpy array, list or tuple of shape (4,4) or (3,4) giving
          direct data; as [orient | pos].
        
        * A --''-- of shape (6,) giving a pose vector; concatenated
          position and rotation vector.
        
        * Two --''--; the first for orientation and the second for
          position.
        
        * Four --''--; the first three for orientation and the fourth
          for position.
        
        * Twelve numbers, the first nine used for orientation and the
          last three for position.
        
        * An ordered pair of Orientation and Vector.
        """
        if len(args) == 0:
            self._v = Vector()
            self._o = Orientation()
        elif len(args) == 1:
            arg = args[0]
            if type(arg) == Transform or \
                   hasattr(arg,'pos') and hasattr(arg,'orient'):
                self._v = Vector(arg.pos)
                self._o = Orientation(arg.orient)
            else:
                self.__create_on_sequence(arg)
        elif len(args) == 2:
            self._o = Orientation(args[0])
            self._v = Vector(args[1])
        elif len(args) == 4:
            self._o = Orientation(args[:3])
            self._v = Vector(args[3])
        elif len(args) == 12:
            self._o = Orientation(args[:9])
            self._v = Vector(args[9:])
        else:
            raise self.Error(
                'Could not create Transform on arguments : '
                + '"{}"'.format(str(args)))
        # Guard against reference to data.
        self._from_ov(self._o, self._v)
        
    def _from_ov(self, o, v):
        self._data = np.identity(4, dtype=np.float64)
        ## First take over the data from Orientation and Vector
        self._data[:3,:3] = o._data
        self._data[:3,3] = v._data
        ## Then share data with Orientation and Vector.
        self._o._data = self._data[:3,:3]
        self._v._data = self._data[:3,3]

    def get_pos(self):
        """Return a reference (Beware!) to the position object."""
        return self._v
    def set_pos(self, new_pos):
        """Set the position."""
        if type(new_pos) in self.__value_types:
            self._data[:3,3] = new_pos
        elif type(new_pos) == Vector:
            self._data[:3,3] = new_pos._data
        else:
            raise self.Error('Trying to set "pos" by an object of '
                             + 'type "{}". '.format(str(type(new_pos)))
                             + 'Needs tuple, list, ndarray, or Vector.')
    pos = property(get_pos, set_pos)

    def get_orient(self):
        """Return a reference (Beware!) to the orientation object."""
        return  self._o
    def set_orient(self, new_orient):
        """Set the orientation."""
        if type(new_orient) in self.__value_types:
            self._data[:3,:3] = new_pos
        elif type(new_orient) == Orientation:
            self._data[:3,:3] = new_orient._data
        else:
            raise self.Error('Trying to set "orient" by an object of '
                             + 'type "{}". '.format(str(type(new_pos)))
                             + 'Needs tuple, list, ndarray, or Orientation.')
    orient = property(get_orient, set_orient)

    def get_data(self):
        """Return a copy of the raw data."""
        utils._deprecation_warning('get_data -> get_array')
        return self._data.copy()
    #data = property(get_data)
                
    def __copy__(self):
        """Copy method for creating a (deep) copy of this
        Transform.
        """
        return Transform(self)
    
    def __deepcopy__(self, memo):
        return self.__copy__()

    def copy(self, other=None):
        """Copy data from 'other' to self. If no argument given,
        i.e. 'other==None', return a copy of this Transform.
        """
        if other is None:
            return Transform(self)
        else:
            self._data[:,:] = other._data
        
    def __repr__(self):
        return ('<Transform:\n{}\n{}\n>'
                .format(repr(self.orient), repr(self.pos)))

    def __str__(self):
        return self.__repr__()
        
    def __eq__(self,other):
        if type(other) == Transform:
            return np.sum((self._data-other._data)**2) < utils._eps
        else:
            return NotImplemented
            # raise self.Error('Could not compare to non-Transform!')

    def from_xyp(self, vec_x, vec_y, origo):
        """Make this transform correspond to the orientation given by
        the given 'vec_x' and 'vec_y' directions and translation given by
        'origo'."""
        self._o.from_xy(vec_x, vec_y)
        self._v = origo
        self._from_ov(self._o, self._v)
    def fromXYP(self, vec_x, vec_y, p):
        utils._deprecation_warning('fromXYP -> from_xyp')
        self.from_xyp(vec_x, vec_y, p)
        
    def from_xzp(self, vec_x, vec_z, origo):
        """Make this transform correspond to the orientation given by
        the given 'vec_x' and 'vec_z' directions and translation given
        by 'p'."""
        self._o.from_xz(vec_x, vec_z)
        self._v = origo
        self._from_ov(self._o, self._v)

    def dist_squared(self, other):
        """Return the square of the metric distance, as the unweighted
        sum of linear and angular distance, to the 'other'
        transform. Note that the units and scale among linear and
        angular representations matters heavily."""
        return self._v.dist_squared(other._v) + self._o.ang_dist(other._o) ** 2
    
    def dist(self, other):
        """Return the metric distance, as unweighted combined linear
        and angular distance, to the 'other' transform. Note that the
        units and scale among linear and angular representations
        matters heavily."""
        return np.sqrt(self.dist_squared(other))

    def get_inverse(self):
        """Return an inverse of this Transform."""
        return Transform(np.linalg.inv(self._data))
    inverse = property(get_inverse)

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
        elif type(other) == Vector:
            #return self._o * other + other._isPosition * self._v
            v = np.ones(4)
            v[:3] = other._data
            return Vector(np.dot(self._data, v)[:3])
        elif type(other) == np.ndarray and other.shape == (3,):
            return np.dot(self._o._data, other)+self._v._data
        elif utils.is_sequence(other):
            return list(map(self.__mul__,other))
        else:
            return NotImplemented
            # raise self.Error('Inadequate data type for multiplication '
            #                  + 'in "other" : %s' % str(type(other)))

    def get_pose_vector(self):
        """Get the transform in pose vector representation '(x, y, z,
        rx, ry, rz)'."""
        return np.append(self._v._data, self._o.rotation_vector)
    pose_vector = property(get_pose_vector)
    
    def get_structured_array(self):
        """Return a tuple pair of an 3x3 orientation array and
        position as 3-array."""
        #return (self._o._data.copy(),self._v._data.copy())
        return (self._data[:3,:3], self._data[:3,3])
    structured_array = property(get_structured_array)

    def get_structued_list(self):
        """Return a list with separate orientation and position in list form."""
        #return [self._o._data.tolist(),self._v._data.tolist()]
        return [self._data[:3,:3].tolist(), self._data[:3,3].tolist()]
    structued_list = property(get_structued_list)

    def get_matrix(self):
        """Property for getting a (4,4) np-matrix with the data
        from the transform."""
        return np.matrix(self._data)
    matrix = property(get_matrix)

    def get_array(self):
        """Return a copy of the (4,4) ndarray which is the fundamental
        data of the Transform. Caution: Use this method only for
        optimization, since it eliminates copying, and be sure not to
        compromize the data.
        """
        return self._data.copy()
    array = property(get_array)

    def get_array_ref(self):
        """Return a reference to the (4,4) ndarray, which is the
        fundamental data of the transform.
        """
        return self._data
    array_ref = property(get_array_ref)

    def get_list(self):
        """Return the fundamental data of the Transform as a list."""
        return self._data.tolist()
    list = property(get_list)

    @classmethod
    def new_from_xyp(self, vec_x, vec_y, origo):
        """Create a transform corresponding to the orientation given
        by the given 'vec_x' and 'vec_y' directions and translation given by
        'origo'."""
        t = Transform()
        t.from_xyp(vec_x, vec_y, origo)
        return t

    @classmethod
    def new_from_xzp(self, vec_x, vec_z, origo):
        """Create a transform corresponding to the orientation given
        by the given 'vec_x' and 'vec_z' directions and translation given by
        'origo'."""
        t = Transform()
        t.from_xzp(vec_x, vec_z, origo)
        return t


def newTransFromXYP(cx, cy, p):
    """Create a transform corresponding to the orientation given by
    the given 'cx' and 'cy' directions and translation given by 'p'."""
    utils._deprecation_warning('newTransFromXYP -> Transform.new_from_xyp')
    t = Transform()
    t.from_xyp(cx, cy, p)
    return t

def newTransFromXZP(cx, cz, p):
    """Create a transform corresponding to the orientation given by
    the given 'cx' and 'cz' directions and translation given by 'p'."""
    utils._deprecation_warning('newTransFromXZP -> Transform.new_from_xzp')
    t = Transform()
    t.from_xzp(cx, cz, p)
    return t

def _test():
    cx = Vector(2, 3, 0)
    cz = Vector.e2
    p = Vector(1, 2, 3)
    t = Transform.new_from_xzp(cx, cz, p)
    print(t*cx)
    it = t.inverse
    print(t*it)
