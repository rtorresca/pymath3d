"""
Module implementing the Orientation class. The orientation is
represented internally by an orthogonal 3x3 matrix.
"""

__author__ = "Morten Lind"
__copyright__ = "Morten Lind 2012"
__credits__ = ["Morten Lind"]
__license__ = "GPLv3"
__maintainer__ = "Morten Lind"
__email__ = "morten@lind.dyndns.dk"
__status__ = "Production"

import numpy as np

# # Circular dependencies prevents direct import of Quaternion, hence
# # global addressing
import math3d as m3d

from . import utils
from .vector import Vector

def isOrientation(o):
    utils._deprecation_warning('type(o) == math3d.Orientation')
    return type(o) == Orientation


class Orientation(object):
    """An Orientation is a member of SO(3) which can be used either to
    perform a rotational transformation, or for keeping an orientation
    in 3D.
    """
    
    class Error(Exception):
        """Exception class."""
        def __init__(self, message):
            self.message = message
            Exception.__init__(self, self.message)
        def __repr__(self):
            return self.message

    def __create_on_sequence(self, seq):
        if type(seq) in (list, tuple):
            seq = np.array(seq)
        if type(seq) != np.ndarray:
            raise self.Error('Creating on a sequence requires numpy array, '
                             + 'list or tuple')
        if seq.shape in ((9,), (3,3)):
            self._data = seq.copy()
            # // Ensure the right shape.
            self._data.shape = (3,3)
        elif seq.shape == (3,):
            self._data = np.identity(3)
            self.from_rotation_vector(seq)
        else:
            raise self.Error('Creating on a numpy array requires shape '
                             + '(3,), (9,) or (3,3)!')

    def __init__(self, *args):
        """Create an orientation on either of the following arguments:

        * An Orientation.

        * A Quaternion.

        * Three Vectors or numpy arrays of shape (3,) interpreted as
          columns of the matrix.

        * One Vector, numpy array, list, or tuple of shape (3,)
          interpreted as a rotation vector.

        * A numpy array, list, or tuple of shape (3,3) or (9,) for
          giving direct matrix data; using row major order.
        """
        if len(args) == 1:
            arg=args[0]
            if type(arg) == Orientation:
                self._data = arg.data
            elif type(arg) == m3d.Quaternion:
                self._data = arg.orientation._data
            elif type(arg) == Vector:
                # Interpret as a rotation vector
                self._data = np.identity(3)
                self.from_rotation_vector(arg)
            elif utils.is_sequence(arg):
                self.__create_on_sequence(arg)
            else:
                raise self.Error(
                    'Creating on type {} is not supported'
                    .format(str(type(arg))))
        elif len(args) == 3:
            if np.all(np.array([type(a)==Vector for a in args])):
                array_args = (a._data for a in args)
            elif np.all(np.array([type(a)==np.ndarray for a in args])):
                array_args = args
            else:
                raise self.Error(
                    'Creating on three arguments requires three vectors '
                    + 'or three numpy arrays of shape (3,)!')
            # Stack the vector data vertically and transpose to get
            # them into columns.
            self._data = np.transpose(np.vstack((va for va in array_args)))
        elif len(args) == 0:
            self._data = np.identity(3)
        # Always ensure that we use float64 as fundamental type
        self._data = self._data.astype(np.float64)

    def __copy__(self):
        """Copy method for creating a copy of this Orientation."""
        return Orientation(self)

    def __deepcopy__(self, memo):
        return self.__copy__()

    def copy(self, other=None):
        """Copy data from 'other' to self. If no argument given,
        i.e. 'other==None', return a copy of this Orientation
        """
        if other is None:
            return Orientation(self)
        else:
            self._data[:,:] = other._data

    @property
    def vec_x(self):
        """Return the x-direction of the moving coordinate system in
        base reference as a Vector.
        """
        return Vector(self._data[:,0])

    @property
    def col_x(self):
        """Return the x-direction of the moving coordinate system in
        base reference as an array."""
        return self._data[:,0]
    
    @property
    def vec_y(self):
        """Return the y-direction of the moving coordinate system in
        base reference as a Vector.
        """
        return Vector(self._data[:,1])

    @property
    def col_y(self):
        """Return the y-direction of the moving coordinate system in
        base reference as an array.
        """
        return self._data[:,1]
    
    @property
    def vec_z(self):
        """Return the z-direction of the moving coordinate system in
        base reference as a Vector.
        """
        return Vector(self._data[:,2])

    @property
    def col_z(self):
        """Return the z-direction of the moving coordinate system in
        base reference as an array.
        """
        return self._data[:,2]

    def __getattr__(self, name):
        if name == 'data':
            return self._data.copy()
        elif name[:3] in ['vec', 'col'] and name[-1].lower() in 'xyz':
            idx = 'xyz'.find(name[-1].lower())
            a = self._data[:,idx]
            if name[:3] == 'vec':
                a = Vector(a)
            return a
        else:
            raise AttributeError(
                'Attribute "{}" not found in Orientation'.format(name))
            #raise self.Error, 'Orientation does not have attribute "%s"' % name

    def __getitem__(self, indices):
        return self._data.__getitem__(indices)

    # def __coerce__(self, other):
    #     if type(other) == Orientation:
    #         return (self, other)
    #     else:
    #         return None

    def __eq__(self,other):
        if type(other) == Orientation:
            return np.sum((self._data-other._data)**2) < utils._eps
        else:
            return NotImplemented
            # raise self.Error('Could not compare to non-Orientation!')

    def __setattr__(self, name, val):
        if name == '_data':
            ## This is dangerous, since there is no consistency check.
            self.__dict__['_data']=val
        elif name[:3] in ['vec', 'col'] and name[-1].lower() in 'xyz':
            ## This is dangerous since there is no automatic
            ## re-normalization
            idx = 'xyz'.find(name[-1].lower())
            if type(val) == Vector:
                val = val.data
            self._data[:3,idx] = val
        else:
            object.__setattr__(self, name, val)

    @property
    def error(self):
        """Compute and return the square root of the sum of squared
        dot products of the axis vectors, as a representation of the
        error of the orientation matrix.
        """
        vec_x = self.vec_x
        vec_y = self.vec_y
        vec_z = self.vec_z
        sq_sum = (vec_x*vec_y)**2
        sq_sum += (vec_y*vec_z)**2
        sq_sum += (vec_z*vec_x)**2
        return np.sqrt(sq_sum)
    
    # def renormalize(self):
    #     """Correct the axis vectors by a Gram-Schmidt procedure."""
    #     colx = self._data[:,0]
    #     coly = self._data
        

    def from_xy(self, x_vec, y_vec):
        """Reset this orientation to the one that conforms with the
        given x and y directions.
        """
        self.vec_x = x_vec.normalized()
        self.vec_y = y_vec.normalized()
        self.vec_z = x_vec.cross(y_vec).normalized()
        ## A last normalization check!
        #if self.vec_x.dist(self.vec_y.cross(self.vec_z)) > utils._eps:
        self.vec_x=self.vec_y.cross(self.vec_z).normalized()
    def fromXY(self, x_vec, y_vec):
        utils._deprecation_warning('fromXY -> from_xy')
        self.from_xy(x_vec, y_vec)
        
    def from_xz(self, x_vec, z_vec):
        """Reset this orientation to the one that conforms with the
        given x and z directions."""
        if x_vec * z_vec > utils._eps:
            print('warning ... orthogonalizing!')
        self.vec_x = x_vec.normalized()
        self.vec_z = z_vec.normalized()
        self.vec_y = z_vec.cross(x_vec).normalized()
        ## A last normalization check!
        #if self.vec_x.dist(self.vec_y.cross(self.vec_z)) > utils._eps:
        self.vec_x = self.vec_y.cross(self.vec_z)
    def fromXZ(self, x_vec, z_vec):
        utils._deprecation_warning('fromXZ -> from_xz')
        self.from_xz(x_vec, z_vec)

    @property
    def rotation_vector(self):
        """Return a rotation vector representing this
        orientation. This is essentially the logarithm of the rotation
        matrix. """
        return self.quaternion.rotation_vector

    @property
    def quaternion(self):
        """Return a quaternion representing this orientation."""
        return m3d.Quaternion(self)

    def toRotationVector(self):
        """Return a rotation vector representing this
        orientation. This is essentially the logarithm of the rotation
        matrix."""
        utils._deprecation_warning('toRotationVector() -> [prop] rotation_vector')
        return self.rotation_vector

    def from_rotation_vector(self, rot_vec):
        """Set this Orientation to represent the one given in a
        rotation vector in 'rot_vec'. 'rot_vec' must be a Vector or an
        numpy array of shape (3,)
        """
        if type(rot_vec) == Vector:
            rot_vec = rot_vec.data
        angle = np.linalg.norm(rot_vec)
        if np.abs(angle) < utils._eps:
            self._data = np.identity(3)
        else:
            axis = rot_vec / angle
            self.from_axis_angle(axis, angle)
    def fromRotationVector(self, rot_vec):
        utils._deprecation_warning('fromRotationVector() -> from_rotation_vector()')
        self.from_rotation_vector(rot_vec)
        
    @property 
    def axis_angle(self):
        """Return an (axis,angle) pair representing the equivalent
        orientation."""
        return m3d.Quaternion(self).axis_angle
    def toAxisAngle(self):
        """Return an (axis,angle) pair representing the equivalent
        orientation."""
        utils._deprecation_warning('toAxisAngle() -> [prop] axis_angle')
        return self.axis_angle

    def from_axis_angle(self, axis, angle):
        """Set this orientation to the equivalent to rotation of
        'angle' around 'axis'.
        """
        if type(axis) == Vector:
            axis = axis.data
        ## Force normalization
        axis /= np.linalg.norm(axis)
        x = axis[0]
        y = axis[1]
        z = axis[2]
        ct =np.cos(angle)
        st = np.sin(angle)
        self._data[:,:] = np.array([ \
            [ct + (1 - ct) * x**2,
             (1 - ct) * x * y - st *z,
             (1 - ct) * x * z + st * y],
            [(1 - ct) * x * y + st * z,
             ct + (1 - ct) * y**2,
             (1 - ct) * y * z - st * x],
            [(1 - ct) * x * z - st * y,
             (1 - ct) * y * z + st * x,
             ct + (1 - ct) * z**2]])
    def fromAxisAngle(self, axis, angle):
        utils._deprecation_warning('fromAxisAngle() -> from_axis_angle()')
        
    def set_to_x_rotation(self, angle):
        """Replace this orientation by that of a rotation around x."""
        ca = np.cos(angle)
        sa = np.sin(angle)
        self._data[:,:] = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    def rotX(self, angle):
        utils._deprecation_warning('rotX() -> set_to_x_rotation()')
        return self.set_to_x_rotation(angle)
    
    def set_to_y_rotation(self, angle):
        """Replace this orientation by that of a rotation around y."""
        ca=np.cos(angle)
        sa=np.sin(angle)
        self._data[:,:] = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
    def rotY(self, angle):
        utils._deprecation_warning('rotY() -> set_to_y_rotation()')
        return self.set_to_y_rotation(angle)

    def set_to_z_rotation(self, angle):
        """Replace this orientation by that of a rotation around z."""
        ca = np.cos(angle)
        sa = np.sin(angle)
        self._data[:,:] = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    def rotZ(self, angle):
        utils._deprecation_warning('rotZ() -> set_to_z_rotation()')
        return self.set_to_z_rotation(angle)

    def rotate_t(self, axis, angle):
        """In-place rotation of this orientation angle radians in axis
        perceived in the transformed reference system.
        """
        o = Orientation()
        o.from_axis_angle(axis, angle)
        self.copy(self * o)
    rotate = rotateT = rotate_t
    
    def rotate_b(self, axis, angle):
        """In-place rotation of this orientation angle radians in axis
        perceived in the base reference system.  Arguments:
        axis -- the axis to rotate about (unit vector with direction).
        angle -- the angle in radians to rotate about the axis.
        """
        o = Orientation()
        o.from_axis_angle(axis, angle)
        self.copy(o * self)
    rotateB = rotate_b
    
    def rotate_xb(self, angle):
        """In-place rotation of this oriantation by a rotation around
        x axis in the base reference system. (Inefficient!)
        """
        self.rotate_b(Vector.e0, angle)
    rotateXB = rotate_xb
    
    def rotate_yb(self, angle):
        """In-place rotation of this oriantation by a rotation around
        y axis in the base reference system. (Inefficient!)
        """
        self.rotate_b(Vector.e1, angle)
    rotateYB = rotate_yb
    
    def rotate_zb(self, angle):
        """In-place rotation of this oriantation by a rotation around
        z axis in the base reference system. (Inefficient!)
        """
        self.rotate_b(Vector.e2, angle)
    rotateZB = rotate_zb
    
    def rotate_xt(self, angle):
        """In-place rotation of this oriantation by a rotation around
        x axis in the transformed reference system. (Inefficient!)
        """
        self.rotate_t(Vector.e0, angle)
    rotate_x = rotateX = rotateXT = rotate_xt
    
    def rotate_yt(self,angle):
        """In-place rotation of this oriantation by a rotation around
        y axis in the transformed reference system. (Inefficient!)
        """
        self.rotate_t(Vector.e1,angle)
    rotate_y = rotateY = rotateYT = rotate_yt
    
    def rotate_zt(self, angle):
        """In-place rotation of this oriantation by a rotation around
        z axis in the transformed reference system. (Inefficient!)
        """
        self.rotate_t(Vector.e2, angle)
    rotate_z = rotateZ = rotateZT = rotate_zt
    
    def __repr__(self):
        return '<Orientation: \n' + repr(self._data) + '>'

    def __str__(self):
        return self.__repr__()

    def ang_dist_sq(self, other):
        """Return the square of the orientation distance (the angle of
        rotation) to the 'other' orientation.
        """
        return (self.inverse()*other).rotation_vector.length_sq
    def angDist2(self, other):
        utils._deprecation_warning('angDist2 -> ang_dist_sq')
        return self.ang_dist_sq(other)

    def ang_dist(self, other):
        """Return the orientation distance (the angle of rotation) to
        the 'other' orientation.
        """
        return np.sqrt(self.ang_dist_sq(other))
    def angDist(self, other):
        utils._deprecation_warning('angDist -> ang_dist')
        return self.ang_dist(other)

    def invert(self):
        """In-place inversion of this orientation."""
        self._data[:,:] = self._data.transpose().copy()

    def inverse(self):
        """Return an inverse of this orientation as a rotation."""
        o = Orientation(self._data)
        o.invert()
        return o

    def __mul__(self, other):
        if type(other) == Orientation:
            return Orientation(np.dot(self._data, other._data))
        elif type(other) == Vector:
            return Vector(np.dot(self._data, other._data))
        elif type(other) == np.ndarray and other.shape == (3,):
            return np.dot(self._data, other)
        elif utils.is_sequence(other):
            return [self * o for o in other]
        else:
            return NotImplemented
            # raise self.Error('Multiplication by something other than'
            #                  + 'Orientation, Vector, or a sequence '
            #                  + 'of these, is not allowed!')

    @property
    def matrix(self):
        """Property for getting a np-matrix with the data from the
        orientation."""
        return np.matrix(self._data)

    @property
    def array(self):
        """Return a copy of the ndarray which is the fundamental data
        of the Orientation."""
        return self._data.copy()

    @property
    def list(self):
        """Return the fundamental data of the Orientation as a
        list."""
        return self._data.tolist()

    @classmethod
    def new_from_xy(cls, x_vector, y_vector):
        """Factory for a new orientation with given x- and
        y-direction."""
        o = Orientation()
        o.from_xy(x_vector, y_vector)
        return o

    @classmethod
    def new_from_xz(cls, x_vector, z_vector):
        """Factory for a new orientation with given x- and
        z-direction."""
        o = Orientation()
        o.from_xz(x_vector, z_vector)
        return o

    @classmethod
    def new_rot_x(cls, angle):
        """Factory for a new orientation which is a rotation in the
        signed angle 'angle' around the x-direction.
        """
        o = Orientation()
        o.set_to_x_rotation(angle)
        return o

    @classmethod
    def new_rot_y(cls, angle):
        """Factory for a new orientation which is a rotation in the
        signed angle 'angle' around the y-direction.
        """
        o = Orientation()
        o.set_to_y_rotation(angle)
        return o

    @classmethod
    def new_rot_z(cls, angle):
        """Factory for a new orientation which is a rotation in the
        signed angle 'angle' around the z-direction.
        """
        o = Orientation()
        o.set_to_z_rotation(angle)
        return o

    # // The Euler encoding map is used to easily apply rotations
    # according the the 'xyzXYZ' encoding scheme for Euler angles
    _euler_encoding_map = {'x':rotate_xb, 'y':rotate_yb, 'z':rotate_zb,
                           'X':rotate_xt, 'Y':rotate_yt, 'Z':rotate_zt,}

    @classmethod
    def new_vec_to_vec(cls, from_vec, to_vec):
        """Factory for a new orientation which is the rotation in the
        signed angle 'angle' around the z-direction which rotates
        'from_vec' to 'to_vec'."""
        angle = from_vec.angle(to_vec)
        if angle <= 1.0e-8:
            # // Identity
            return Orientation()
        elif angle < np.pi-1.0e-8:
            # // Regular, minimal rotation
            return Orientation(angle * from_vec.cross(to_vec).normalized())
        else:
            # // Find a suitable rotation axis
            x_angle = Vector.ex.angle(from_vec)
            if x_angle > 1e-3 and x_angle < np.pi - 1.0e-3:
                return Orientation(angle * 
                                   Vector.ex.cross(from_vec).normalized())
            y_angle = Vector.ey.angle(from_vec)
            if y_angle > 1e-3 and y_angle < np.pi - 1.0e-3:
                return Orientation(angle * 
                                   Vector.ey.cross(from_vec).normalized())
            z_angle = Vector.ez.angle(from_vec)
            if z_angle > 1e-3 and z_angle < np.pi - 1.0e-3:
                return Orientation(angle * 
                                   Vector.ez.cross(from_vec).normalized())

    def new_euler(cls, angles, encoding):
        """Factory for generating a new orientation from Euler or
        Tait-Bryan angles. 'angles' must be a sequence of three real
        numbers giving the Euler or Tait-Bryan angles. 'encoding' must
        be three characters, all from the set 'xyzXYZ'. The encoding
        denotes the sequence of axes to rotate around and the case of
        the characters in the encoding string tells if it should be
        intrinsic or extrinsic axes for the rotation; all rotations
        must be either intrinsic or extrinsic. Here the notation is
        adopted from 'http://en.wikipedia.org/wiki/Euler_angles'. A
        lower case character, e.g. 'x', denotes a rotation around the
        extrinsic axis, i.e. the given axis of the initial coordinate
        system. An upper-case character, e.g. 'X', denotes a rotation
        around the axis in the intrinsic, i.e. moved, coordinate
        system at that particular instance of the sequence. A
        classical example of proper Euler angles are alpha-beta-gamma
        angles consisting of all intrinsic rotations, first alpha
        around the inital z-axis, then beta around rotated x-axis, and
        finally gamma around rotated z-axis; this is encoded by
        'ZXZ'. Note that proper Euler angles always address two
        different axes, the same (intrinsic) axis is used for the
        first and third rotation. Tait-Bryan angles address three
        different axes and classical examples are roll-pitch-yaw,
        which are encoded as 'ZYX', or yaw-pitch-roll, encoded
        by'XYZ'. Any sequence of intrinsic rotations may be converted
        to a corresponding sequence of extrinsic rotations by
        reversing the angle sequence; e.g. arguments
        ((alpha,beta,gamma), 'ZYX') gives the same rotation as
        ((gamma,beta,alpha), 'zyx').
        """
        enc = encoding
        # All rotations must either be intrinsic or extrinsic
        if enc.upper() == enc:
            intrinsic = True
        elif enc.lower() == enc:
            intrinsic = False
        else:
            raise self.Error(
                'Rotation encoding must either be all intrinsic or extrinsic!')
        o = Orientation()
        for r,a in zip(encoding, angles):
            cls._euler_encoding_map[r](o, a)
        return o

    def to_euler(self, encoding):
        """The Euler angles 'encoding' follow the documentation for
        the factory method 'new_euler'. The routine is taken from Ken
        Shoemake's chapter 'Euler Angle Conversion' in 'Graphics Gems
        IV', Academic Press, 1994, ISBN 0-12-336155-9.
        """
        enc = encoding
        # All rotations must either be intrinsic or extrinsic
        if enc.upper() == enc:
            intrinsic = True
        elif enc.lower() == enc:
            intrinsic = False
        else:
            raise self.Error(
                'Rotation encoding must either be all intrinsic or extrinsic!')
        lenc = enc.lower()
        repetition = lenc[0] == lenc[2]
        if intrinsic:
            parity = lenc[1:] not in ['yx', 'zy', 'xz']
        else:
            parity = lenc[:2] not in ['xy', 'yz', 'zx']
        inner = lenc[2] if intrinsic else lenc[0] 
        i = 'xyz'.index(inner)
        j = (i + 1 + parity) % 3
        k = (i + 2 - parity) % 3
        # h = k if repetition else i
        m = self._data
        if repetition:
            sy = np.sqrt(m[i, j]**2 + m[i, k]**2)
            if sy > 16 * np.finfo(np.float32).eps:
                ax = np.arctan2(m[i, j], m[i, k])
                ay = np.arctan2(sy, m[i, i])
                az = np.arctan2(m[j, i], -m[k, i])
            else:
                ax = np.arctan2(-m[j, k], m[j, j])
                ay = np.arctan2(sy, m[i, i])
                az = 0.0
        else: # not repetition
            cy = np.sqrt(m[i, i]**2 + m[j, i]**2)
            if cy > 16 * np.finfo(np.float32).eps:
                ax = np.arctan2(m[k, j], m[k, k])
                ay = np.arctan2(-m[k, i], cy)
                az = np.arctan2(m[j, i], m[i, i])
            else:
                ax = np.arctan2(-m[j, k], m[j, j])
                ay = np.arctan2(-m[k, i], cy)
                az = 0.0
        if parity:
            ax, ay, az = -ax, -ay, -az
        if intrinsic:
            ax, az = az, ax
        return np.array([ax, ay, az])
                
def newOrientFromXY(x_vec, y_vec):
    """Create an orientation conforming with the given 'x' and 'y'
    directions."""
    utils._deprecation_warning('newOrientFromXY -> Orientation.new_from_xy')
    o = Orientation()
    o.from_xy(x_vec, y_vec)
    return o

def newOrientFromXZ(x_vec, z_vec):
    """Create an orientation conforming with the given 'x' and 'z'
    directions."""
    utils._deprecation_warning('newOrientFromXZ -> Orientation.new_from_xz')
    o = Orientation()
    o.from_xz(x_vec, z_vec)
    return o

def newOrientRotZ(angle):
    """Create an orientation corresponding to a rotation for 'angle'
    around the z direction."""
    utils._deprecation_warning('newOrientRotZ -> Orientation.new_rot_z')
    o = Orientation()
    o.set_to_z_rotation(angle)
    return o

def newOrientRotX(angle):
    """Create an orientation corresponding to a rotation for 'angle'
    around the x direction."""
    utils._deprecation_warning('newOrientRotX -> Orientation.new_rot_x')
    o = Orientation()
    o.set_to_x_rotation(angle)
    return o

def newOrientRotY(angle):
    """Create an orientation corresponding to a rotation for 'angle'
    around the y direction."""
    utils._deprecation_warning('newOrientRotY -> Orientation.new_rot_y')
    o = Orientation()
    o.set_to_y_rotation(angle)
    return o

def _test():
    o = Orientation()
    r = Orientation()
    o.from_xy(Vector(1, 1, 0), Vector(-1, 1, 0))
    r.set_to_z_rotation(np.pi / 2)
    ro = r * o
    print(ro.ang_dist(r))
    print(ro.axis_angle)

def _test_to_euler():
    ang = (0.5, 0.2, 0.1)
    enc = 'ZYZ'
    o = Orientation.new_euler(ang, enc)
    print(np.all(o.to_euler(enc) == np.array(ang)))
