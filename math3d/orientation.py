"""
Module implementing the Orientation class. The orientation is
represented internally by an orthogonal 3x3 matrix.
"""

__author__ = "Morten Lind"
__copyright__ = "Morten Lind 2009-2012"
__credits__ = ["Morten Lind"]
__license__ = "GPL"
__maintainer__ = "Morten Lind"
__email__ = "morten@lind.no-ip.org"
__status__ = "Production"

import numpy as np

# # Circular dependencies prevents direct import of Orientation, hence
# # global addressing
import math3d as m3d
from .utils import isSequence,  isNumTypes,  _eps,  _deprecation_warning
from .vector import Vector

def isOrientation(o):
    print('Deprecation warning: "isOrientation(o)".'
          + ' Use "type(o) == math3d.Orientation".')
    return type(o) == Orientation

class Orientation(object):
    """An Orientation is a member of SO(3) which can be used either
    to perform a rotational transformation, or for keeping an
    orientation in 3D."""
    
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
            raise self.Error('Creating on a sequence requires numpy array, list or tuple')
        if seq.shape in ((9,), (3,3)):
            self._data = seq.copy()
            # // Ensure the right shape.
            self._data.shape = (3,3)
        elif seq.shape == (3,):
            self._data = np.identity(3)
            self.fromRotationVector(seq)
        else:
            raise self.Error('Creating on a numpy array requires shape (3,), (9,) or (3,3)!')

    def __init__(self, *args):
        """Create an orientation on either of the following arguments:
        * An Orientation.
        * A Quaternion.
        * Three Vectors or numpy arrays of shape (3,) interpreted as columns of the matrix.
        * One Vector, numpy array, list, or tuple of shape (3,) interpreted as a rotation vector.
        * A numpy array, list, or tuple of shape (3,3) or (9,) for giving direct matrix data; using row major order.
        """
        if len(args) == 1:
            arg=args[0]
            if type(arg) == Orientation:
                self._data = arg.data
            elif type(arg) == m3d.Quaternion:
                self._data = arg.orientation._data
            elif type(arg) == Vector:
                ## Interpret as a rotation vector
                self._data = np.identity(3)
                self.fromRotationVector(arg)
            elif isSequence(arg):
                self.__create_on_sequence(arg)
            else:
                raise self.Error('Creating on type %s is not supported' % str(type(arg)))
        elif len(args) == 3:
            if np.all(np.array([type(a)==Vector for a in args])):
                array_args = (a._data for a in args)
            elif np.all(np.array([type(a)==np.ndarray for a in args])):
                array_args = args
            else:
                raise self.Error('Creating on three arguments requires three vectors or three numpy arrays of shape (3,)!')
            # // Stack the vector data vertically and transpose to get them into columns.
            self._data = np.transpose(np.vstack((va for va in array_args)))
        elif len(args) == 0:
            self._data = np.identity(3)
        # // Always ensure that we use float64 as fundamental type
        self._data=self._data.astype(np.float64)

    def __copy__(self):
        """Copy method for creating a copy of this Orientation."""
        o = Orientation()
        o.copy(self)
        return o

    def __deepcopy__(self, memo):
        return self.__copy__()

    def copy(self, other=None):
        """Copy data from other to self. """
        if other is None:
            return copy(self)
        else:
            self._data[:,:] = other._data.copy()

    @property
    def vec_x(self):
        """ Return the x-direction of the moving coordinate system in
        base reference as a Vector."""
        return Vector(self._data[:,0])

    @property
    def col_x(self):
        """ Return the x-direction of the moving coordinate system in
        base reference as an array."""
        return self._data[:,0]
    
    @property
    def vec_y(self):
        """ Return the y-direction of the moving coordinate system in
        base reference as a Vector."""
        return Vector(self._data[:,1])

    @property
    def col_y(self):
        """ Return the y-direction of the moving coordinate system in
        base reference as an array."""
        return self._data[:,1]
    
    @property
    def vec_z(self):
        """ Return the z-direction of the moving coordinate system in
        base reference as a Vector."""
        return Vector(self._data[:,2])

    @property
    def col_z(self):
        """ Return the z-direction of the moving coordinate system in
        base reference as an array."""
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
            raise AttributeError('Attribute "%s" not found in Orientation'%name)
            #raise self.Error, 'Orientation does not have attribute "%s"' % name

    def __getitem__(self, indices):
        return self._data.__getitem__(indices)

    def __coerce__(self, other):
        if type(other) == Orientation:
            return (self, other)
        else:
            return None

    def __eq__(self,other):
        if type(other) == Orientation:
            return np.sum((self._data-other._data)**2) < _eps
        else:
            raise self.Error('Could not compare to non-Orientation!')

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

    def from_xy(self, x_vec, y_vec):
        """ Reset this orientation to the one that conforms with the
        given x and y directions."""
        if cx * cy > _eps:
            print('warning ... orthogonalizing!')
            #print ('%s %s'%(str(cx),str(cy))
        self.colX = cx.normalized()
        self.colY = cy.normalized()
        self.colZ = cx.cross(cy).normalized()
        ## A last normalization check!
        #if self.colX.dist(self.colY.cross(self.colZ)) > _eps:
        self.colX=self.vec_y.cross(self.vec_z)
    def fromXY(self, cx, cy):
        _deprecation_warning('fromXY -> from_xy')
        
    def from_xz(self, x_vec, z_vec):
        """ Reset this orientation to the one that conforms with the
        given x and z directions."""
        if cx * cz > _eps:
            print('warning ... orthogonalizing!')
        self.colX = cx.normalized()
        self.colZ = cz.normalized()
        self.colY = cz.cross(cx).normalized()
        ## A last normalization check!
        #if self.colX.dist(self.colY.cross(self.colZ)) > _eps:
        self.colX = self.vec_y.cross(self.vec_z)
    def fromXZ(self, cx, cz):
        _deprecation_warning('fromXZ -> from_xz')

    @property
    def rotation_vector(self):
        """ Return a rotation vector representing this
        orientation. This is essentially the logarithm of the rotation
        matrix."""
        return self.quaternion.rotation_vector

    @property
    def quaternion(self):
        """ Return a quaternion representing this orientation."""
        return m3d.Quaternion(self)

    def toRotationVector(self):
        """ Return a rotation vector representing this
        orientation. This is essentially the logarithm of the rotation
        matrix."""
        _deprecation_warning('toRotationVector() -> [prop] rotation_vector')
        return self.rotation_vector

    def from_rotation_vector(self, rot_vec):
        """ Set this Orientation to represent the one given in a
        rotation vector in 'rot_vec'. 'rot_vec' must be a Vector or an
        numpy array of shape (3,)"""
        if type(rot_vec) == Vector:
            rot_vec = rot_vec.data
        angle = np.linalg.norm(rot_vec)
        if np.abs(angle) < _eps:
            self._data = np.identity(3)
        else:
            axis = rot_vec / angle
            self.from_axis_angle(axis, angle)
    def fromRotationVector(self, rotVec):
        _deprecation_warning('fromRotationVector() -> from_rotation_vector()')

    @property 
    def axis_angle(self):
        """ Return an (axis,angle) pair representing the equivalent
        orientation."""
        return m3d.Quaternion(self).axis_angle
    def toAxisAngle(self):
        """ Return an (axis,angle) pair representing the equivalent
        orientation."""
        _deprecation_warning('toAxisAngle() -> [prop] axis_angle')
        return self.axis_angle

    def from_axis_angle(self, axis, angle):
        """ Set this orientation to the equivalent to rotation of
        'angle' around 'axis'."""
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
        _deprecation_warning('fromAxisAngle() -> from_axis_angle()')
        
    def set_to_x_rotation(self, angle):
        """ Replace this orientation by that of a rotation around x."""
        ca = np.cos(angle)
        sa = np.sin(angle)
        self._data[:,:] = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    def rotX(self, angle):
        _deprecation_warning('rotX() -> set_to_x_rotation()')
        return self.set_to_x_rotation(angle)
    
    def set_to_y_rotation(self, angle):
        """ Replace this orientation by that of a rotation around y."""
        ca=np.cos(angle)
        sa=np.sin(angle)
        self._data[:,:] = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
    def rotY(self, angle):
        _deprecation_warning('rotY() -> set_to_y_rotation()')
        return self.set_to_y_rotation(angle)

    def set_to_z_rotation(self, angle):
        """ Replace this orientation by that of a rotation around z. """
        ca = np.cos(angle)
        sa = np.sin(angle)
        self._data[:,:] = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    def rotZ(self, angle):
        _deprecation_warning('rotZ() -> set_to_z_rotation()')
        return self.set_to_z_rotation(angle)

    def rotateT(self, axis, angle):
        """ In-place rotation of this orientation angle radians in
        axis perceived in the transformed reference system."""
        o = Orientation()
        o.from_axis_angle(axis, angle)
        self.copy(self * o)
    rotate = rotateT
    
    def rotateB(self, axis, angle):
        """ In-place rotation of this orientation angle radians in
        axis perceived in the base reference system.
        Arguments:
        axis -- the axis to rotate about (unit vector with direction).
        angle -- the angle in radians to rotate about the axis."""
        o = Orientation()
        o.from_axis_angle(axis, angle)
        self.copy(o * self)

    def rotateXB(self, angle):
        """ In-place rotation of this oriantation by a rotation around
        x axis in the base reference system. (Inefficient!)"""
        self.rotateB(Vector.e0, angle)
    
    def rotateYB(self, angle):
        """ In-place rotation of this oriantation by a rotation around
        y axis in the base reference system. (Inefficient!)"""
        self.rotateB(Vector.e1, angle)
    
    def rotateZB(self, angle):
        """ In-place rotation of this oriantation by a rotation around
        z axis in the base reference system. (Inefficient!)"""
        self.rotateB(Vector.e2, angle)
    
    def rotateXT(self, angle):
        """ In-place rotation of this oriantation by a rotation around
        x axis in the transformed reference system. (Inefficient!)"""
        self.rotateT(Vector.e0, angle)
    rotate_x = rotateX = rotateXT
    
    def rotateYT(self,angle):
        """ In-place rotation of this oriantation by a rotation around
        y axis in the transformed reference system. (Inefficient!)"""
        self.rotateT(Vector.e1,angle)
    rotate_y = rotateY = rotateYT
    
    def rotateZT(self, angle):
        """ In-place rotation of this oriantation by a rotation around
        z axis in the transformed reference system. (Inefficient!)"""
        self.rotateT(Vector.e2, angle)
    rotate_z = rotateZ = rotateZT
    
    def __repr__(self):
        return '<Orientation: \n' + repr(self._data) + '>'

    def __str__(self):
        return self.__repr__()

    def ang_dist_sq(self, other):
        """ Return the square of the orientation distance (the angle
        of rotation) to the 'other' orientation."""
        return (self.inverse()*other).rotation_vector.length_sq
    def angDist2(self, other):
        _deprecation_warning('angDist2 -> ang_dist_sq')
        return self.ang_dist_sq(other)

    def ang_dist(self, other):
        """ Return the orientation distance (the angle of rotation) to
        the 'other' orientation."""
        return np.sqrt(self.ang_dist_sq(other))
    def angDist(self, other):
        _deprecation_warning('angDist -> ang_dist')
        return self.ang_dist(other)

    def invert(self):
        """ In-place inversion of this orientation."""
        self._data[:,:] = self._data.transpose().copy()

    def inverse(self):
        """ Return an inverse of this orientation as a rotation."""
        o = Orientation(self._data)
        o.invert()
        return o

    def __mul__(self, other):
        if type(other) == Orientation:
            return Orientation(np.dot(self._data, other._data))
        elif type(other) == Vector:
            return Vector(np.dot(self._data, other._data))
        elif isSequence(other):
            return [self * o for o in other]
        else:
            raise self.Error('Multiplication by something other than '
                             + 'Orientation, Vector, or a sequence of these, is not allowed!')
        
def newOrientFromXY(cx, cy):
    """ Create an orientation conforming with the given 'x' and 'y'
    directions."""
    o = Orientation()
    o.from_xy(cx, cy)
    return o

def newOrientFromXZ(cx, cz):
    """ Create an orientation conforming with the given 'x' and 'z'
    directions."""    
    o = Orientation()
    o.from_xz(cx, cz)
    return o

def newOrientRotZ(angle):
    """ Create an orientation corresponding to a rotation for 'angle'
    around the z direction."""
    o = Orientation()
    o.set_to_z_rotation(angle)
    return o

def newOrientRotX(angle):
    """ Create an orientation corresponding to a rotation for 'angle'
    around the x direction."""
    o = Orientation()
    o.set_to_x_rotation(angle)
    return o

def newOrientRotY(angle):
    """ Create an orientation corresponding to a rotation for 'angle'
    around the y direction."""
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
