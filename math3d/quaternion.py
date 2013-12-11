"""
Module implementing the Quaternion class.
"""

__author__ = "Morten Lind"
__copyright__ = "Morten Lind 2012"
__credits__ = ["Morten Lind"]
__license__ = "GPLv3"
__maintainer__ = "Morten Lind"
__email__ = "morten@lind.dyndns.dk"
__status__ = "Production"

import numpy as np

# # Circular dependencies prevents direct import of Orientation, hence
# # global addressing
import math3d as m3d

from . import utils
from .vector import Vector

def isQuaternion(q):
    utils._deprecation_warning('type(q) == math3d.Quaternion')
    return type(q) == Quaternion

class Quaternion(object):
    """Quaternion class."""
    
    class Error(Exception):
        """Exception class."""
        def __init__(self, message):
            self.message = message
            Exception.__init__(self, self.message)
        def __repr__(self):
            return self.message

    def __init__(self, *args):
        """Create a quaternion. Args may be () for default
        constructor; (Orientation) for createing a quaternion
        representing the given orientation; (Quaternion) for a copy
        constructor, (s,x,y,z) or (s,Vector) for the direct quaternion
        data; (Vector) for creating the equivalent to a rotation
        vector; or (Vector, angle) for creating the equivalent of axis
        angle."""
        if len(args) == 0:
            ## Default constructor
            self._s = 1.0
            self._v = Vector()
        elif len(args) == 1:
            ## Try with orientation or quaternion
            if type(args[0]) == m3d.Orientation:
                self._v = Vector()
                self.orientation = args[0]
            ## Try with rotation vector
            if type(args[0]) == Vector:
                self.from_rotation_vector(args[0])
            ## Copy constructor
            elif type(args[0]) == Quaternion:
                self._s = args[0]._s
                self._v = args[0]._v.copy()
        elif len(args) == 2:
            ## Test for (axis, angle) and (s, v) determined by order
            if utils.is_num_type(args[0]) and type(args[1]) == Vector:
                ## Interpret as s, v
                self._s = args[0]
                self._v = args[1].copy()
            elif utils.is_num_type(args[1]) and type(args[0]) == Vector:
                ## Interpret as axis-angle
                axis = args[0].copy()
                ang = args[1]
                self.fromAxisAngle(axis, ang)
        elif len(args) == 3 and np.all(np.isreal(args)):
            ## Assume three components of a rotation vector
            self.from_rotation_vector(Vector(args))
        elif len(args) == 4 and np.all(np.isreal(args)):
            ## Assume numbers for s, x, y, and z
            self._s = args[0]
            self._v = Vector(args[1:])
        if np.abs(self.norm - 1.0) > utils._eps:
            print('Quaternion.__init__ : Warning : Arguments did not constitute a unit quaternion. Normalizing.')
            self.normalize()

    def __getattr__(self, name):
        if name == 's':
            return self._s
        elif name == 'x':
            return self._v.x
        elif name == 'y':
            return self._v.y
        elif name == 'z':
            return self._v.z
        else:
            raise AttributeError('Attribute "{}" not found in Quaternion '
                                 + 'class'.format(name))

    @property
    def vector_part(self):
        """Return a copy of the vector part of the quaternion."""
        return self._v.copy()

    @property
    def scalar_part(self):
        """Return the scalar part of the quaternion."""
        return self._s

    def __setattr__(self, name, val):
        if name in ['s', 'x', 'y', 'z']:
            raise AttributeError('Not allowed to set attribute "{}" in '
                                 + 'Quaternion'.format(name))
        else:
            object.__setattr__(self, name, val)
        
    def __getitem__(self, index):
        if index == 0:
            return self._s
        else:
            return self._v[index-1]

    def __repr__(self):
        return '<Quaternion: [{:.5f}, ({:.5f}, {:.5f}, {:.5f})]>'.format(
            self._s, *self._v._data)

    def __copy__(self):
        """Copy method for creating a copy of this Quaternion."""
        return Quaternion(self)
    
    def __deepcopy__(self, memo):
        return self.__copy__()
    
    def copy(self, other=None):
        """Copy data from 'other' to self. If no argument given,
        i.e. 'other==None', return a copy of this Quaternion."""
        if other is None:
            return Quaternion(self)
        else:
            self._s = other._s
            self._v = other._v.copy()
            
    def __mul__(self, other):
        """Multiplication is interpreted by either transforming
        (rotating) a Vector, ordinary Quaternion multiplication, or
        multiplication by scalar."""
        if type(other) == Vector:
            ## Do a rotation of the vector
            return (self * Quaternion(0, other) * self.inverse())._v
        elif type(other) == Quaternion:
            ## Ordinary quaternion multiplication
            return Quaternion(self._s * other._s - self._v * other._v,
                               self._v.cross(other._v) +
                              self._s * other._v + other._s * self._v)
        elif utils.is_num_type(other):
            return Quaternion(other * self._s, other * self._v)
        else:
            return NotImplemented

    def __rmul__(self, rother):
        """Right-multiply by number. """
        if utils.is_num_type(rother):
            return Quaternion(rother * self._s, rother * self._v)

    def __imul__(self, other):
        """In-place multiply."""
        if utils.is_num_type(other):
            self._s *= other
            self._v *= other
        else:
            return NotImplemented
        return self
            
    def __ipow__(self, x):
        """In-place exponentiation of this quaternion to the power of
        'x'."""
        if abs(1 - abs(self._s)) < 1e-7:
            self._s = 1
            self._v = Vector(0, 0, 0)
        else:
            theta = np.arccos(self._s)
            sintheta = np.sin(theta)
            logv = theta / sintheta * self._v
            alpha = x * logv.length()
            v = logv.normalized()
            self._s = np.cos(alpha)
            self._v = np.sin(alpha) * v
        return self

    def __pow__(self, x):
        """Return this quaternion to the power of 'x'."""
        q = Quaternion(self)
        q **= x
        return q

    def __neg__(self):
        """Return the negative quaternion to self."""
        q = Quaternion(self)
        q *= -1.0
        return q

    @property
    def ang_norm(self):
        """Return the angular norm, i.e. the angular rotation, of
        this quaternion."""
        return 2*np.arccos(self._s)
    def angNorm(self):
        utils._deprecation_warning('angNorm() -> [prop] ang_norm')
        return self.ang_norm

    def ang_dist(self, other):
        """Compute the rotation angle distance to the 'other'
        quaternion."""
        return (self.conjugated()*other).ang_norm()
    def angDist(self, other):
        utils._deprecation_warning('angDist() -> ang_dist()')
        return self.ang_dist(other)

    def dist2(self, other):
        """Compute the square of the usual quaternion metric distance to the
        'other' quaternion."""
        return (self._s - other._s)**2 + (self._v - other._v).length_sq

    def dist(self, other):
        """Compute the usual quaternion metric distance to the
        'other' quaternion."""
        return np.sqrt(self.dist2(other))
    
    @property
    def axis_angle(self):
        """Return an '(axis, angle)' pair representing the orientation
        of this quaternion."""
        alpha = 2 * np.arccos(self._s)
        if alpha != 0:
            n = self._v / np.sin(alpha / 2)
        else:
            n = Vector()
        return (n, alpha)
    def toAxisAngle(self):
        """Return an '(axis, angle)' pair representing the orientation
        of this quaternion."""
        utils._deprecation_warning('toAxisAngle() -> [prop] axis_angle')
        return self.axis_angle

    @axis_angle.setter
    def axis_angle(self, axisangle):
        """Set this quaternion to the equivalent of the given axis
        and angle given in the ordered pair 'axisangle'."""
        axis, angle = axisangle
        if type(axis) != Vector:
            axis = Vector(axis)
        sa = np.sin(0.5 * angle)
        ca = np.cos(0.5 * angle)
        axis.normalize()
        self._s = ca
        self._v._data[:] = (sa * axis)._data
    def fromAxisAngle(self, axis, angle):
        _deprecation_warning('q.fromAxisAngle(axis, angle) '
                             + '-> [prop] q.axis_angle = (axis, angle)')
        self.axis_angle = (axis, angle)

    @property
    def rotation_vector(self):
        """Return a rotation vector representing the rotation of this
        quaternion."""
        n, alpha = self.axis_angle
        if alpha != 0.0:
            return alpha * n
        else:
            return n
    def toRotationVector(self):
        """Return a rotation vector representing the rotation of this
        quaternion."""
        _deprecation_warning('toRotationVector -> rotation_vector')
        return self.rotation_vector
    
    @property
    def rotation_vector(self):
        """Return a rotation vector representing the rotation of this
        quaternion."""
        n, alpha = self.axis_angle
        if alpha != 0.0:
            return alpha * n
        else:
            return n
    def toRotationVector(self):
        """Return a rotation vector representing the rotation of this
        quaternion."""
        utils._deprecation_warning('toRotationVector -> rotation_vector')
        return self.rotation_vector

    @rotation_vector.setter
    def rotation_vector(self, rot_vec):
        """Set this quaternion to the equivalent of the given
        rotation vector 'w'."""
        if type(rot_vec) != Vector:
            rot_vec = Vector(rot_vec)
        angle = rot_vec.length()
        if angle > utils._eps:
            axis = rot_vec.normalized()
        else:
            ## Select arbitrary x-direction as axis and set angle to zero
            axis = Vector.e1
            angle = 0.0
        self.fromAxisAngle(axis, angle)
    def from_rotation_vector(self, rot_vec):
        _deprecation_warning('q.from_rotation_vector(rv) -> q.rotation_vector = rv')
        self.rotation_vector = rot_vec
    def fromRotationVector(self, rot_vec):
        utils._deprecation_warning('q.fromRotationVector(rv) -> q.rotation_vector = rv')
        self.rotation_vector = rot_vec

        
    @property
    def orientation(self):
        """Return an orientation object representing the same
        rotation as this quaternion."""
        ## Return an Orientation representing this quaternion
        self.normalize()
        s = self._s
        v = self._v
        x = v.x
        y = v.y
        z = v.z
        x2 = x**2
        y2 = y**2
        z2 = z**2
        return m3d.Orientation(np.array([
            [1 - 2 * (y2 + z2), 2 * x * y - 2 * s * z, 2 * s * y + 2 * x * z],
            [2 * x * y + 2 * s * z, 1 - 2 * (x2 + z2), -2 * s * x + 2 * y * z],
            [-2 * s * y + 2 * x * z, 2 * s * x + 2 * y * z, 1 - 2 * (x2 + y2)]
            ]))
    def toOrientation(self):
        utils._deprecation_warning('toOrientation -> [prop] orientation')
        return self.orientation

    @orientation.setter
    def orientation(self, orient):
        """Set the assigned orientation in 'orient' to this Quaternion."""
        self.from_orientation(orient)

    def from_orientation(self, orient, positive=True):
        """Set this quaternion to represent the given
        orientation. The used method should be robust;
        cf. http://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation.
        The mentioned method from wikipedia has problems with certain
        orientations, like the identity. Therfore another robust
        method from
        http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
        is used; the one from 'Angel'
        """
        M = orient._data
        tr = M.trace() + 1.0
        if tr > 1e-10:
            s = 0.5 / np.sqrt(tr)
            self._s = 0.25 / s
            self._v.x = s * (M[2, 1] - M[1, 2])
            self._v.y = s * (M[0, 2] - M[2, 0])
            self._v.z = s * (M[1, 0] - M[0, 1])
        else:
            diag = M.diagonal()
            u = diag.argmax()
            v = (u + 1) % 3
            w = (v + 1) % 3
            r = np.sqrt(1 + M[u, u] - M[v, v] - M[w, w])
            if abs(r) < 1e-10:
                self._s = 1.0
                self._v = Vector(0, 0, 0)
            else:
                tworinv = 1.0 / (2 * r)
                self._s = (M[w, v] - M[v, w]) * tworinv
                #if not positive and self._s > 0:
                #    self._s *= -1
                self._v[u] = 0.5 * r
                self._v[v] = (M[u, v] + M[v, u]) * tworinv
                self._v[w] = (M[w, u] + M[u, w]) * tworinv
        if positive and self._s < 0:
            self *= -1.0
        self.normalize()
    def fromOrientation(self, orient, positive=True):
        utils._deprecation_warning('q.fromOrientation(orient, positive) '
                                   + '-> q.from_orientation(orient, positive)'
                                   + 'or q.orientation = orient (if positive==True)')
        self.from_orientation(orient, positive)

    @property
    def norm(self):
        """Return the norm of this quaternion."""
        return np.sqrt(self.norm_sq)

    @property
    def norm_sq(self):
        """Return the square of the norm of this quaternion."""
        return self._s**2 + self._v.length_sq
    def norm2(self):
        utils._deprecation_warning('norm2() -> [prop] norm_sq')
        return self.norm_sq
    
    def conjugate(self):
        """In-place conjugation of this quaternion."""
        self._v = -self._v

    def conjugated(self):
        """Return a quaternion which is the conjugated of this quaternion."""
        qc = self.copy()
        qc.conjugate()
        return qc
        
    def normalize(self):
        """Normalize this quaternion. """
        n = self.norm
        if abs(n) < 1e-10:
            self._s = 1
            self._v = Vector(0.0, 0.0, 0.0)
        else:
            ninv = 1.0 / n
            self._s *= ninv
            self._v *= ninv
         
    def normalized(self):
        """Return a normalised version of this quaternion. """
        q = Quaternion(self)
        q.normalize()
        return q

    def invert(self):
        """In-place inversion of this quaternion. """
        n2 = self.norm_sq
        self.conjugate()
        self *= 1 / n2
        
    def inverse(self):
        """Return an inverse of this quaternion."""
        qi = self.copy()
        qi.invert()
        return qi  

    @property
    def array(self):
        """Return an ndarray with the fundamental data
        of the Quaternion.  The layout is as described by the
        Quaternion.list property"""
        return np.array(self.list)

    @property
    def list(self):
        """Return the fundamental data of the Quaternion as a
        list. The scalar part is placed in the first element, at index
        0, and the vector data at the remainder, slice [1:]."""
        return [self._s]+self._v.list
        
    
