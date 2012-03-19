"""
Module implementing the Quaternion class.
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
from .utils import isNumType, _eps
from .vector import Vector

def isQuaternion(q):
    return type(q) == Quaternion

class Quaternion(object):
    """Quaternion class."""
    
    class Error(Exception):
        """Exception class."""
        def __init__(self, message):
            self.message = 'Quaternion Error : ' + message
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
                self.fromOrientation(args[0])
            ## Try with rotation vector
            if type(args[0]) == Vector:
                self.fromRotationVector(args[0])
## Copy constructor
            elif type(args[0]) == Quaternion:
                self._s = args[0]._s
                self._v = args[0]._v.copy()
        elif len(args) == 2:
            ## Test for (axis, angle) and (s, v) determined by order
            if isNumType(args[0]) and type(args[1]) == Vector:
                ## Interpret as s, v
                self._s = args[0]
                self._v = args[1].copy()
            elif isNumType(args[1]) and type(args[0]) == Vector:
                ## Interpret as axis-angle
                axis = args[0].copy()
                ang = args[1]
                self.fromAxisAngle(axis, ang)
        elif len(args) == 3 and np.all(np.isreal):
            ## Assume three components of a rotation vector
            self.fromRotationVector(Vector(args))
        elif len(args) == 4 and np.all(np.isreal):
            ## Assume numbers for s, x, y, and z
            self._s = args[0]
            self._v = Vector(args[1:])

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
            raise AttributeError('Attribute "%s" not found in Quaternion class'%name)

    def __setattr__(self, name, val):
        if name in ['s', 'x', 'y', 'z']:
            raise AttributeError('Not allowed to set attribute "%s" in Quaternion'%name)
        else:
            object.__setattr__(self, name, val)
        
    def __getitem__(self, index):
        if index == 0:
            return self._s
        else:
            return self._v[index-1]

    def __repr__(self):
        return '[ %.5f , ( %.5f , %.5f , %.5f ) ]' % (self._s, self._v.x, self._v.y, self._v.z)

    def __copy__(self):
        """Copy method for creating a copy of this Quaternion."""
        return Quaternion(self)
    
    def __deepcopy__(self, memo):
        return self.__copy__()
    
    def copy(self, other=None):
        """Set this quaternion to a copy of other, if not
        None. Otherwise, return a quaternion which is a copy of this
        quaternion."""
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
        elif isNumType(other):
            return Quaternion(other * self._s, other * self._v)

    def __rmul__(self, rother):
        """Right-multiply by number. """
        if isNumType(rother):
            return Quaternion(rother * self._s, rother * self._v)

    def __imul__(self, other):
        """In-place multiply."""
        if isNumType(other):
            self._s *= other
            self._v *= other
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

    def angNorm(self):
        """Return the angular norm, i.e. the angular rotation, of
        this quaternion."""
        return 2*np.arccos(self._s)

    def angDist(self, other):
        """Compute the rotation angle distance to the 'other'
        quaternion."""
        return (self.conjugated()*other).angNorm()
    
    def dist2(self, other):
        """Compute the square of the usual quaternion metric distance to the
        'other' quaternion."""
        return (self._s - other._s)**2 + (self._v - other._v).length2()

    def dist(self, other):
        """Compute the usual quaternion metric distance to the
        'other' quaternion."""
        return np.sqrt(self.dist2(other))

    def fromAxisAngle(self, axis, angle):
        """Set this quaternion to the equivalent of the given 'axis'
        and 'angle'."""
        sa = np.sin(0.5 * angle)
        ca = np.cos(0.5 * angle)
        axis.normalize()
        self._s = ca
        self._v = sa * axis
        
    def toAxisAngle(self):
        """Return an '(axis, angle)' pair representing the orientation
        of this quaternion."""
        alpha = 2 * np.arccos(self._s)
        if alpha != 0:
            n = self._v / np.sin(alpha / 2)
        else:
            n = Vector()
        return (n, alpha)

    def fromRotationVector(self, w):
        """Set this quaternion to the equivalent of the given
        rotation vector 'w'."""
        angle = w.length()
        if angle > _eps:
            axis = w.normalized()
        else:
            ## Select arbitrary x-direction as axis and set angle to zero
            axis = Vector.e1
            angle = 0.0
        self.fromAxisAngle(axis, angle)
    
    def toRotationVector(self):
        """Return a rotation vector representing the rotation of this
        quaternion."""
        n, alpha = self.toAxisAngle()
        if alpha != 0.0:
            return alpha * n
        else:
            return n
        
    def fromOrientation(self, orient, positive=True):
        """Set this quaternion to represent the given
        orientation. The used method should be robust;
        cf. http://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation.
        The mentioned method from wikipedia has problems with certain
        orientations, like the identity. Therfore another robust
        method from
        http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
        is used; the one from 'Angel'
        """
        M = orient.data
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
                        
    def toOrientation(self):
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
    
    def norm(self):
        """Return the norm of this quaternion."""
        return np.sqrt(self.norm2())
    
    def norm2(self):
        """Return the square of the norm of this quaternion."""
        return self._s**2 + self._v.length2()

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
        n = self.norm()
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
        n2 = self.norm2()
        self.conjugate()
        self *= 1 / n2
        
    def inverse(self):
        """Return an inverse of this quaternion."""
        qi = self.copy()
        qi.invert()
        return qi  

if __name__ == '__main__':
    import readline
    import rlcompleter
    readline.parse_and_bind("tab: complete")

