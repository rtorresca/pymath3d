'''
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
'''
'''
Module for top level imports in PyMath3D.
'''

from math3d.quaternion import isQuaternion, Quaternion
from math3d.orientation import isOrientation, Orientation, newOrientFromXY, newOrientFromXZ, newOrientRotZ, newOrientRotY, newOrientRotX
from math3d.vector import isVector, Vector
from math3d.transform import isTransform, Transform, newTransFromXYP, newTransFromXZP
