"""
Utility function and definitions library for PyMath3D.
"""

__author__ = "Morten Lind"
__copyright__ = "Morten Lind 2009-2012"
__credits__ = ["Morten Lind"]
__license__ = "GPL"
__maintainer__ = "Morten Lind"
__email__ = "morten@lind.no-ip.org"
__status__ = "Production"

import numbers
import inspect

import numpy as np

def _deprecation_warning(msg):
    f = inspect.stack()[1]
    print(f)
    print('math3d: At %s : %d:\n\tA deprecated method was invoked. ' % (f[1], f[2])
          + 'Suggestion for replacement: "%s"' % msg)
    
## Limit for accuracy of consistencies and comparison.
_eps = np.finfo(np.float32).resolution

## Tuple of types considered sequences 
_seqTypes = (list, tuple, np.ndarray)

def isSequence(s):
    return type(s) in _seqTypes

def isThreeSequence(s):
    return type(s) in _seqTypes and len(s) == 3

## Standard numeric types
_numTypes = [float, int]
## Get numeric types from numpy
for i in np.typeDict:
    if type(i) == type('') and (i.find('int') >= 0 or i.find('float') >= 0):
        _numTypes.append(np.typeDict[i])
        
def isNumType(val):
    return type(val) in _numTypes

def isNumTypes(lst):
    return np.all([lambda li: isinstance(li, numbers.Number) for li in lst])
