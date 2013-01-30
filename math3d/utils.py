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
    f = inspect.stack()[2]
    # print(f)
    print('math3d: {} @ {} in {}:\n\tA deprecated method was invoked. '.format(f[1], f[2], f[3])
          + 'Suggestion for replacement: "%s"' % msg)
    
## Limit for accuracy of consistencies and comparison.
_eps = np.finfo(np.float32).resolution

## Tuple of types considered sequences 
_seq_types = (list, tuple, np.ndarray)
_seqTypes = _seq_types

def isSequence(s):
    _deprecation_warning('is_sequence')
    return is_sequence(s)
def is_sequence(s):
    return type(s) in _seq_types

def isThreeSequence(s):
    _deprecation_warning('is_three_sequence')
    return is_three_sequence(s)
def is_three_sequence(s):
    return type(s) in _seq_types and len(s) == 3

## Standard numeric types
_num_types = [float, int]
_numTypes = _num_types
## Get numeric types from numpy
for i in np.typeDict:
    if type(i) == type('') and (i.find('int') >= 0 or i.find('float') >= 0):
        _num_types.append(np.typeDict[i])
        
def isNumType(val):
    _deprecation_warning('is_num_type')
    return is_num_type(val)
def is_num_type(val):
    return type(val) in _num_types

def isNumTypes(lst):
    _deprecation_warning('is_num_types')
    return is_num_types(lst)
def is_num_types(lst):
    return np.all([lambda li: isinstance(li, numbers.Number) for li in lst])
