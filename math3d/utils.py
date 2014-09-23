"""
Utility function and definitions library for PyMath3D.
"""

__author__ = "Morten Lind"
__copyright__ = "Morten Lind 2009-2012"
__credits__ = ["Morten Lind"]
__license__ = "GPLv3"
__maintainer__ = "Morten Lind"
__email__ = "morten@lind.no-ip.org"
__status__ = "Production"

import numbers
import inspect

import numpy as np

def _deprecation_warning(msg):
    f = inspect.stack()[2]
    # print(f)
    print(('math3d: {} @ {} in {}:\n\tA deprecated method was invoked. ')
          .format(f[1], f[2], f[3])
          + 'Suggestion for replacement: "%s"' % msg)
    
## Limit for accuracy of consistencies and comparison.
_eps = np.finfo(np.float32).resolution

## Tuple of types considered sequences 
_seq_types = (list, tuple, np.ndarray)

def is_sequence(s):
    return type(s) in _seq_types

def is_three_sequence(s):
    return type(s) in _seq_types and len(s) == 3

## Standard numeric types
_number_bases = (np.number,numbers.Number)
        
def is_num_type(val):
    # return np.isreal(val)
    return isinstance(val, _number_bases)

def is_num_types(lst):
    return np.all([lambda li: isinstance(li, _number_bases) for li in lst])
    
class Error(Exception):
    """Exception class."""
    def __init__(self, message):
        self.message = message
        Exception.__init__(self, self.message)
    def __repr__(self):
        return self.message

