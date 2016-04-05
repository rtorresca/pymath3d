# coding=utf-8

"""
Utility function and definitions library for PyMath3D.
"""

__author__ = "Morten Lind"
__copyright__ = "Morten Lind 2009-2015"
__credits__ = ["Morten Lind"]
__license__ = "GPLv3"
__maintainer__ = "Morten Lind"
__email__ = "morten@lind.dyndns.dk"
__status__ = "Production"


import numbers
import inspect
import collections

import numpy as np


def _deprecation_warning(msg):
    f = inspect.stack()[2]
    # print(f)
    print(('math3d: {} @ {} in {}:\n\tA deprecated method was invoked. ')
          .format(f[1], f[2], f[3]) +
          'Suggestion for replacement: "{:s}"'.format(msg))

# Limit for accuracy of consistencies and comparison.
_eps32 = np.finfo(np.float32).resolution
_eps64 = np.finfo(np.float64).resolution
_eps = _eps32

# # Tuple of types considered sequences
# _seq_types = (list, tuple, np.ndarray)


def is_sequence(obj):
    """Test if "obj" is a sequence."""
    return isinstance(obj, collections.Iterable)


def is_three_sequence(obj):
    """Test if "obj" is of a sequence type and three long."""
    return isinstance(obj, collections.Iterable) and len(obj) == 3


# Standard numeric types
_number_bases = (np.number, numbers.Number)


def is_num_type(val):
    """Test if "val" is of a number type."""
    return isinstance(val, _number_bases)


def is_num_types(lst):
    """Test if every item in "lst" is of a number type."""
    return np.all([(lambda x: isinstance(x, _number_bases))(li) for li in lst])


class Error(Exception):
    """Exception class."""
    def __init__(self, message):
        self.message = message
        Exception.__init__(self, self.message)

    def __repr__(self):
        return self.message
