"""
Module for class Frame in the ReferenceSystem class.
"""

__author__ = "Morten Lind"
__copyright__ = "Morten Lind 2012"
__credits__ = ["Morten Lind"]
__license__ = "GPLv3"
__maintainer__ = "Morten Lind"
__email__ = "morten@lind.dyndns.dk"
__status__ = "Development"

import numpy as np

from ..transform import Transform

class Frame(object):
    """A frame in the reference system is identified by a label name,
    a root frame, and the transform which represents the frame in its
    root."""
    
    def __init__(self, name, root_frame=None, xform=Transform()):
        """ Initialize a frame by a 'name', a 'root_frame' (defaults
        to None) and a transform (defaults to the identity)."""
        self.name = name
        self.root = root_frame
        ## The transform from this to root coordinates, i.e. 'this in root'
        self._xform = xform
        
    @property
    def xform(self):
        """Give access to the fundamental transform which represents
        this frame in its root frame."""
        return self._xform
    
    def __repr__(self):
        return 'Frame:%s' % self.name
