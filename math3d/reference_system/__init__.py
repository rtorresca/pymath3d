"""
Module implementing the Reference System class.
"""

__author__ = "Morten Lind"
__copyright__ = "Morten Lind 2012"
__credits__ = ["Morten Lind"]
__license__ = "GPLv3"
__maintainer__ = "Morten Lind"
__email__ = "morten@lind.dyndns.dk"
__status__ = "Development"

import numpy as np
from .frame import Frame
from ..transform import Transform

class ReferenceSystem(object):
    """A reference system is an object which holds a tree of reference
    frames with their associated relating transforms. A fundamental
    facility is to be able to give the transform between any two
    registered reference frames."""
    
    class Error(Exception):
        """Exception class for the ReferenceSystem class."""
        def __init__(self, msg):
            self.msg = 'ReferenceSystem Error: ' + msg
        def __str__(self):
            return self.msg
        
    def __init__(self):
        """Initialize an empty reference system, i.e. one that holds
        only the 'World' frame."""
        self._frames = {'world': Frame('world')}

    def _get_root_path(self, frame):
        """Assemble the root path as a list of frames leading from the
        given 'frame' to the root of the reference system. 'frame' may
        be given directly as a Frame object or as a string identifying
        a frame in the reference system."""
        target_frame = frame if type(frame) == Frame else self._frames[frame]
        path = [target_frame]
        f=target_frame.root
        while not f is None:
            path.append(f)
            f=f.root
        return path
    
    def _chain_transform(self, transform_chain):
        """Reduce the sequence of transforms given in 'transform_chain' to one
        transform by multiplication."""
        return reduce(lambda t,f:t*f.xform, transform_chain, Transform())
        
    def _get_common_root_paths(self, frame_1, frame_2):
        """Assemble the paths to the deepest common root of the given
        frames 'frame_1' and 'frame_2'. The frames may be given
        directly as Frame objects or as strings identifying frames in
        the reference system."""
        frame_1 = frame_1 if type(frame_1) == str else self._frames[frame_1]
        frame_2 = frame_2 if type(frame_2) == str else self._frames[frame_2]
        rp1 = self._get_root_path(frame_1)
        rp2 = self._get_root_path(frame_2)
        # // find a common root frame
        for crf in rp1:
            if crf in rp2:
                break
        if not crf in rp1 and cf in rp2:
            print('Error, no common root found!')
        crp1 = rp1[:rp1.index(crf)]
        crp2 = rp2[:rp2.index(crf)]
        return crp1,crp2
    
    def transform(self, frame_target, frame_base):
        """ Return the transform from 'frame_target' coordinates to
        'frame_base' coordinates. The frames may be given directly as
        Frame objects or as strings identifying frames in the
        reference system."""
        crpm, crpb = self._get_common_root_paths(frame_target, frame_base)
        crpm.reverse()
        crpb.reverse()
        return self._chain_transform(crpb).inverse() * self._chain_transform(crpm)

    __call__=transform

    def add_frame(self, frame_name, root, xform=None, frame=None):
        """Add a frame identified by 'frame_name' to the reference
        system, rooted at 'root', and using either 'xform' as
        transform or a given 'frame' object which should provide the
        same interface as a Frame object. 'root' may be given as a
        frame or as a string identifying a frame."""
        if type(root) == Frame:
            root = root
            root_name = root.name
        else:
            root_name = root
            root = self._frames[root]
        if not root_name in self._frames:
            raise self.Error('Root frame "%s" ' % root_name
                             + 'not found in reference system')
        if frame_name in self._frames:
            raise self.Error('Frame name "%s" ' % frame_name
                             + 'already exists in  the reference system')
        if xform is None and frame is None:
            raise self.Error('Need specification of new frame by argument '
                             + '"xform" or "frame".')
        elif not xform is None and not frame is  None:
            raise self.Error('Need one single specification in arguments '
                             + '"xform" or "frame". One of them must be None!')
        if not xform is None:
            self._frames[frame_name] = Frame(frame_name, root, xform)
        elif not frame is None:
            self._frames[frame_name] = frame
            if type(frame.root) == str:
                frame.root = self._frames[frame.root]

    def __getitem__(self, name):
        """Mapping based access to the Frame objects in the reference
        system."""
        return self._frames[name]
    
