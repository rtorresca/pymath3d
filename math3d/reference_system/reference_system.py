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
from .point import Point
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
        self._points = {}
        
    def _get_root_path(self, frame):
        """Assemble the root path as a list of frames leading from the
        given 'frame' to the root of the reference system. 'frame' may
        be given directly as a Frame object or as a string identifying
        a frame in the reference system."""
        target_frame = frame if type(frame) == Frame else self._frames[frame]
        path = [target_frame]
        f=target_frame._root_frame
        while not f is None:
            path.append(f)
            f=f._root_frame
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

    def get_point(self, point_name):
        """Retrieve the point object registered under the given
        'point_name'."""
        return self._points[point_name]
        
    def point_in_ref(self, point, target_frame):
        """Return a position vector for 'point' with reference in the
        given 'target_frame'. 'point' may be given as a Point instance
        with internal reference to a registered frame, or as a string
        naming a registered point."""
        ## Retrieve the point if given as a named point
        if type(point) == str:
            point = self.get_point(point)
        ## Get the transform from the point's root frame to target frame coordinates.
        trf = self.transform(target_frame, point._root_frame)
        ## Return the transformed position vector.
        return trf * point.pos_vec
        
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

    def add_new_point(self, point_name, root, pos_vec):
        """Add a new, specified point to the reference system."""
        if point_name in self._points:
            raise self.Error(
                'Trying to register a new point with name "%s",'
                ' which is already registered!' % point_name)
        if type(root) == str:
            root = self.get_frame(root)
        self._points[point_name] = Point(point_name, root, pos_vec) 


    def add_new_frame(frame_name, root, xform_getter=None):
        """Add a new, specified frame to the reference system."""
        if frame_name in self._frames:
            raise self.Error(
                'Trying to register a new frame with name "%s",'
                ' which is already registered!' % frame_name)
        if type(root) == str:
            root = self.get_frame(root)
        if not root in self._frames.values():
            raise self.Error(
                'Trying to register a new frame with root "%s",'
                ' which is not in the reference system!' % root.name)
        self._frames[frame_name] = Frame(frame_name, root, xform_getter)

    def add_frame_object(frame_object):
        """Add a specialized frame object, """
        if frame.name in self._frames:
            raise self.Error(
                'Trying to register a frame with name "%s",'
                ' which is already registered!' % frame.name)
        self._frames[frame.name] = frame_object
        
    def add_frame(self, frame_name, root=None, xform=None):
        """Add a frame identified by 'frame_name' to the reference
        system, rooted at 'root', and using either 'xform' as
        transform or a given 'frame' object which should provide the
        same interface as a Frame object. 'root' may be given as a
        frame or as a string identifying a frame."""
        if type(root) == Frame:
            root = root
            root_name = root._name
        elif type(root) == str:
            root_name = root
            root = self._frames[root]
        else:
            ## Check if a frame_object is given and get root from that.
            if frame_object is None:
                raise self.Error(
                    'Either a valid root frame must be given in "root" argument,'
                    ' or a specific "frame_object" with reference must'
                    ' be given.')
            else:
                root = frame_object._root_frame
                root_name = root._name
        if not root_name in self._frames:
            raise self.Error('Root frame "%s" ' % root_name
                             + 'not found in reference system')
        if frame_name in self._frames:
            raise self.Error('Frame name "%s" ' % frame_name
                             + 'already exists in  the reference system')
        if xform is None and frame_object is None:
            raise self.Error('Need specification of new frame by argument '
                             + '"xform" or "frame".')
        elif not xform is None and not frame_object is  None:
            raise self.Error('Need one single specification in arguments '
                             + '"xform" or "frame". One of them must be None!')
        if not xform is None:
            self._frames[frame_name] = Frame(frame_name, root, xform)
        elif not frame is None:
            self._frames[frame_name] = frame
            if type(frame._root_frame) == str:
                frame._root_frame = self._frames[frame._root_frame]

    def __getitem__(self, name):
        """Mapping based access to the Frame objects in the reference
        system."""
        return self._frames[name]
    
