'''
Implementation of a quadtree points structure for use in the fast multipole method.
'''

__author__ = 'Luis Barroso-Luque'

import numpy as np
from itertools import chain

eps = 7./3 - 4./3 -1

def _loopchildren(parent):
    for child in parent._children:
        if child._children:
            for subchild in _loopchildren(child):
                yield subchild
        yield child


class Node():
    """Single Tree Node"""

    # Can we implement this in a less hacky way? Maybe with some permutations?
    # When looking for nearest neighbors if the cardinal neighbors of a cell
    # are larger then that cell and the corresponding neigbor index for the given
    # Child index is in DISREGARD then don't skip looking for corner neighbors
    # DISREGARD = (child index->cneighbor index)
    DISREGARD = (1,2,0,3)
    # Again when a cardinal neighbor is larger than cell then look for the
    # child given by the table for the given cneighbor index
    # CORNER_CHILDREN = {cneighbor index: child index of neigbor}
    CORNER_CHILDREN = (3, 2, 0, 1)


    def __init__(self, width, height, x0, y0, points=None,
                 children=None, parent=None, level=0):

        self._points = []
        self._children = children
        self._cneighbors = 4*[None,]
        self._nneighbors = None
        self._cindex = 0
        self.parent = parent
        self.x0, self.y0, self.w, self.h = x0, y0, width, height
        self.verts = ((x0, x0 + width), (y0, y0 + height))
        self.center = (x0 + width/2, y0 + height/2)
        self.level = level
        self.inner, self.outer = None, None

        if points is not None:
            self.add_points(points)

    def __iter__(self):
        if self._has_children():
            for child in self._children:
                yield child

    def __len__(self):
        if self._points is not None:
            return len(self._points)
        return 0

    def _has_children(self):
        return (self._children is not None)

    def _get_child(self, i):
        if self._children is None:
            return self
        return self._children[i]

    def _split(self):
        if self._has_children():
            return

        w = self.w/2
        h = self.h/2
        x0, y0 = self.verts[0][0], self.verts[1][0]

        # Create children order [NW, NE, SW, SE] -> [0,1,2,3]
        self._children = [Node(w, h, xi, yi, points=self._points,
                               level=self.level+1, parent=self)
                          for yi in (y0 + h, y0) for xi in (x0, x0 + w)]
        # part of that terrible DISREGARD hack
        for i, c in enumerate(self._children):
            c._cindex = i

        #self._points = None
        #self.set_cneighbors()

    def _contains(self, x, y):
        return ((x >= self.verts[0][0] and x < self.verts[0][1]) and
                (y >= self.verts[1][0] and y < self.verts[1][1]))

    def is_leaf(self):
        return (self._children is None)

    def thresh_split(self, thresh):
        if len(self) > thresh:
            self._split()
        if self._has_children():
            for child in self._children:
                child.thresh_split(thresh)
            #self.set_cneighbors()

    def set_cneighbors(self):
        for i, child in enumerate(self._children):
            # Set sibling neighbors
            sn = (abs(1 + (i^1) - i), abs(1 + (i^2) - i))
            child._cneighbors[sn[0]] = self._children[i^1]
            child._cneighbors[sn[1]] = self._children[i^2]
            # Set other neighbors from parents neighbors
            pn = tuple(set((0,1,2,3)) - set((sn)))
            nc = lambda j, k: j^((k+1)%2+1)
            child._cneighbors[pn[0]] = (self._cneighbors[pn[0]]._get_child(nc(i, pn[1]))
                                        if self._cneighbors[pn[0]] is not None
                                        else None)
            child._cneighbors[pn[1]] = (self._cneighbors[pn[1]]._get_child(nc(i, pn[0]))
                                        if self._cneighbors[pn[1]] is not None
                                        else None)
            # Recursively set cneighbors
            if child._has_children():
                child.set_cneighbors()

    def add_points(self, points):
        if self._has_children():
            for child in self._children:
                child.add_points(points)
        else:
            for d in points:
                if self._contains(d.x, d.y):
                    self._points.append(d)

    def get_points(self):
        return self._points
        #if self._has_children():
        #    return chain(*(child.get_points() for child in self._children))
        #else:
        #    return self._points

    def traverse(self):
        if self._has_children():
            for child in _loopchildren(self):
                yield child

    @property
    def nearest_neighbors(self):
        if self._nneighbors is not None:
            return self._nneighbors

        # Find remaining nearest neighbors of same level
        nn = [cn._cneighbors[(i+1)%4]
              for i, cn in enumerate(self._cneighbors)
              if cn is not None and cn.level == self.level]
        # Find remaining nearest neigbor at lower levels #So hacky!
        nn += [cn._cneighbors[(i+1)%4]._get_child(self.CORNER_CHILDREN[i])
               for i, cn in enumerate(self._cneighbors)
               if cn is not None and cn._cneighbors[(i+1)%4] is not None and
               (cn.level < self.level and i != self.DISREGARD[self._cindex])]

        nn = [n for n in self._cneighbors + nn if n is not None]
        self._nneighbors = nn
        return nn

    def interaction_set(self):
        nn, pn = self.nearest_neighbors, self.parent.nearest_neighbors
        int_set = []
        for n in pn:
            if n._has_children():
                int_set += [c for c in n if c not in nn]
            elif n not in nn:
                int_set.append(n)
        return int_set


class QuadTree():
    """Quad Tree Class"""

    def __init__(self, points, thresh, bbox=(1,1), boundary='wall'):
        self.threshold = thresh
        self.root = Node(*bbox, 0, 0)
        if boundary == 'periodic':
            self.root._cneighbors = 4*[self.root,]
        elif boundary == 'wall':
            self.root._cneighbors = 4*[None,]
        else:
            raise AttributeError('Boundary of type {} is'
                                 ' not recognized'.format(boundary))
        self._build_tree(points)
        self._depth = None

    def _build_tree(self, points):
        self.root.add_points(points)
        self.root.thresh_split(self.threshold)
        self.root.set_cneighbors()

    def __len__(self):
        l = len(self.root)
        for node in self.root.traverse():
            l += len(node)
        return l

    def __iter__(self):
        for points in self.root.get_points():
            yield points

    @property
    def depth(self):
        if self._depth is None:
            self._depth = max([node.level for node in self.root.traverse()])
        return self._depth

    @property
    def nodes(self):
        return [node for node in self.root.traverse()]

    def traverse_nodes(self):
        for node in self.root.traverse():
            yield node


def build_tree(points, tree_thresh=None, bbox=None, boundary='wall'):
    if bbox is None:
        coords = np.array([(p.x, p.y) for p in points])
        bbox = (max(coords[:, 0]) + eps, max(coords[:, 1]) + eps)
    if tree_thresh is None:
        tree_thresh = 5#max(len(points)//10, 5)  # Something less error prone?

    return QuadTree(points, tree_thresh, bbox=bbox, boundary=boundary)
