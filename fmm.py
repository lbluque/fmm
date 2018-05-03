'''
Implementation of the 2D Fast Multipole Method for a Coulomb potential
'''

__author__ = 'Luis Barroso-Luque'

from itertools import chain
import numpy as np
from scipy.special import binom
from quadtree import build_tree


class Point():
    """Point in 2D"""

    def __init__(self, x, y):
        self.x, self.y = x, y
        self.pos = (x, y)


class Particle(Point):
    """A Charged Particle"""

    def __init__(self, x, y, charge):
        super(Particle, self).__init__(x, y)
        self.q = charge
        self.phi = 0


def distance(p1, p2):
    """Distance bewteen 2 points in 2D"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def multipole(particles, center=(0,0), nterms=5):
    """Compute a multiple expansion up to nterms terms"""

    coeffs = np.empty(nterms + 1, dtype=complex)
    coeffs[0] = sum(p.q for p in particles)
    coeffs[1:] = [sum([-p.q*complex(p.x - center[0], p.y - center[1])**k/k
                  for p in particles]) for k in range(1, nterms+1)]

    return coeffs


def _shift_mpexp(coeffs, z0):
    """Update multipole expansion coefficients according for a center shift"""
    shift = np.empty_like(coeffs)
    shift[0] = coeffs[0]
    shift[1:] = [sum([coeffs[k]*z0**(l - k)*binom(l-1, k-1) - (coeffs[0]*z0**l)/l
                  for k in range(1, l)]) for l in range(1, len(coeffs))]

    return shift


def _outer_mpexp(tnode, nterms):
    """Compute outer multipole expansion recursively"""

    if tnode.is_leaf():
        tnode.outer = multipole(tnode.get_points(), center=tnode.center, nterms=nterms)
    else:
        tnode.outer = np.zeros((nterms + 1), dtype=complex)
        for child in tnode:
            _outer_mpexp(child, nterms)
            z0 = complex(*child.center) - complex(*tnode.center)
            tnode.outer += _shift_mpexp(child.outer, z0)


def _convert_oi(coeffs, z0):
    """Convert outer to inner expansion about z0"""

    inner = np.empty_like(coeffs)
    inner[0] = (sum([(coeffs[k]/z0**k)*(-1)**k for k in range(1, len(coeffs))]) +
          coeffs[0]*np.log(-z0))
    inner[1:] = [(1/z0**l)*sum([(coeffs[k]/z0**k)*binom(l+k-1, k-1)*(-1)**k
                 for k in range(1, len(coeffs))]) - coeffs[0]/((z0**l)*l)
                 for l in range(1, len(coeffs))]
    return inner


def _shift_texp(coeffs, z0):
    """Shift inner expasions (Taylor) to new center"""
    shift = np.empty_like(coeffs)
    shift = [sum([coeffs[k]*binom(k,l)*(-z0)**(k-l)
              for k in range(l,len(coeffs))])
              for l in range(len(coeffs))]
    return shift


def _inner(tnode):
    """Compute the inner expansions for all cells recursively and potential
    for all particles"""

    z0 = complex(*tnode.parent.center) - complex(*tnode.center) # check sign
    tnode.inner = _shift_texp(tnode.parent.inner, z0)
    for tin in tnode.interaction_set():
        z0 = complex(*tin.center) - complex(*tnode.center)
        tnode.inner += _convert_oi(tin.outer, z0)

    if tnode.is_leaf():
        # Compute potential due to all far enough particles
        z0, coeffs = complex(*tnode.center), tnode.inner
        for p in tnode.get_points():
            z = complex(*p.pos)
            p.phi -= np.real(np.polyval(coeffs[::-1], z-z0))
        # Compute potential directly from particles in interaction set
        for nn in tnode.nearest_neighbors:
            potentialDDS(tnode.get_points(), nn.get_points())

        # Compute all-to-all potential from all particles in leaf cell
        _ = potentialDS(tnode.get_points())
    else:
        for child in tnode:
            _inner(child)


def potential(particles, bbox=None, tree_thresh=None, nterms=5, boundary='wall'):
    """Fast Mulitipole Method evaluation of all-to-all potential"""

    tree = build_tree(particles, tree_thresh, bbox=bbox, boundary=boundary)
    _outer_mpexp(tree.root, nterms)
    tree.root.inner = np.zeros((nterms + 1), dtype=complex)
    any(_inner(child) for child in tree.root)


def potentialFMM(tree, nterms=5):
    """Same as above but takes a prebuilt tree"""
    _outer_mpexp(tree.root, nterms)
    tree.root.inner = np.zeros((nterms + 1), dtype=complex)
    any(_inner(child) for child in tree.root)


def potentialDDS(particles, sources):
    """Direct sum calculation of all-to-all potential from seperate sources"""

    for i, particle in enumerate(particles):
        for source in sources:
            r = distance(particle.pos, source.pos)
            particle.phi -= particle.q*np.log(r)


def potentialDS(particles):
    """Direct sum calculation of all-to-all potential"""

    phi = np.zeros((len(particles),))

    for i, particle in enumerate(particles):
        for source in (particles[:i] + particles[i+1:]):
            r = distance(particle.pos, source.pos)
            particle.phi -= particle.q*np.log(r)
        phi[i] = particle.phi

    return phi
