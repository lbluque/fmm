from time import time
from functools import wraps, partial
import numpy as np
from copy import deepcopy
import fmm
import matplotlib.pyplot as plt

#############################################################################
#  Error vs terms
#############################################################################
def reset_particles(particles):
    for p in particles:
        p.phi = 0

particles = [fmm.Particle(*p, 1) for p in np.random.rand(100,2)]
phiDS = fmm.potentialDS(particles)

def run(particles, nterms, tree_thresh=1):
    reset_particles(particles)
    fmm.potential(particles, nterms=nterms, tree_thresh=tree_thresh)
    phi = np.array([p.phi for p in particles])
    norm_err = 100*np.linalg.norm(phiDS - phi)/np.linalg.norm(phiDS)
    return norm_err

error_func = partial(run, particles)

terms = np.array((10, 15, 20, 25, 30))
thresh = np.array((10,))

errors = np.array([[error_func(ts, tree_thresh=tr) for ts in terms] for tr in thresh])

for er in errors:
    plt.semilogy(terms, er, 'o--')
plt.legend(thresh)
plt.xlabel('Number of terms')
plt.ylabel('Percent total error')
plt.tight_layout()
plt.show()
