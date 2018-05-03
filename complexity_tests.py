from time import time
from functools import wraps, partial
import numpy as np
from copy import deepcopy
import fmm
import matplotlib.pyplot as plt

#############################################################################
#  Time complexity testing
#############################################################################
def time_function(func, *args, **kwargs):
    @wraps(func)
    def tfunc(*args, **kwargs):
        start = time()
        res = func(*args, **kwargs)
        runtime = (time() - start)
        return runtime
    return tfunc

fmmtimed = time_function(fmm.potential)
fmmtimedf = time_function(partial(fmm.potential, tree_thresh=10))
directtimed = time_function(fmm.potentialDS)
treetimed = time_function(fmm.build_tree)

particles = [fmm.Particle(*p, 1) for p in np.random.rand(100,2)]

size = (100, 250, 500, 100) #np.array([100, 250, 500]), #750, 1000, 2500, 5000, 7500, 10000])

pinit = lambda n: [fmm.Particle(*p, 1) for p in np.random.rand(n,2)]

#t_fmm_notree = list(map(fmmNOtree, map(tinit, size)))
#t_fmm = list(map(fmmtimed, map(pinit, size)))
t_fmmf = list(map(fmmtimedf, map(pinit, size)))
t_direct = list(map(directtimed, map(pinit, size)))

size = np.array(size)
sizen = size/size[0]
plt.semilogy(size, t_direct[0]*(sizen)**2)
plt.semilogy(size, t_fmmf[0]*sizen*np.log(np.e*sizen))
plt.semilogy(size, t_fmmf[0]*sizen)
#plt.semilogy(size, t_fmm, 'o')
plt.semilogy(size, t_fmmf, 'o--')
plt.semilogy(size, t_direct, 'o--')
plt.legend([r'O(n$^2$)', 'O(nlog(n))', 'O(n)', 'FMM', 'Direct'])
plt.xlabel('Number of Particles')
plt.ylabel('Runtime (s)')
plt.tight_layout()
plt.show()
