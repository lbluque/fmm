import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import fmm

default_dpi = plt.rcParamsDefault['figure.dpi']
params = {"ytick.color" : "w",
          "xtick.color" : "w",
          "axes.labelcolor" : "w",
          "axes.edgecolor" : "w",
          "figure.dpi" : 2*default_dpi}
plt.rcParams.update(params)


tree = fmm.build_tree(particles, tree_thresh=5)
fig, ax = plt.subplots(1)
pos = np.array([p.pos for p in particles])
pca = PatchCollection([Rectangle((p.x0,p.y0), p.w, p.h) for p in tree.root.traverse()],
                     facecolors='none', edgecolors='c', linewidth=1, alpha=.3)
ax.add_collection(pc)
ax.scatter(pos[:,0], pos[:,1], c='m',marker='.')

particles = [fmm.Particle(*p, 1) for p in np.random.rand(100,2)]

c = tree.root._children[2]._children[1]._children[1]._children[2]
cr = Rectangle((c.x0,c.y0), c.w, c.h, facecolor='none', edgecolor='r', linewidth=3)

pca = PatchCollection([Rectangle((p.x0,p.y0), p.w, p.h) for p in tree.root.traverse()],
                     facecolors='none', edgecolors='c', linewidth=1, alpha=.3)

pc = []
for n in c.nearest_neighbors:
    pc.append(Rectangle((n.x0,n.y0), n.w, n.h))
pc = PatchCollection(pc, facecolors='none', edgecolors='y', linewidth=2, alpha=.8)

pci = []
for n in c.interaction_set():
    pci.append(Rectangle((n.x0,n.y0), n.w, n.h))
pci = PatchCollection(pci, facecolors='none', edgecolors='m', linewidth=1, alpha=.8)

fig, ax = plt.subplots()
ax.add_collection(pca)
ax.add_collection(pci)
#ax.legend('interaction set')
ax.add_collection(pc)
#ax.legend('nearest neighbors')
ax.add_patch(cr)
ax.axis('off')
fig.tight_layout()


pcc = []
for n in c._cneighbors:
    if n is not None:
        pcc.append(Rectangle((n.x0,n.y0), n.w, n.h))
pcc = PatchCollection(pcc, facecolors='none', edgecolors='r', linewidth=2, alpha=.8)
