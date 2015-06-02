
import plot
import model

import sys
import os.path as path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sklearn.decomposition

vecs = model.load(path.realpath(sys.argv[1]))

model = sklearn.decomposition.PCA()
PCs = model.fit_transform(vecs)

print('variance explained: ', np.sum(model.explained_variance_ratio_[0:4] * 100))

fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(3, 2)

# PC[0], PC[1]
ax = plt.subplot(gs[0:2, 0])
plt.scatter(PCs[:, 0], PCs[:, 1], lw=0, alpha=0.5, c='SteelBlue')
plt.xlabel('PC 0')
plt.ylabel('PC 1')

# PC[2], PC[3]
ax = plt.subplot(gs[0:2, 1])
plt.scatter(PCs[:, 2], PCs[:, 3], lw=0, alpha=0.5, c='SteelBlue')
plt.xlabel('PC 2')
plt.ylabel('PC 3')

plt.subplot(gs[2, :])
plt.plot(np.arange(0, PCs.shape[1]), model.explained_variance_ratio_ * 100, color='IndianRed')
plt.xlabel('PC')
plt.ylabel('explained [%]')

fig.set_tight_layout(True)
plt.show()
