
import plot
import model

import sys
import numpy as np
import os.path as path
import matplotlib.pyplot as plt

distance = model.load(path.realpath(sys.argv[1]))

# the histogram of the data
fig = plt.figure(figsize=(4, 3))
density, distance, _ = plt.hist(distance.data, 50, normed=1, facecolor='SteelBlue', alpha=0.75)
plt.xlabel('Distance')
plt.ylabel('Density')
plt.grid(True)

properbility = np.cumsum(np.diff(distance) * density)
print('threshold: %f' % distance[np.where(properbility > 0.001)[0][0]])

fig.set_tight_layout(True)
plt.show()
