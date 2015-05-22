
import plot
import model

import sys
import os.path as path
import matplotlib.pyplot as plt

distance = model.load(path.realpath(sys.argv[1]))

# the histogram of the data
plt.hist(distance.data, 200, normed=1, facecolor='green', alpha=0.75)
plt.xlabel('Distance')
plt.ylabel('Probability')
plt.grid(True)
plt.show()
