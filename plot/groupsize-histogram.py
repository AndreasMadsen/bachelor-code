
import plot
import model

import sys
import numpy as np
import os.path as path
import matplotlib.pyplot as plt

group_size = model.load(path.realpath(sys.argv[1]))['group_size']
sizes = np.unique(group_size)

print("%7s | %3s" % ("size", "count"))

for size in sizes:
    print("%7d | %3d" % (size, np.sum(group_size == size)))
