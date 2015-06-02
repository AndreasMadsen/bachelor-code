
import plot
import model
import dataset

import sys
import numpy as np
import os.path as path

node_id = int(sys.argv[1])
cluster = model.load(path.realpath(sys.argv[2]))

print('group id: %d' % cluster['node_to_group'][node_id])
