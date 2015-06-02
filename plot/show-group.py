
import plot
import model
import dataset

import sys
import numpy as np
import os.path as path

group_id = int(sys.argv[1])
cluster = model.load(path.realpath(sys.argv[2]))

articles = dataset.news.fetch(100000)
nodes = cluster['group'][group_id, 0:cluster['group_size'][group_id]]

print('nodes: ', nodes)
for node_id in nodes:
    print("%6d | %s" % (node_id, articles[node_id]['title']))
