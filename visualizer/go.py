
import os.path as path
import sys
import numpy as np

thisdir = path.dirname(path.realpath(__file__))
sys.path.append(path.join(thisdir, '..'))

import dataset
import model

from graph_server import GraphServer

build_dir = path.join(thisdir, '..', 'outputs', 'builds')

clusters = model.load(path.join(build_dir, 'word2vec.cluster.npz'))
distance = model.load(path.join(build_dir, 'word2vec.distance.hd5'))
connectivity = model.load(path.join(build_dir, 'connectivity.hd5'))
nodes = dataset.news.fetch(100000)

server = GraphServer(clusters, distance, connectivity, nodes, verbose=True)
server.listen()

# test
# print(server._groups_from_title('Denmark'))
# print(server._fetch_single_group(300))
