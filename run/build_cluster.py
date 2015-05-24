
import run
import dataset
import model

import sys
import os.path as path

if (len(sys.argv) < 2):
    print('python3 run/build_cluster.py name')
    sys.exit(1)

name = sys.argv[1]

if (name == 'word2vec-l2'):
    threshold = 0.11
elif (name == 'word2vec-cos'):
    threshold = 0.2
else:
    raise NotImplementedError('no defined threshold for %s' % name)

distance = model.load(path.join(run.output_dir, 'builds/%s.distance.hd5' % name))

m = model.Cluster(threshold=threshold, verbose=True)
cluster = m.transform(distance)
model.save(path.join(run.output_dir, 'builds/%s.cluster.npz' % name), cluster)
