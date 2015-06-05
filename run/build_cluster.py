
import run
import dataset
import model

import sys
import os.path as path

if (len(sys.argv) < 2):
    print('python3 run/build_cluster.py name')
    sys.exit(1)

name = sys.argv[1]

if (name == 'word2vec-both-l2'):
    threshold = 0.14
elif (name == 'word2vec-both-cos'):
    threshold = 0.092
elif (name == 'word2vec-title-l2'):
    threshold = 0.35
elif (name == 'word2vec-title-cos'):
    threshold = 0.024
elif (name == 'doc2vec-full-l2'):
    threshold = 1.35
elif (name == 'doc2vec-full-cos'):
    threshold = -0.66
else:
    raise NotImplementedError('no defined threshold for %s' % name)

distance = model.load(path.join(run.output_dir, 'builds/%s.distance.hd5' % name))

m = model.Cluster(threshold=threshold, verbose=True)
cluster = m.transform(distance)
model.save(path.join(run.output_dir, 'builds/%s.cluster.npz' % name), cluster)
