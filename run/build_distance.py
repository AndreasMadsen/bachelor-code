
import run
import dataset
import model

import sys
import os.path as path

if (len(sys.argv) < 3):
    print('python3 run/build_distance.py name norm')
    sys.exit(1)

name = sys.argv[1]
norm = sys.argv[2]

connectivity = model.load(path.join(run.output_dir, 'builds/connectivity.hd5'))
vecs = model.load(path.join(run.output_dir, 'builds/%s.vecs.npy' % name))

m = model.Distance(norm=norm, verbose=True)
distance = m.transform(connectivity, vecs)
model.save(path.join(run.output_dir, 'builds/%s-%s.distance.hd5' % (name, norm)), distance)
