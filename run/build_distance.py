
import run
import dataset
import model

import os.path as path

connectivity = model.load(path.join(run.output_dir, 'builds/connectivity.hd5'))
vecs = model.load(path.join(run.output_dir, 'builds/word2vec.vecs.npy'))

m = model.Distance(verbose=True)
distance = m.transform(connectivity, vecs)
model.save(path.join(run.output_dir, 'builds/word2vec.distance.hd5'), distance)
