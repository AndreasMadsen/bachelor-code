
import run
import dataset
import model

import os.path as path

distance = model.load(path.join(run.output_dir, 'builds/word2vec.distance.hd5'))

m = model.Cluster(verbose=True)
cluster = m.transform(distance)
model.save(path.join(run.output_dir, 'builds/word2vec.cluster.npz'), cluster)
