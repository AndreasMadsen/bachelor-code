
import run
import dataset
import model

import os.path as path

m = model.Doc2Vec(verbose=True, workers=3)

vecs = m.fit_transform(dataset.news.words(100000))
model.save(path.join(run.output_dir, 'builds/doc2vec.vecs.npy'), vecs)
