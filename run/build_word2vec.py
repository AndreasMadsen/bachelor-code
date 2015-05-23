
import run
import dataset
import model

import os.path as path

m = model.Word2Vec(verbose=True)
vecs = m.transform(dataset.news.words(100000))
model.save(path.join(run.output_dir, 'builds/word2vec.vecs.npy'), vecs)
