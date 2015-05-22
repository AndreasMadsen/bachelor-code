
import gensim
import numpy as np
import os.path as path
import time

from model._latent_abstraction import LatentAbstraction

thisdir = path.dirname(path.realpath(__file__))

class Word2Vec(LatentAbstraction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if (self._verbose): print('Loading Word2Vec')

        tick = time.time()

        self._latent_size = 300
        self._model = gensim.models.Word2Vec.load_word2vec_format(
            path.join(thisdir, '../outputs/builds/word2vec.bin.gz'), binary=True
        )

        if (self._verbose):
            print('\tpretrained model loaded, took %d min' % ((time.time() - tick) / 60))

    def representation(self, sentense):
        return np.sum([self._model[word] for word in sentense if (word in self._model)], axis=0)
