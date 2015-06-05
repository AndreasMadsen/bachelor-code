
import gensim
import numpy as np
import os.path as path
import time

from model._latent_abstraction import LatentAbstraction

thisdir = path.dirname(path.realpath(__file__))

class Doc2Vec:
    def __init__(self, epochs=10, latent_size=500, verbose=False, **kwargs):
        self._verbose = verbose
        if (self._verbose): print('Creating Doc2Vec object')

        tick = time.time()

        self._latent_size = latent_size
        self._epochs = epochs
        self._model = gensim.models.Doc2Vec(
            size=self._latent_size, alpha=0.025, min_alpha=0.025, **kwargs
        )

    def fit_transform(self, dataset):
        if (self._verbose):
            print("Creating vector representations for %d documents" % dataset.observations)
        tick = time.time()

        if (self._verbose): print("\tBuilding corpus")
        sentences = list([
            gensim.models.doc2vec.LabeledSentence(words=sent, labels=["SENT_%d" % i])
            for i, sent
            in enumerate(dataset.data)
        ])

        if (self._verbose): print("\tBuilding vocabulary")
        self._model.build_vocab(sentences)

        if (self._verbose): print("\tOptimizing parameters")
        for epoch in range(0, self._epochs):
            self._model.train(sentences)
            self._model.alpha -= 0.002  # decrease the learning rate
            self._model.min_alpha = self._model.alpha  # fix the learning rate, no decay
            if (self._verbose): print("\tEpoch %d completed" % (epoch + 1))

        if (self._verbose): print("\tAllocating representation matrix")
        transform = np.empty((dataset.observations, self._latent_size), dtype='float32')

        for i in range(dataset.observations):
            transform[i, :] = self._model['SENT_%d' % i]

        if (self._verbose): print("\tDone, took %d sec" % (time.time() - tick))

        return transform
