
import math
import numpy as np
import time

class LatentAbstraction:
    def __init__(self, verbose=False):
        self._verbose = verbose
        pass

    def transform(self, dataset):
        total = dataset.observations
        every = math.ceil(total / 100)

        if (self._verbose): print("Creating vector representations for %d documents" % total)
        tick = time.time()

        if (self._verbose): print("\tAllocating representation matrix")
        transform = np.empty((dataset.observations, self._latent_size), dtype='float32')

        for i, text in enumerate(dataset.data):
            if (self._verbose and i % every == 0 or i == (total - 1)):
                print("\tProgress %3.0f%%" % (100 * (i / total)))

            transform[i, :] = self.representation(text)

        if (self._verbose): print("\tDone, took %d sec" % (time.time() - tick))
        return transform
