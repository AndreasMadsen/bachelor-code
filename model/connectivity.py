import time
import math

import numpy as np
import scipy
import theano
import theano.tensor as T


class Connectivity:
    def __init__(self, days=7, verbose=False):
        """Construct an object, with the primary method transform, there can
        create a sparse connectivity matrix.

        Parameters
        ----------
        days: Integer
            Connected observerions are those within [-days, days]

        verbose : boolean
            If true progressiv information will be printed.
        """

        # Initialize verbose flags
        self._verbose = verbose

        # Build theano function
        visits = T.vector('visits', dtype='int16')
        diff = T.abs_(visits.dimshuffle(('x', 0)) - visits.dimshuffle((0, 'x')))
        connected = diff < T.constant(days, dtype='int16')

        self._fn = theano.function(
            inputs=[visits],
            outputs=[connected],
            name='connectivity'
        )

        if (self._verbose): print("Initialized new Connectivity builder")

    def transform(self, visits):
        """Builds the connectivity matrix, as such no fitting is done.

        Parameters
        ----------
        visits : iterable, [n_samples]
            An iterable which yields visit dates.

        Returns
        -------
        X : array, [n_samples, n_sampels]
            Sparse connectivity matrix.
        """
        if (self._verbose):
            print("Creating connectivity matrix from %d documents" % visits.shape[0])

        tick = time.time()

        # Convert visits to days since unix epoch
        visits = visits.astype('datetime64[D]').astype('int16')
        connectivity = self._fn(visits)
        if (self._verbose): print("\tSparseifying results")
        connectivity = scipy.sparse.triu(sparse.coo_matrix(connectivity), 1)

        if (self._verbose): print("\tDone, took %d min" % ((time.time() - tick) / 60))
        return connectivity
