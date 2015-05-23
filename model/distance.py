import time
import math

import numpy as np
import scipy
import theano
import theano.tensor as T


class Distance:
    def __init__(self, norm='L2', verbose=False):
        """Construct an object, with the primary method transform, there can
        create a sparse distance matrix.

        Parameters
        ----------
        norm: String
            Describes which norm to use (default is L2)

        verbose : boolean
            If true progressiv information will be printed.
        """

        # Initialize verbose flags
        self._verbose = verbose

        # Build theano function
        vecs = T.matrix('vecs')
        ri = T.ivector('ri')
        ci = T.ivector('ci')

        if (norm == 'L2'):
            distance = T.sqrt(T.sum((vecs[ri] - vecs[ci])**2, axis=1))

        self._fn = theano.function(
            inputs=[ri, ci, vecs],
            outputs=[distance],
            name='distance'
        )

        if (self._verbose): print("Initialized new Distance builder")

    def transform(self, connectivity, vecs):
        """Builds the distance matrix, as such no fitting is done.

        Parameters
        ----------
        connectivity : ndarray, [observerions, observerions]
            A connectivity matrix, describing which distances should be calculated.
        vecs : ndarray, [observerions, latent]
            A matrix containing a vector representation for each observation

        Returns
        -------
        X : array, [observerions, observerions]
            Sparse distance matrix
        """
        if (self._verbose):
            print("Creating distance matrix from %d documents" % vecs.shape[0])

        tick = time.time()

        jobs = math.ceil(connectivity.row.shape[0] / 100000)
        every = math.ceil(jobs / 100)

        if (self._verbose): print("\tAllocating distance matrix")
        distance = np.empty((connectivity.row.shape[0], ), dtype='float32')

        for i in range(0, jobs):
            if (self._verbose and i % every == 0 or i == (jobs - 1)):
                print("\tProgress %3.0f%%" % (100 * (i / jobs)))

            start = i * 100000
            end = min((i + 1) * 100000, connectivity.row.shape[0])

            distance[start:end] = self._fn(
                connectivity.row[start:end], connectivity.col[start:end],
                vecs
            )[0]

        if (self._verbose): print("\tSparseifying results")
        distance = scipy.sparse.coo_matrix(
            (distance, (connectivity.row, connectivity.col)),
            shape=connectivity.shape,
            dtype='float32'
        )

        if (self._verbose): print("\tDone, took %d min" % ((time.time() - tick) / 60))
        return distance
