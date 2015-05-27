
import itertools

import numpy as np
import theano
import theano.tensor as T

class OptimizerAbstract:
    def __init__(self, *args, clipping_stratagy=None, **kwargs):
        self.params = []
        self._state = []

        self._clipping_stratagy = clipping_stratagy

        self._clip = T.scalar('clip')
        self._clip.tag.test_value = 10
        if (self._clipping_stratagy is not None):
            self.params.append(
                theano.Param(self._clip, default=10, name='clip')
            )

    def initialize(self, Wi):
        shape = Wi.get_value(
            borrow=True,
            return_internal_type=True
        ).shape

        return np.zeros(shape, dtype='float32')

    def reset_state(self):
        for state in self._state:
            state.set_value(self.initialize(state), borrow=True)

    def each_update(self, Wi, gWi):
        raise NotImplemented

    def _clip_max(self, gW):
        # Clips gW such that max(|gW|) \in [-clip, clip]
        max_value = 0
        for gWi in gW:
            max_value = T.maximum(T.max(T.abs_(gWi)), max_value)
        scale = T.minimum(1, self._clip / max_value)

        return [gWi * scale for gWi in gW]

    def _clip_L2(self, gW):
        # Clips gW such that ||gW||_2 \in [-clip, clip]
        norm = T.sqrt(T.sum([T.sum(gWi ** 2) for gWi in gW]))
        scale = T.minimum(1, self._clip / norm)

        return [gWi * scale for gWi in gW]

    def update(self, W, gW):
        """
        Generate update equations for the weights
        """
        if (self._clipping_stratagy == 'max'):
            gW = self._clip_max(gW)
        elif (self._clipping_stratagy == 'L2'):
            gW = self._clip_L2(gW)

        return list(itertools.chain(*[
            self.each_update(Wi, gWi) for (Wi, gWi) in zip(W, gW)
        ]))
