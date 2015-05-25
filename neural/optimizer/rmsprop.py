
import numpy as np
import theano
import theano.tensor as T

from neural.optimizer._abstract import OptimizerAbstract

class RMSprop(OptimizerAbstract):
    """
    Implements simple RMSprop

    source: http://climin.readthedocs.org/en/latest/rmsprop.html
    see also: https://github.com/BRML/climin/blob/master/climin/rmsprop.py
    """

    def __init__(self, *args, clipping_stratagy='L2', **kwargs):
        super().__init__(*args, clipping_stratagy=clipping_stratagy, **kwargs)

        self._learning_rate = T.scalar('eta')
        self._learning_rate.tag.test_value = 0.1
        self.params.append(
            theano.Param(self._learning_rate, default=0.1, name='learning_rate')
        )

        self._decay = T.scalar('gamma')
        self._decay.tag.test_value = 0.9
        self.params.append(
            theano.Param(self._decay, default=0.9, name='decay')
        )

    def each_update(self, Wi, gWi):
        r_tm1 = theano.shared(
            self.initialize(Wi),
            name="r" + Wi.name[1:], borrow=True)
        self._state.append(r_tm1)

        r = (1 - self._decay) * (gWi**2) + self._decay * r_tm1
        ΔWi = (self._learning_rate / T.sqrt(r + 1e-4)) * gWi

        return [(r_tm1, r), (Wi, Wi + ΔWi)]
