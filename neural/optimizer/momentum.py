
import numpy as np
import theano
import theano.tensor as T

from neural.optimizer._abstract import OptimizerAbstract

class Momentum(OptimizerAbstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._learning_rate = T.scalar('eta')
        self._learning_rate.tag.test_value = 0.07
        self.params.append(
            theano.Param(self._learning_rate, default=0.07, name='learning_rate')
        )

        self._momentum = T.scalar('m')
        self._momentum.tag.test_value = 0.2
        self.params.append(
            theano.Param(self._momentum, default=0.2, name='momentum')
        )

    def each_update(self, Wi, gWi):
        ΔWi_tm1 = theano.shared(
            self.initialize(Wi),
            name="Δ" + Wi.name, borrow=True)
        self._state.append(ΔWi_tm1)

        ΔWi = - self._momentum * ΔWi_tm1 - self._learning_rate * gWi
        return [(ΔWi_tm1, ΔWi), (Wi, Wi + ΔWi)]
