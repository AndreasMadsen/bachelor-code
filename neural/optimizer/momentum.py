
import numpy as np
import theano
import theano.tensor as T

from neural.optimizer._abstract import OptimizerAbstract

class Momentum(OptimizerAbstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._learning_rate = T.scalar('eta')
        self._learning_rate.tag.test_value = 0.1
        self.params.append(
            theano.Param(self._learning_rate, default=0.1, name='learning_rate')
        )

        self._momentum = T.scalar('m')
        self._momentum.tag.test_value = 0.9
        self.params.append(
            theano.Param(self._momentum, default=0.9, name='momentum')
        )

    def each_update(self, Wi, gWi):
        ΔWi_tm1 = theano.shared(
            np.zeros(self.get_weight_shape(Wi), dtype='float32'),
            name="Δ" + Wi.name, borrow=True)

        ΔWi = - self._momentum * ΔWi_tm1 - self._learning_rate * gWi
        return [(ΔWi_tm1, ΔWi), (Wi, Wi + ΔWi)]
