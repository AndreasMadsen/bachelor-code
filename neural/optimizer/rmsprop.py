
import numpy as np
import theano
import theano.tensor as T

from neural.optimizer._abstract import OptimizerAbstract

class RMSprop(OptimizerAbstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            np.zeros(self.get_weight_shape(Wi), dtype='float32'),
            name="r" + Wi.name[1:], borrow=True)

        r = (1 - self._decay) * (gWi**2) + self._decay * r_tm1
        ΔWi = - (self._learning_rate / T.sqrt(r)) * gWi

        return [(r_tm1, r), (Wi, Wi + ΔWi)]
