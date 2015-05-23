
import numpy as np
import theano
import theano.tensor as T

from neural.optimizer._abstract import OptimizerAbstract

class RMSgrave(OptimizerAbstract):
    """
    Implements simple RMSprop as Alex Graves does it

    source: http://theanets.readthedocs.org/en/latest/generated/theanets.trainer.RmsProp.html
    source: http://arxiv.org/abs/1308.0850 , page 23
    """

    def __init__(self, *args, clipping_stratagy='max', **kwargs):
        super().__init__(*args, clipping_stratagy=clipping_stratagy, **kwargs)

        self._learning_rate = T.scalar('eta')
        self._learning_rate.tag.test_value = 1e-4
        self.params.append(
            theano.Param(self._learning_rate, default=1e-4, name='learning_rate')
        )

        self._momentum = T.scalar('m')
        self._momentum.tag.test_value = 0.9
        self.params.append(
            theano.Param(self._momentum, default=0.9, name='momentum')
        )

        self._decay = T.scalar('gamma')
        self._decay.tag.test_value = 0.95
        self.params.append(
            theano.Param(self._decay, default=0.95, name='decay')
        )

    def each_update(self, Wi, gWi):
        f_tm1 = theano.shared(
            np.zeros(self.get_weight_shape(Wi), dtype='float32'),
            name="f" + Wi.name[1:], borrow=True)

        g_tm1 = theano.shared(
            np.zeros(self.get_weight_shape(Wi), dtype='float32'),
            name="g" + Wi.name[1:], borrow=True)

        v_tm1 = theano.shared(
            np.zeros(self.get_weight_shape(Wi), dtype='float32'),
            name="v" + Wi.name[1:], borrow=True)

        f_t = self._decay * f_tm1 + (1 - self._decay) * gWi  # (39)
        g_t = self._decay * g_tm1 + (1 - self._decay) * (gWi**2)  # (38)

        Δ_t = gWi / T.sqrt(g_t - f_t**2 + 1e-4)
        v_t = self._momentum * v_tm1 - self._learning_rate * Δ_t  # (40)

        return [(f_tm1, f_t), (g_tm1, g_t), (v_tm1, v_t), (Wi, Wi + v_t)]
