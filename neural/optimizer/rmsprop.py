
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

        self._clip = T.scalar('clip')
        self._clip.tag.test_value = 1
        self.params.append(
            theano.Param(self._clip, default=1, name='clipping_value')
        )

    def each_update(self, Wi, gWi):
        r_tm1 = theano.shared(
            np.zeros(self.get_weight_shape(Wi), dtype='float32'),
            name="r" + Wi.name[1:], borrow=True)

        r = (1 - self._decay) * (gWi**2) + self._decay * r_tm1
        ΔWi = (self._learning_rate / T.sqrt(r + 1e-7)) * gWi

        return [(r_tm1, r), ΔWi]

    def update(self, W, gW):
        updates = [self.each_update(Wi, gWi) for (Wi, gWi) in zip(W, gW)]

        norms = [T.sum(ΔWi ** 2) for (r_update, ΔWi) in updates]
        norm = T.sqrt(T.sum(norms))

        clipped_updates = []

        for Wi, update_i in zip(W, updates):
            (r_update, ΔWi) = update_i
            clipped_updates.append(r_update)

            ΔWi_clipped = T.switch(
                T.ge(norm, self._clip),
                ΔWi / norm * self._clip,
                ΔWi
            )

            clipped_updates.append((Wi, Wi - ΔWi_clipped))

        return clipped_updates
