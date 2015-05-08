
import numpy as np

class LayerAbstract:
    def __init__(self, *args, **kwargs):
        self.weights = []
        self.outputs_info = []
        self.indexed = False

    def _create_weights(self, *shape):
        return 0.5 * np.random.randn(*shape).astype('float32')

    def reset_weights(self):
        for weight in self.weights:

            shape = weight.get_value(
                borrow=True,
                return_internal_type=True
            ).shape

            weight.set_value(self._create_weights(*shape), borrow=True)

    def set_weights(self, new_weights):
        for new_weight, weight in zip(new_weights, self.weights):
            weight.set_value(new_weight, borrow=True)

    def get_weights(self):
        return [weight.get_value(borrow=True) for weight in self.weights]

    def infer_taps(self):
        taps = 0

        for info in self.outputs_info:
            # If info is dict it can have a taps array, default this is [-1]
            if (isinstance(info, dict)):
                # However of no `inital` property is provided it is treated
                # as None (no taps)
                if ('initial' in info):
                    if ('taps' in info):
                        taps += len(info.taps)
                    else:
                        taps += 1
            # If info is a numpy array or a scalar
            elif (info is not None):
                taps += 1

        return taps

    def setup(self, *args, **kwargs):
        raise NotImplemented

    def scanner(self, *args, **kwargs):
        raise NotImplemented
