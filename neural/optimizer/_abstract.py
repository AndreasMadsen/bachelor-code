
import itertools

class OptimizerAbstract:
    def __init__(self, *args, **kwargs):
        self.params = []

    def get_weight_shape(self, Wi):
        return Wi.get_value(
            borrow=True,
            return_internal_type=True
        ).shape

    def each_update(self, Wi, gWi):
        raise NotImplemented

    def update(self, W, gW):
        """
        Generate update equations for the weights
        """
        return list(itertools.chain(*[
            self.each_update(Wi, gWi) for (Wi, gWi) in zip(W, gW)
        ]))
