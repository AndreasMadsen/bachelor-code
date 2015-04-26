import theano
import theano.tensor as T
from .base import Layer


__all__ = [
    "RepeatLayer"
]

class RepeatLayer(Layer):
    '''
        Takes a input layer of shape (batch_size, n_features) and repeats the
        n_features n times such that the output shape is
        (batch_size, n_repeats, n_featues).

        This can be used for recurrent layers where we want to use the
        same input for all timesteps e.g. in sutskever models.
    '''
    def __init__(self, incomings, n_repeat, **kwargs):
        super(RepeatLayer, self).__init__(incomings, **kwargs)

        self.n_repeat = n_repeat

        assert(len(self.input_shape) == 2,
               "Input shape must be (batch_size, num_features")

    def get_output_shape_for(self, input_shape):
        #
        return (input_shape[0], self.n_repeat, input_shape[1])

    def get_output_for(self, inputs, *args, **kwargs):

        return theano.tensor.stack(*[inputs] * self.n_repeat)
