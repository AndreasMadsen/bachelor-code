import theano
import theano.tensor as T
from .base import Layer


__all__ = [
    "SoftmaxLayer"
]

class SoftmaxLayer(Layer):
    '''
        Takes a input layer of shape (batch_size, n_features) and passes
        it though a softmax transform.
    '''
    def __init__(self, incomings, **kwargs):
        super(SoftmaxLayer, self).__init__(incomings, **kwargs)

        assert len(self.input_shape) == 2, \
            "Input shape must be (batch_size, num_features)"

    def get_output_shape_for(self, input_shape):
        #
        return input_shape

    def get_output_for(self, input, *args, **kwargs):
        return T.nnet.softmax(input)
