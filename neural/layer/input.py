
import numpy as np
import theano
import theano.tensor as T

from neural.layer._abstract import LayerAbstract

class Input(LayerAbstract):
    def __init__(self, size, *args, indexed=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.output_size = size
        self.input_size = np.nan
        self.layer_index = 0
        self.indexed = indexed
