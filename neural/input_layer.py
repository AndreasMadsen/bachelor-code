
import numpy as np
import theano
import theano.tensor as T

class Input:
    def __init__(self, size):
        self.output_size = size
        self.input_size = NA
        self.layer_index = 0
        self.weights = []
        self.outputs_info = []

    def scanner(self):
        raise NotImplemented
