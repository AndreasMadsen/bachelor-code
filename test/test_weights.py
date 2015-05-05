
import test
from nose.tools import *

import numpy as np

import dataset
import neural

def all_equal(old, new):
    return np.all([
        np.allclose(old_item, new_item) for old_item, new_item in zip(old, new)
    ])

def create_network():
    sutskever = neural.network.Sutskever(max_output_size=9)
    # Setup theano tap.test_value
    sutskever.test_value(*dataset.network.copy(10).astuple())

    # Setup layers
    sutskever.set_encoder_input(neural.layer.Input(10))
    sutskever.push_encoder_layer(neural.layer.LSTM(9))

    sutskever.set_decoder_input(neural.layer.Input(10))
    sutskever.push_decoder_layer(neural.layer.LSTM(9))
    sutskever.push_decoder_layer(neural.layer.Softmax(10))

    return sutskever

def test_get():
    sutskever = create_network()
    assert(all_equal(sutskever.get_weights(), sutskever.get_weights()))

def test_reset():
    sutskever = create_network()
    old = sutskever.get_weights()
    sutskever.reset_weights()
    assert(not all_equal(old, sutskever.get_weights()))

def test_set():
    sutskever = create_network()
    old = sutskever.get_weights()
    sutskever.reset_weights()
    sutskever.set_weights(old)
    assert(all_equal(sutskever.get_weights(), sutskever.get_weights()))
