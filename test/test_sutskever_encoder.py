
import test
from nose.tools import *

import dataset
import numpy as np
import theano
import theano.tensor as T

import neural

def test_sutskever_encoder_float_fast():
    # obs: 2, dims: 3, time: 6
    x1 = np.asarray([
        [0, 0, 0, 1, 1, 1],  # <EOS>
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ], dtype='int8')
    x2 = np.asarray([
        [0, 0, 0, 0, 0, 1],  # <EOS>
        [0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 0]
    ], dtype='int8')
    x = np.asarray([x1, x2])

    x_input = T.tensor3('x', dtype='int8')
    x_input.tag.test_value = x

    # Create encoder
    encoder = neural.network.SutskeverEncoder()
    encoder.test_value(x, None)

    # Setup layers for a logistic classifier model
    encoder.set_input(neural.layer.Input(3))
    encoder.push_layer(neural.layer.LSTM(4))

    # Enforce weights for consistent results
    weights = encoder.weight_list()
    weights[0].set_value(np.tile(np.asarray([
        [-0.48490700, -0.86166465, -0.68156427, +0.22538510],
        [+0.92648524, +0.27489561, -1.65431046, -0.99628597],
        [+0.79819334, -0.28792551, -0.34098715, +0.58490205]
    ], dtype='float32'), (1, 4)))
    weights[1].set_value(np.tile(np.asarray([
        [+1.90923262, -0.70851284, +1.05009556, +0.38148439],
        [-1.22912812, -0.98054105, +0.43493664, -0.03531076],
        [-0.01739309, -0.03275032, -0.12585467, +1.56854463],
        [-0.20889190, +0.67603844, +1.11195946, +0.08580784]
    ], dtype='float32'), (1, 4)))

    # Perform forward pass
    (s_enc, b_enc) = encoder.forward_pass(x_input)

    # Check that the gradient can be calculated
    T.grad(T.sum(s_enc), weights)
    T.grad(T.sum(b_enc), weights)

    # Assert output
    assert_equal(s_enc.tag.test_value.shape, (2, 4))
    assert_equal(b_enc.tag.test_value.shape, (2, 4))

    assert(np.allclose(s_enc.tag.test_value, [
        [1.75992775, 0.16468449, 0.47424576, 0.59295964],
        [3.56025982, 0.03904305, 1.34779572, 1.18748903]
    ]))

    assert(np.allclose(b_enc.tag.test_value, [
        [1.58437216, 0.04333938, 0.31211659, 0.42492560],
        [3.54513669, 0.00469692, 1.24734366, 1.04889858]
    ]))

def test_sutskever_encoder_indexed_fast():
    # obs: 2, dims: 3, time: 6
    x1 = np.asarray([1, 1, 2, 0, 0, 0], dtype='int32')
    x2 = np.asarray([2, 2, 1, 1, 2, 0], dtype='int32')
    x = np.asarray([x1, x2], dtype='int32')

    x_input = T.imatrix('x')
    x_input.tag.test_value = x

    # Create encoder
    encoder = neural.network.SutskeverEncoder(indexed_input=True)
    encoder.test_value(x, None)

    # Setup layers for a logistic classifier model
    encoder.set_input(neural.layer.Input(3, indexed=True))
    encoder.push_layer(neural.layer.LSTM(4))

    # Enforce weights for consistent results
    weights = encoder.weight_list()
    weights[0].set_value(np.tile(np.asarray([
        [-0.48490700, -0.86166465, -0.68156427, +0.22538510],
        [+0.92648524, +0.27489561, -1.65431046, -0.99628597],
        [+0.79819334, -0.28792551, -0.34098715, +0.58490205]
    ], dtype='float32'), (1, 4)))
    weights[1].set_value(np.tile(np.asarray([
        [+1.90923262, -0.70851284, +1.05009556, +0.38148439],
        [-1.22912812, -0.98054105, +0.43493664, -0.03531076],
        [-0.01739309, -0.03275032, -0.12585467, +1.56854463],
        [-0.20889190, +0.67603844, +1.11195946, +0.08580784]
    ], dtype='float32'), (1, 4)))

    # Perform forward pass
    (s_enc, b_enc) = encoder.forward_pass(x_input)

    # Check that the gradient can be calculated
    T.grad(T.sum(s_enc), weights)
    T.grad(T.sum(b_enc), weights)

    # Assert output
    assert_equal(s_enc.tag.test_value.shape, (2, 4))
    assert_equal(b_enc.tag.test_value.shape, (2, 4))

    assert(np.allclose(s_enc.tag.test_value, [
        [1.75992775, 0.16468449, 0.47424576, 0.59295964],
        [3.56025982, 0.03904305, 1.34779572, 1.18748903]
    ]))

    assert(np.allclose(b_enc.tag.test_value, [
        [1.58437216, 0.04333938, 0.31211659, 0.42492560],
        [3.54513669, 0.00469692, 1.24734366, 1.04889858]
    ]))

def test_sutskever_encoder_float_train():
    def generator(items):
        d = dataset.encoder.mode(items)
        return (d.data, d.target)

    encoder = neural.network.SutskeverEncoder()
    # Setup theano tap.test_value
    encoder.test_value(*generator(10))

    # Setup layers
    encoder.set_input(neural.layer.Input(10))
    encoder.push_layer(neural.layer.LSTM(15))
    encoder.push_layer(neural.layer.Softmax(10))

    # Setup loss function
    encoder.set_loss(neural.loss.NaiveEntropy(time=False))

    # Compile train, test and predict functions
    encoder.compile()

    test.classifier(
        encoder, generator,
        y_shape=(100, 10), performance=0.8, asserts=True,
        epochs=600, learning_rate=0.05, momentum=0.04
    )

def test_sutskever_encoder_indexed_train():
    def generator(items):
        d = dataset.encoder.mode(items, indexed=True)
        return (d.data, d.target)

    encoder = neural.network.SutskeverEncoder(indexed_input=True)
    # Setup theano tap.test_value
    encoder.test_value(*generator(10))

    # Setup layers
    encoder.set_input(neural.layer.Input(10, indexed=True))
    encoder.push_layer(neural.layer.LSTM(15))
    encoder.push_layer(neural.layer.Softmax(10))

    # Setup loss function
    encoder.set_loss(neural.loss.NaiveEntropy(time=False))

    # Compile train, test and predict functions
    encoder.compile()

    test.classifier(
        encoder, generator,
        y_shape=(100, 10), performance=0.8, asserts=True,
        epochs=600, learning_rate=0.05, momentum=0.04
    )
