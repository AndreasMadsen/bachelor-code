
import test
from nose.tools import *

import numpy as np
import neural

def test_softmax_bias():
    softmax = neural.layer.Softmax(4, bias=True)
    softmax.setup(2, 1, neural.layer.Input(3))

    assert_equal(len(softmax.weights), 2)
    assert_equal(softmax.weights[0].get_value().shape, (3, 4))
    assert_equal(softmax.weights[1].get_value().shape, (4,))

    softmax.weights[0].set_value(np.asarray([
        [1, 1, 1, 1],
        [2, 2, 0, 0],
        [1, 1, 0, 1],
    ], dtype='float32'))

    softmax.weights[1].set_value(np.asarray([-1, 1, 0, 0], dtype='float32'))

    (y, ) = softmax.scanner([
        [1, 1, 1],
        [2, 2, 2]
    ])

    assert(np.allclose(y.eval(), [
        [0.11245721, 0.83095266, 0.01521942, 0.04137069],
        [0.11840511, 0.87490203, 0.00079780, 0.00589504]
    ]))

def test_softmax_nobias():
    softmax = neural.layer.Softmax(4)
    softmax.setup(2, 1, neural.layer.Input(3))

    assert_equal(len(softmax.weights), 1)
    assert_equal(softmax.weights[0].get_value().shape, (3, 4))

    softmax.weights[0].set_value(np.asarray([
        [1, 1, 1, 1],
        [2, 2, 0, 0],
        [1, 1, 0, 1],
    ], dtype='float32'))

    (y, ) = softmax.scanner([
        [1, 1, 1],
        [2, 2, 2]
    ])

    assert(np.allclose(y.eval(), [
        [0.45764028, 0.45764028, 0.02278457, 0.06193488],
        [0.49485490, 0.49485490, 0.00122662, 0.00906358]
    ]))

def test_rnn_bias():
    rnn = neural.layer.RNN(4, bias=True)
    rnn.setup(2, 1, neural.layer.Input(3))

    assert_equal(len(rnn.weights), 3)
    assert_equal(rnn.weights[0].get_value().shape, (3, 4))
    assert_equal(rnn.weights[1].get_value().shape, (4, 4))
    assert_equal(rnn.weights[2].get_value().shape, (4,))

    rnn.weights[0].set_value(np.asarray([
        [1, 1, 1, 1],
        [2, 2, 0, 0],
        [1, 1, 0, 1],
    ], dtype='float32'))

    rnn.weights[1].set_value(np.asarray([
        [1, 1, 1, 1],
        [2, 2, 0, 0],
        [2, 2, 2, 1],
        [1, 1, 1, 0],
    ], dtype='float32'))

    rnn.weights[2].set_value(np.asarray([-1, 1, 0, 0], dtype='float32'))

    (b1, ) = rnn.scanner([
        [1, 1, 1],
        [2, 2, 2]
    ], [
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    assert(np.allclose(b1.eval(), [
        [0.95257413, 0.99330715, 0.73105858, 0.88079708],
        [0.99908895, 0.99987661, 0.88079708, 0.98201379]
    ]))

    (b2, ) = rnn.scanner([
        [1, 1, 1],
        [2, 2, 2]
    ], b1)

    assert(np.allclose(b2.eval(), [
        [0.99974706, 0.99996576, 0.98655336, 0.97548460],
        [0.99999708, 0.99999960, 0.99680414, 0.99721269]
    ]))

    (b2m, ) = rnn.scanner([
        [1, 1, 1],
        [2, 2, 2]
    ], b1, mask=np.asarray([[1], [0]], dtype='int8'))

    assert(np.allclose(b2m.eval(), [
        [0.95257413, 0.99330715, 0.73105858, 0.88079708],
        [0.99999708, 0.99999960, 0.99680414, 0.99721269]
    ]))

def test_rnn_nobias():
    rnn = neural.layer.RNN(4)
    rnn.setup(2, 1, neural.layer.Input(3))

    assert_equal(len(rnn.weights), 2)
    assert_equal(rnn.weights[0].get_value().shape, (3, 4))
    assert_equal(rnn.weights[1].get_value().shape, (4, 4))

    rnn.weights[0].set_value(np.asarray([
        [1, 1, 1, 1],
        [2, 2, 0, 0],
        [1, 1, 0, 1],
    ], dtype='float32'))

    rnn.weights[1].set_value(np.asarray([
        [1, 1, 1, 1],
        [2, 2, 0, 0],
        [2, 2, 2, 1],
        [1, 1, 1, 0],
    ], dtype='float32'))

    (b1, ) = rnn.scanner([
        [1, 1, 1],
        [2, 2, 2]
    ], [
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    assert(np.allclose(b1.eval(), [
        [0.98201379, 0.98201379, 0.73105858, 0.88079708],
        [0.99966465, 0.99966465, 0.88079708, 0.98201379]
    ]))

    (b2, ) = rnn.scanner([
        [1, 1, 1],
        [2, 2, 2]
    ], b1)

    assert(np.allclose(b2.eval(), [
        [0.99990757, 0.99990757, 0.98693836, 0.97617886],
        [0.99999892, 0.99999892, 0.99680597, 0.99721429]
    ]))

    (b2m, ) = rnn.scanner([
        [1, 1, 1],
        [2, 2, 2]
    ], b1, mask=np.asarray([[1], [0]], dtype='int8'))

    assert(np.allclose(b2m.eval(), [
        [0.98201379, 0.98201379, 0.73105858, 0.88079708],
        [0.99999892, 0.99999892, 0.99680597, 0.99721429]
    ]))

def test_lstm_bias():
    lstm = neural.layer.LSTM(4, bias=True)
    lstm.setup(2, 1, neural.layer.Input(3))

    assert_equal(len(lstm.weights), 3)
    assert_equal(lstm.weights[0].get_value().shape, (3, 4 * 4))
    assert_equal(lstm.weights[1].get_value().shape, (4, 4 * 4))
    assert_equal(lstm.weights[2].get_value().shape, (4 * 4,))

    W01 = np.asarray([
        [1, 1, 1, 1],
        [2, 2, 0, 0],
        [1, 1, 0, 1],
    ], dtype='float32')
    lstm.weights[0].set_value(np.tile(W01, (1, 4)))

    W11 = np.asarray([
        [1, 1, 1, 1],
        [2, 2, 0, 0],
        [2, 2, 2, 1],
        [1, 1, 1, 0],
    ], dtype='float32')
    lstm.weights[1].set_value(np.tile(W11, (1, 4)))

    Wb = np.asarray([-1, 1, 0, 0], dtype='float32')
    lstm.weights[2].set_value(np.tile(Wb, 4))

    (s1, b1) = lstm.scanner([
        [1, 1, 1],
        [2, 2, 2]
    ], [
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], [
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    assert(np.allclose(s1.eval(), [
        [0.90739747, 0.98665909, 0.53444665, 0.77580349],
        [0.99817873, 0.99975323, 0.77580349, 0.96435108]
    ]))
    assert(np.allclose(b1.eval(), [
        [0.67867792, 0.72355227, 0.46094678, 0.60314779],
        [0.73003464, 0.73091985, 0.60314779, 0.71097024]
    ]))

    (s2, b2) = lstm.scanner([
        [1, 1, 1],
        [2, 2, 2]
    ], s1, b1)

    assert(np.allclose(s2.eval(), [
        [1.90364485, 1.98613660, 1.43706036, 1.66232439],
        [1.99813383, 1.99974715, 1.74953353, 1.95012906]
    ]))
    assert(np.allclose(b2.eval(), [
        [0.86917968, 0.87917997, 0.77646605, 0.80566524],
        [0.88058782, 0.88076874, 0.84380403, 0.87125371]
    ]))

    (s2m, b2m) = lstm.scanner([
        [1, 1, 1],
        [2, 2, 2]
    ], s1, b1, mask=np.asarray([[1], [0]], dtype='int8'))

    assert(np.allclose(s2m.eval(), [
        [0.90739747, 0.98665909, 0.53444665, 0.77580349],
        [1.99813383, 1.99974715, 1.74953353, 1.95012906]
    ]))
    assert(np.allclose(b2m.eval(), [
        [0.67867792, 0.72355227, 0.46094678, 0.60314779],
        [0.88058782, 0.88076874, 0.84380403, 0.87125371]
    ]))

def test_lstm_nobias():
    lstm = neural.layer.LSTM(4)
    lstm.setup(2, 1, neural.layer.Input(3))

    assert_equal(len(lstm.weights), 2)
    assert_equal(lstm.weights[0].get_value().shape, (3, 4 * 4))
    assert_equal(lstm.weights[1].get_value().shape, (4, 4 * 4))

    W01 = np.asarray([
        [1, 1, 1, 1],
        [2, 2, 0, 0],
        [1, 1, 0, 1],
    ], dtype='float32')
    lstm.weights[0].set_value(np.tile(W01, (1, 4)))

    W11 = np.asarray([
        [1, 1, 1, 1],
        [2, 2, 0, 0],
        [2, 2, 2, 1],
        [1, 1, 1, 0],
    ], dtype='float32')
    lstm.weights[1].set_value(np.tile(W11, (1, 4)))

    (s1, b1) = lstm.scanner([
        [1, 1, 1],
        [2, 2, 2]
    ], [
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], [
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    assert(np.allclose(s1.eval(), [
        [0.96435108, 0.96435108, 0.53444665, 0.77580349],
        [0.99932941, 0.99932941, 0.77580349, 0.96435108]
    ]))
    assert(np.allclose(b1.eval(), [
        [0.71097024, 0.71097024, 0.46094678, 0.60314779],
        [0.73068160, 0.73068160, 0.60314779, 0.71097024]
    ]))

    (s2, b2) = lstm.scanner([
        [1, 1, 1],
        [2, 2, 2]
    ], s1, b1)

    assert(np.allclose(s2.eval(), [
        [1.96295198, 1.96295198, 1.43999274, 1.66573434],
        [1.99931289, 1.99931289, 1.74955030, 1.95013820]
    ]))
    assert(np.allclose(b2.eval(), [
        [0.87643815, 0.87643815, 0.77786746, 0.80716727],
        [0.88072007, 0.88072007, 0.84381131, 0.87125741]
    ]))

    (s2m, b2m) = lstm.scanner([
        [1, 1, 1],
        [2, 2, 2]
    ], s1, b1, mask=np.asarray([[1], [0]], dtype='int8'))

    assert(np.allclose(s2m.eval(), [
        [0.96435108, 0.96435108, 0.53444665, 0.77580349],
        [1.99931289, 1.99931289, 1.74955030, 1.95013820]
    ]))
    assert(np.allclose(b2m.eval(), [
        [0.71097024, 0.71097024, 0.46094678, 0.60314779],
        [0.88072007, 0.88072007, 0.84381131, 0.87125741]
    ]))
