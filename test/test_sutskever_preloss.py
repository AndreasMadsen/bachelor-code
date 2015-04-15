
import test
from nose.tools import *

import numpy as np
import theano
import theano.tensor as T

import neural

e_0 = 0.00333333
e_1 = 0.99

def test_sutskever_preloss_yat():
    # S(P) = S(T) = L(P,T)
    y_1 = np.asarray([
        [0.1, 0.1, 0.2, 0.7],
        [0.1, 0.6, 0.5, 0.1],
        [0.8, 0.3, 0.3, 0.2]
    ])
    eosi_1 = 3
    t_1 = [2, 1, 1, 0]

    # S(P) < S(T) < L(P,T)
    y_2 = np.asarray([
        [0.6, 0.6, 0.6, 0.6],
        [0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2]
    ])
    eosi_2 = 0
    t_2 = [1, 0, 0, 0]

    # S(P) > S(T) < L(P,T)
    y_3 = np.asarray([
        [0.2, 0.2, 0.1, 0.8],
        [0.6, 0.7, 0.4, 0.1],
        [0.2, 0.1, 0.5, 0.1]
    ])
    eosi_3 = 3
    t_3 = [1, 0, 0, 0]

    # No <EOS>
    y_4 = np.asarray([
        [0.2, 0.2, 0.1, 0.2],
        [0.6, 0.7, 0.4, 0.6],
        [0.2, 0.1, 0.5, 0.2]
    ])
    eosi_4 = 3
    t_4 = [2, 2, 0, 0]

    # Create tensors
    y = T.tensor3('y')
    y.tag.test_value = np.asarray(
        [y_1, y_2, y_3, y_4], dtype='float32'
    )
    eosi = T.ivector('eosi')
    eosi.tag.test_value = np.asarray(
        [eosi_1, eosi_2, eosi_3, eosi_4], dtype='int32'
    )
    t = T.imatrix('t')
    t.tag.test_value = np.asarray(
        [t_1, t_2, t_3, t_4], dtype='int32'
    )

    # Run preloss transform
    sutskever = neural.network.Sutskever()
    (y_pad, t_pad) = sutskever._preloss(eosi, y, t)

    # Assert
    assert(np.allclose(y_pad[0, :, :].tag.test_value, [
        [0.1, 0.1, 0.2, 0.7],
        [0.1, 0.6, 0.5, 0.1],
        [0.8, 0.3, 0.3, 0.2]
    ]))
    assert(np.allclose(y_pad[1, :, :].tag.test_value, [
        [0.6, e_1, e_1, e_1],
        [0.2, e_0, e_0, e_0],
        [0.2, e_0, e_0, e_0]
    ]))
    assert(np.allclose(y_pad[2, :, :].tag.test_value, [
        [0.2, 0.2, 0.1, 0.8],
        [0.6, 0.7, 0.4, 0.1],
        [0.2, 0.1, 0.5, 0.1]
    ]))
    assert(np.allclose(y_pad[3, :, :].tag.test_value, [
        [0.2, 0.2, 0.1, 0.2],
        [0.6, 0.7, 0.4, 0.6],
        [0.2, 0.1, 0.5, 0.2]
    ]))

    assert(np.allclose(t_pad.tag.test_value, [
        [2, 1, 1, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [2, 2, 0, 0]
    ]))

def test_sutskever_preloss_yget():
    # S(P) = S(T) = L(P,T)
    y_1 = np.asarray([
        [0.1, 0.1, 0.2, 0.3, 0.7],
        [0.1, 0.6, 0.5, 0.3, 0.1],
        [0.8, 0.3, 0.3, 0.4, 0.2]
    ])
    eosi_1 = 4
    t_1 = [2, 1, 1, 0]

    # S(P) < S(T) < L(P,T)
    y_2 = np.asarray([
        [0.6, 0.6, 0.6, 0.6, 0.6],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2, 0.2]
    ])
    eosi_2 = 0
    t_2 = [1, 0, 0, 0]

    # Create tensors
    y = T.tensor3('y')
    y.tag.test_value = np.asarray(
        [y_1, y_2], dtype='float32'
    )
    eosi = T.ivector('eosi')
    eosi.tag.test_value = np.asarray(
        [eosi_1, eosi_2], dtype='int32'
    )
    t = T.imatrix('t')
    t.tag.test_value = np.asarray(
        [t_1, t_2], dtype='int32'
    )

    # Run preloss transform
    sutskever = neural.network.Sutskever()
    (y_pad, t_pad) = sutskever._preloss(eosi, y, t)

    # Assert
    assert(np.allclose(y_pad[0, :, :].tag.test_value, [
        [0.1, 0.1, 0.2, 0.3, 0.7],
        [0.1, 0.6, 0.5, 0.3, 0.1],
        [0.8, 0.3, 0.3, 0.4, 0.2]
    ]))
    assert(np.allclose(y_pad[1, :, :].tag.test_value, [
        [0.6, e_1, e_1, e_1, e_1],
        [0.2, e_0, e_0, e_0, e_0],
        [0.2, e_0, e_0, e_0, e_0]
    ]))

    assert(np.allclose(t_pad.tag.test_value, [
        [2, 1, 1, 0, 0],
        [1, 0, 0, 0, 0]
    ]))

def test_sutskever_preloss_ylet():
    # S(P) = S(T) = L(P,T)
    y_1 = np.asarray([
        [0.1, 0.1, 0.7],
        [0.1, 0.6, 0.1],
        [0.8, 0.3, 0.2]
    ])
    eosi_1 = 2
    t_1 = [2, 1, 1, 0]

    # S(P) < S(T) < L(P,T)
    y_2 = np.asarray([
        [0.6, 0.6, 0.6],
        [0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2]
    ])
    eosi_2 = 0
    t_2 = [1, 0, 0, 0]

    # Create tensors
    y = T.tensor3('y')
    y.tag.test_value = np.asarray(
        [y_1, y_2], dtype='float32'
    )
    eosi = T.ivector('eosi')
    eosi.tag.test_value = np.asarray(
        [eosi_1, eosi_2], dtype='int32'
    )
    t = T.imatrix('t')
    t.tag.test_value = np.asarray(
        [t_1, t_2], dtype='int32'
    )

    # Run preloss transform
    sutskever = neural.network.Sutskever()
    (y_pad, t_pad) = sutskever._preloss(eosi, y, t)

    # Assert
    assert(np.allclose(y_pad[0, :, :].tag.test_value, [
        [0.1, 0.1, 0.7, e_1],
        [0.1, 0.6, 0.1, e_0],
        [0.8, 0.3, 0.2, e_0]
    ]))
    assert(np.allclose(y_pad[1, :, :].tag.test_value, [
        [0.6, e_1, e_1, e_1],
        [0.2, e_0, e_0, e_0],
        [0.2, e_0, e_0, e_0]
    ]))

    assert(np.allclose(t_pad.tag.test_value, [
        [2, 1, 1, 0],
        [1, 0, 0, 0]
    ]))
