
import test
from nose.tools import *

import numpy as np
import theano
import theano.tensor as T
import neural

def test_naive_entropy_log():
    entropy = neural.loss.NaiveEntropy(log=True)
    entropy.setup(2)

    y1 = np.asarray([
        [0.11245721, 0.83095266, 0.01521942, 0.04137069],
        [0.11840511, 0.87490203, 0.00079780, 0.00589504],
        [0.11840511, 0.87490203, 0.00079780, 0.00589504]
    ], dtype='float32')
    t1 = np.asarray([0, 1, 2], dtype='int32')
    y2 = np.asarray([
        [0.45764028, 0.45764028, 0.02278457, 0.06193488],
        [0.49485490, 0.49485490, 0.00122662, 0.00906358],
        [0.49485490, 0.49485490, 0.00122662, 0.00906358]
    ], dtype='float32')
    t2 = np.asarray([2, 1, 3], dtype='int32')

    t = T.imatrix('t')
    t.tag.test_value = np.column_stack([t1, t2])
    y = T.tensor3('y')
    y.tag.test_value = np.log(np.dstack([y1, y2]))

    assert_equal(entropy.loss(y, t).tag.test_value, 3.1068553924560547)

def test_naive_entropy():
    entropy = neural.loss.NaiveEntropy()
    entropy.setup(2)

    y1 = np.asarray([
        [0.11245721, 0.83095266, 0.01521942, 0.04137069],
        [0.11840511, 0.87490203, 0.00079780, 0.00589504],
        [0.11840511, 0.87490203, 0.00079780, 0.00589504]
    ], dtype='float32')
    t1 = np.asarray([0, 1, 2], dtype='int32')
    y2 = np.asarray([
        [0.45764028, 0.45764028, 0.02278457, 0.06193488],
        [0.49485490, 0.49485490, 0.00122662, 0.00906358],
        [0.49485490, 0.49485490, 0.00122662, 0.00906358]
    ], dtype='float32')
    t2 = np.asarray([2, 1, 3], dtype='int32')

    t = T.imatrix('t')
    t.tag.test_value = np.column_stack([t1, t2])
    y = T.tensor3('y')
    y.tag.test_value = np.dstack([y1, y2])

    assert_equal(entropy.loss(y, t).tag.test_value, 3.1068315505981445)
