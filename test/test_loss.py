
import test
from nose.tools import *

import numpy as np
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
    t2 = np.asarray([2, 1, 0], dtype='int32')

    t = np.column_stack([t1.T, t2.T])
    y = np.log(np.dstack([y1, y2]))

    assert_equal(entropy.loss(y, t).eval(), 2.4401886463165283)

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
    t2 = np.asarray([2, 1, 0], dtype='int32')

    t = np.column_stack([t1.T, t2.T])
    y = np.dstack([y1, y2])

    assert_equal(entropy.loss(y, t).eval(), 2.4401886463165283)

test_naive_entropy()
