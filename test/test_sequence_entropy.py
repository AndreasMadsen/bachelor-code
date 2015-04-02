
import test

import numpy as np
import theano
import theano.tensor as T

#
# Entropy Input
#

# S(P) = S(T) = L(P,T)
a1_y = np.asarray([
    [0.1, 0.1, 0.2, 0.7],
    [0.1, 0.6, 0.5, 0.1],
    [0.8, 0.3, 0.3, 0.2]
], dtype='float64')
a1_yt = 3

a1_t = [2, 1, 1, 0]


# S(P) < S(T) < L(P,T)
b1_y = np.asarray([
    [0.6, 0.6, 0.6, 0.6],
    [0.2, 0.2, 0.2, 0.2],
    [0.2, 0.2, 0.2, 0.2]
], dtype='float64')
b1_yt = 0

b1_t = [1, 0, 0, 0]

# S(P) > S(T) < L(P,T)
c1_y = np.asarray([
    [0.2, 0.2, 0.1, 0.8],
    [0.6, 0.7, 0.4, 0.1],
    [0.2, 0.1, 0.5, 0.1]
], dtype='float64')
c1_yt = 3

c1_t = [1, 0, 0, 0]

# No <EOS>
d1_y = np.asarray([
    [0.2, 0.2, 0.1, 0.2],
    [0.6, 0.7, 0.4, 0.6],
    [0.2, 0.1, 0.5, 0.2]
], dtype='float64')
d1_yt = 3

d1_t = [2, 2, 0, 0]

#
# Entropy precalc
#

# S(P) = S(T) = L(P,T)
a2_y = np.asarray([
    [0.1, 0.1, 0.2, 0.7],
    [0.1, 0.6, 0.5, 0.1],
    [0.8, 0.3, 0.3, 0.2]
], dtype='float64')

a2_t = [2, 1, 0, 0]

# S(P) < S(T) < L(P,T)
b2_y = np.asarray([
    #      Fill, Igno, Igno
    [0.60, 0.33, 1.00, 1.00],
    [0.20, 0.33, 0.00, 0.00],
    [0.20, 0.33, 0.00, 0.00]
], dtype='float64')
# from max(S(T), S(P)) to L(T, P) set 1
# Use a selection matrix b_y[:, matrix]
# Remeber <EOF> is repeated, by scan

b2_t = [1, 0, 0, 0]  # <EOF> extend


# S(P) > S(T) < L(P,T)
c2_y = np.asarray([
    [0.2, 0.2, 0.1, 0.8],
    [0.6, 0.7, 0.4, 0.1],
    [0.2, 0.1, 0.5, 0.1]
], dtype='float64')

c2_t = [1, 0, 0, 0]  # <EOF> extend

# No <EOS>
d2_y = np.asarray([
    [0.2, 0.2, 0.1, 0.2],
    [0.6, 0.7, 0.4, 0.6],
    [0.2, 0.1, 0.5, 0.2]
], dtype='float64')

d2_t = [2, 2, 0, 0]

#
# Testing
#

# TODO: consider if L(P) < L(T)
# TODO: consider if L(T) < L(P)

# Setup the batch
y = T.tensor3('y')
y.tag.test_value = np.asarray([
    a1_y, b1_y, c1_y, d1_y
], dtype='float32')

yt = T.ivector('yt')
yt.tag.test_value = np.asarray([
    a1_yt, b1_yt, c1_yt, d1_yt
], dtype='int32')

t = T.imatrix('t')
t.tag.test_value = np.asarray([
    a1_t, b1_t, c1_t, d1_t
], dtype='int32')

print(yt.tag.test_value)

def scanner(y_i, yt_i, t_i, dims, time):
    # Get length of y seqence including the first <EOS>
    # FIXME: consider if there are no <EOS> elements
    yend = yt_i + 1

    # Get length of t seqence including the first <EOS>
    tend = T.nonzero(T.eq(t_i, 0))[0][0] + 1

    # Create a new y sequence with T elements
    # TODO: this can be optimized, being clever with subtensor and ones
    y2_i = T.zeros((dims, time), dtype='float32')
    # Keep the actual y elements
    y2_i = T.set_subtensor(y2_i[:, :yend], y_i[:, :yend])
    # Fill in missing y2 elements with an even distribution.
    #   If yend >= tend, then this won't do anything
    y2_i = T.set_subtensor(y2_i[:, yend:tend], 1.0 / dims)
    # Add ignore padding to y2 for the remaining elments
    y2_i = T.set_subtensor(y2_i[:, T.max([yend, tend]):], 1.0)

    # Createa a new t seqnece with T elements
    t2_i = T.zeros((time,), dtype='int32')
    # TODO: this can be optimized, being clever with subtensor
    # Keep the actual t elements
    t2_i = T.set_subtensor(t2_i[:tend], t_i[:tend])
    # Add <EOS> padding to t2 for the remaining elments, the y2 padding will ignore this
    t2_i = T.set_subtensor(t2_i[tend:], 0)

    return [y2_i, t2_i]

(y2, t2), _ = theano.scan(
    fn=scanner,
    sequences=[y, yt, t],
    outputs_info=[None, None],
    non_sequences=[
        y.shape[1],
        T.max([y.shape[2], t.shape[1]])
    ]
)

t3 = t2.ravel()
y3 = y2.transpose(0, 2, 1).reshape((y2.shape[2] * y2.shape[0], y2.shape[1]))
L = T.mean(T.nnet.categorical_crossentropy(y3, t3))

T.grad(L, [y])

train = theano.function(
    inputs=[y, yt, t],
    outputs=L
)

print(train(y.tag.test_value, yt.tag.test_value, t.tag.test_value))

print(y2.tag.test_value[0, :, :])
print(t2.tag.test_value[0, :])
print(y2.tag.test_value[1, :, :])
print(t2.tag.test_value[1, :])
print(y2.tag.test_value[2, :, :])
print(t2.tag.test_value[2, :])
print(y2.tag.test_value[3, :, :])
print(t2.tag.test_value[3, :])
