
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
from datasets import generate_accumulated

# compute_test_value is 'off' by default, meaning this feature is inactive
theano.config.compute_test_value = 'warn'  # Use 'warn' to activate this feature

# https://gist.github.com/AndreasMadsen/abd0624f2e88057bcc19#file-rnn-py-L37
# https://groups.google.com/forum/#!msg/theano-users/93gGzBkyswM/0oppbNxrm5IJ
# https://github.com/lisa-lab/DeepLearningTutorials/blob/master/code/lstm.py#L193

# Step one, online gradient decent
# Step two, batch gradient decent (transpose, perhaps alloc)

size = [2, 100, 2]
eta = 0.01
momentum = 0.1
epochs = 200

# data input & output
x = T.tensor3('x')
t = T.imatrix('t')

test_value = generate_accumulated(10)
x.tag.test_value = test_value[0]
t.tag.test_value = test_value[1]

#
# forward pass
#
W01 = theano.shared(
    np.random.randn(size[0], size[1]).astype('float32'),
    name="W01", borrow=True)
W11 = theano.shared(
    np.random.randn(size[1], size[1]).astype('float32'),
    name="W11", borrow=True)
W12 = theano.shared(
    np.random.randn(size[1], size[2]).astype('float32'),
    name="W12", borrow=True)

def forward_scanner(x_t, b1_tm1, W01, W11, W12):
    a1_t = T.dot(x_t, W01) + T.dot(b1_tm1, W11)
    b1_t = T.nnet.sigmoid(a1_t)

    a2_t = T.dot(b1_t, W12)
    y_t = T.nnet.softmax(a2_t)

    return (b1_t, y_t)

# because scan assmes the iterable is the first tensor dimension, x is
# transposed into (time, obs, dims).
# When done $b1$ and $y$ will be tensors with the shape (time, obs, dims)
# this is then transposed back to its original format
(b1, y), _ = theano.scan(
    fn=forward_scanner,
    sequences=x.transpose(2, 0, 1),  # iterate (time), row (observations), col (dims)
    outputs_info=[
        T.zeros((x.shape[0], size[1]), dtype='float32'),  # b1_tm1 (ops x dims)
        None  # y_t
    ],
    non_sequences=[
        W01, W11, W12
    ]
)
y = y.transpose(1, 2, 0)  # transpose back to (obs, dims, time)

#
# training error
#
def error_scanner(y_t, t_t):
    # `categorical_crossentropy` returns a vector with the entropy for each value
    return T.mean(T.nnet.categorical_crossentropy(y_t, t_t))

# Because crossentropy assumes a matrix (y) and y is a tensor, scan along the
# time axis. t have the format (obs, time).
# The return value is a vector of the mean crossentropy for each time step,
# mean that to create a single value.
def crossentropy(y, t):
    """
    Calculates crossentropy for 3d tensor y and index matrix t

    Because nnet.categorical_crossentropy only takes y as matrix, y and t
    is reshaped intro the matrix/vector format that nnet.categorical_crossentropy
    expects.
    """
    t = t.ravel()
    y = y.transpose(0, 2, 1).reshape((y.shape[2] * y.shape[0], y.shape[1]))
    return T.mean(T.nnet.categorical_crossentropy(y, t))

L = crossentropy(y, t)

#
# backward pass
#
(gW01, gW11, gW12) = T.grad(L, [W01, W11, W12])

def momentum_gradient_decent(gW, W):
    dW_tm1 = theano.shared(
        np.zeros_like(W.get_value(), dtype='float32'),
        name="dW" + W.name, borrow=True)

    dW = - momentum * dW_tm1 - eta * gW
    return ((dW_tm1, dW), (W, W + dW))

# Compile
# NOTE: all updates are made with the old values, thus the order of operation
# doesn't matter. To make momentum work without a delay as in
# http://stackoverflow.com/questions/28205589/the-update-order-of-theano-functions-update-list
# the update equation (dW1) is inserted into the `W1 = W1 + dW1` update.
train = theano.function(
    inputs=[x, t],
    outputs=L,
    updates=(
        momentum_gradient_decent(gW01, W01) +
        momentum_gradient_decent(gW11, W11) +
        momentum_gradient_decent(gW12, W12)
    )
)

error = theano.function(inputs=[x, t], outputs=L)
predict = theano.function(inputs=[x], outputs=y)

# Generate dataset
(train_X, train_t) = generate_accumulated(500)
(test_X, test_t) = generate_accumulated(100)

train_error = np.zeros(epochs)
test_error = np.zeros(epochs)

for epoch in range(0, epochs):
    train_error[epoch] = train(train_X, train_t)
    test_error[epoch] = train(test_X, test_t)

predict_y = predict(test_X)

print(predict_y[1:10, :, :])
print(test_t[1:10, :])

print('last epoch')
print('  train error: %f' % (train_error[-1]))
print('  test error: %f' % (test_error[-1]))

# NOTE: naïve mean guess should give 3/8
plt.plot(np.arange(0, epochs), train_error, label='train', alpha=0.5)
plt.plot(np.arange(0, epochs), test_error, label='test', alpha=0.5)
plt.legend()
plt.ylabel('loss')
plt.show()
