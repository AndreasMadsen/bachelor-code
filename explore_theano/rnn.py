
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
eta = 0.4
momentum = 0.9
epochs = 100

# data input & output
x = T.matrix('x')
t = T.ivector('t')

test_value = generate_accumulated(1)
x.tag.test_value = test_value[0][0].T
t.tag.test_value = test_value[1][0]

# forward pass
W01 = theano.shared(
    np.random.randn(size[0], size[1]).astype('float32'),
    name="W01", borrow=True)
W11 = theano.shared(
    np.random.randn(size[1], size[1]).astype('float32'),
    name="W11", borrow=True)
W12 = theano.shared(
    np.random.randn(size[1], size[2]).astype('float32'),
    name="W12", borrow=True)


def scanner(x_t, b1_tm1, W01, W11, W12):
    a1_t = T.dot(x_t, W01) + T.dot(b1_tm1, W11)
    b1_t = T.nnet.sigmoid(a1_t)

    a2_t = T.dot(b1_t, W12)
    y_t = T.nnet.softmax(a2_t)[0]  # softmax always returns a matrix -- wired

    return (b1_t, y_t)

# b1 and y are matrices with each result at time $t$, in row $t$
(b1, y), _ = theano.scan(
    fn=scanner,
    sequences=x,  # iterate over rows (time)
    outputs_info=[
        T.zeros(size[1], dtype='float32'),  # b1_tm1
        None  # y_t
    ],
    non_sequences=[
        W01, W11, W12
    ]
)

# training error
# `categorical_crossentropy` returns a vector with the entropy for each value
L = T.mean(T.nnet.categorical_crossentropy(y, t))

# backward pass
(gW01, gW11, gW12) = T.grad(L, [W01, W11, W12])

# Compile
# NOTE: all updates are made with the old values, thus the order of operation
# doesn't matter. To make momentum work without a delay as in
# http://stackoverflow.com/questions/28205589/the-update-order-of-theano-functions-update-list
# the update equation (dW1) is inserted into the `W1 = W1 + dW1` update.
train = theano.function(
    inputs=[x, t],
    outputs=L,
    updates=((W01, W01 - eta * gW01),
             (W11, W11 - eta * gW11),
             (W12, W12 - eta * gW12))
)

error = theano.function(inputs=[x, t], outputs=L)
predict = theano.function(inputs=[x], outputs=y)

# Generate dataset
(train_X, train_t) = generate_accumulated(500)
(test_X, test_t) = generate_accumulated(100)

train_error = np.zeros(epochs)
test_error = np.zeros(epochs)

for epoch in range(0, epochs):
    # NOTE: not randomize

    epoch_train_error = np.zeros_like(train_t)
    for i, obs_X, obs_t in zip(range(0, train_t.shape[0]), train_X, train_t):
        epoch_train_error[i] = train(obs_X.T, obs_t)
    train_error[epoch] = np.mean(epoch_train_error)

    epoch_test_error = np.zeros_like(test_t)
    for i, obs_X, obs_t in zip(range(0, test_t.shape[0]), test_X, test_t):
        epoch_test_error[i] = error(obs_X.T, obs_t)
    test_error[epoch] = np.mean(epoch_test_error)

predict_y = np.zeros((test_t.shape[0], test_t.shape[1], size[2]))
for i, obs_X, obs_t in zip(range(0, test_t.shape[0]), test_X, test_t):
    predict_y[i, :, :] = predict(obs_X.T)

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
