
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
from datasets import generate_quadrant

# https://gist.github.com/AndreasMadsen/abd0624f2e88057bcc19#file-rnn-py-L37
# https://groups.google.com/forum/#!msg/theano-users/93gGzBkyswM/0oppbNxrm5IJ
# https://github.com/lisa-lab/DeepLearningTutorials/blob/master/code/lstm.py#L193

# Step one, online gradient decent
# Step two, batch gradient decent (transpose, perhaps alloc)

size = [2, 10, 4]
eta = 0.4
momentum = 0.9
epochs = 400

# data input & output
x = T.matrix('x')
t = T.ivector('t')

# forward pass
W01 = theano.shared(
    np.random.randn(size[0], size[1]).astype('float32'),
    name="W01", borrow=True)
W11 = theano.shared(
    np.random.randn(size[1], size[1]).astype('float32'),
    name="W11", borrow=True)
a1 = T.dot(x, W01)
b1 = T.nnet.sigmoid(a1)

W12 = theano.shared(
    np.random.randn(size[1], size[2]).astype('float32'),
    name="W12", borrow=True)
a2 = T.dot(b1, W12)

y = T.nnet.softmax(a2)

# training error
# `categorical_crossentropy` returns a vector with the entropy for each value
L = T.mean(T.nnet.categorical_crossentropy(y, t))

# backward pass
(gW01, gW12) = T.grad(L, [W01, W12])
dW01 = theano.shared(
    np.zeros_like(W01.get_value(), dtype='float32'),
    name="dW1", borrow=True)
dW12 = theano.shared(
    np.zeros_like(W12.get_value(), dtype='float32'),
    name="dW1", borrow=True)

# Compile
# NOTE: all updates are made with the old values, thus the order of operation
# doesn't matter. To make momentum work without a delay as in
# http://stackoverflow.com/questions/28205589/the-update-order-of-theano-functions-update-list
# the update equation (dW1) is inserted into the `W1 = W1 + dW1` update.
train = theano.function(
    inputs=[x, t],
    outputs=L,
    updates=((dW01, - momentum * dW01 - eta * gW01), (W01, W01 - momentum * dW01 - eta * gW01),
             (dW12, - momentum * dW12 - eta * gW12), (W12, W12 - momentum * dW12 - eta * gW12))
)

error = theano.function(inputs=[x, t], outputs=L)
predict = theano.function(inputs=[x], outputs=y)

# Generate dataset
(train_X, train_t) = generate_quadrant(1000)
(test_X, test_t) = generate_quadrant(300)

train_error = np.zeros(epochs)
test_error = np.zeros(epochs)

for epoch in range(0, epochs):
    train_error[epoch] = train(train_X, train_t)
    test_error[epoch] = error(test_X, test_t)
print(W01.get_value())
print(W12.get_value())

predict_y = np.argmax(predict(test_X), axis=1)


plt.subplot(2, 1, 1)
plt.plot(np.arange(0, epochs), train_error, label='train', alpha=0.5)
plt.plot(np.arange(0, epochs), test_error, label='test', alpha=0.5)
plt.legend()
plt.ylabel('loss')

plt.subplot(2, 1, 2)
colors = np.asarray(["#ca0020", "#f4a582", "#92c5de", "#0571b0"])
plt.scatter(test_X[:, 0], test_X[:, 1], c=colors[predict_y], lw=0)
plt.axhline(y=0, xmin=-1, xmax=1, color="gray")
plt.axvline(x=0, ymin=-1, ymax=1, color="gray")
plt.xlim([-1, 1])
plt.ylim([-1, 1])

plt.show()
