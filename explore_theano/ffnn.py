
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
from datasets import generate_quadrant

size = [2, 10, 4]
eta = 0.4
momentum = 0.9
epochs = 400

# data input & output
x = T.matrix('x')
t = T.ivector('t')

# forward pass
W1 = theano.shared(np.random.randn(
    size[0], size[1]).astype('float32'),
    name="W1", borrow=True)
Wb1 = theano.shared(
    np.asarray(0, dtype='float32'),
    name="Wb1", borrow=True)
a1 = T.dot(x, W1) + Wb1
b1 = T.nnet.sigmoid(a1)

W2 = theano.shared(
    np.random.randn(size[1], size[2]).astype('float32'),
    name="W2", borrow=True)
Wb2 = theano.shared(
    np.asarray(0, dtype='float32'),
    name="Wb1", borrow=True)
a2 = T.dot(b1, W2) + Wb2

y = T.nnet.softmax(a2)

# training error
# Confim that categorical_crossentropy doesn't sum
L = T.mean(T.nnet.categorical_crossentropy(y, t))

# backward pass
(gW1, gWb1, gW2, gWb2) = T.grad(L, [W1, Wb1, W2, Wb2])
dW1 = theano.shared(
    np.zeros_like(W1.get_value(), dtype='float32'),
    name="dW1", borrow=True)
dWb1 = theano.shared(
    np.zeros_like(Wb1.get_value(), dtype='float32'),
    name="dWb1", borrow=True)
dW2 = theano.shared(
    np.zeros_like(W2.get_value(), dtype='float32'),
    name="dW1", borrow=True)
dWb2 = theano.shared(
    np.zeros_like(Wb2.get_value(), dtype='float32'),
    name="dWb2", borrow=True)

# Compile
# NOTE: all updates are made with the old values, thus the order of operation
# doesn't matter. To make momentum work without a delay as in
# http://stackoverflow.com/questions/28205589/the-update-order-of-theano-functions-update-list
# the update equation (dW1) is inserted into the `W1 = W1 + dW1` update.
train = theano.function(
    inputs=[x, t],
    outputs=L,
    updates=((dW1, - momentum * dW1 - eta * gW1), (W1, W1 - momentum * dW1 - eta * gW1),
             (dWb1, - momentum * dWb1 - eta * gWb1), (Wb1, Wb1 - momentum * dWb1 - eta * gWb1),
             (dW2, - momentum * dW2 - eta * gW2), (W2, W2 - momentum * dW2 - eta * gW2),
             (dWb2, - momentum * dWb2 - eta * gWb2), (Wb2, Wb2 - momentum * dWb2 - eta * gWb2))
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
print(W1.get_value(), Wb1.get_value())
print(W2.get_value(), Wb2.get_value())

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
