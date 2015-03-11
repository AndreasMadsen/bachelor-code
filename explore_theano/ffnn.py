
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T


def generate_dataset(items):
    X = np.random.uniform(low=-1, high=1, size=(items, 2)).astype('float32')

    t = np.zeros(items, dtype='int32')
    t += np.all([X[:, 0] < 0, X[:, 1] >= 0], axis=0) * 1
    t += np.all([X[:, 0] < 0, X[:, 1] < 0], axis=0) * 2
    t += np.all([X[:, 0] >= 0, X[:, 1] < 0], axis=0) * 3
    return (X, t)

size = [2, 10, 4]
eta = 0.1

# data input & output
x = T.matrix('x')
t = T.ivector('t')

# forward pass
# TODO: add another hidden layer and bias unit
W1 = theano.shared(np.random.randn(
    size[0], size[1]).astype('float32'),
    name="W1", borrow=True)
a1 = T.dot(x, W1)
b1 = T.nnet.sigmoid(a1)

W2 = theano.shared(
    np.random.randn(size[1], size[2]).astype('float32'),
    name="W2", borrow=True)
a2 = T.dot(b1, W2)

y = T.nnet.softmax(a2)

# training error
# Confim that categorical_crossentropy doesn't sum
L = T.sum(T.nnet.categorical_crossentropy(y, t))

# backward pass
(gW1, gW2) = T.grad(L, [W1, W2])

# Compile
# TODO: add momentum
train = theano.function(
    inputs=[x, t],
    outputs=[],
    updates=((W1, W1 - eta * gW1), (W2, W2 - eta * gW2)))

predict = theano.function(inputs=[x], outputs=y)

# Generate dataset
(train_X, train_t) = generate_dataset(1000)
(test_X, test_t) = generate_dataset(300)

for epoch in range(0, 10):
    train(train_X, train_t)
print(W1.get_value())
print(W2.get_value())

predict_y = np.argmax(predict(test_X), axis=1)
print(predict_y)
colors = np.asarray(["#ca0020", "#f4a582", "#92c5de", "#0571b0"])
plt.scatter(test_X[:, 0], test_X[:, 1], c=colors[predict_y], lw=0)
plt.axhline(y=0, xmin=-1, xmax=1, color="gray")
plt.axvline(x=0, ymin=-1, ymax=1, color="gray")
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.show()
