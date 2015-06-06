
import plot

import os.path as path
import sys

import datetime
import numpy as np

import dataset
import neural

data = dataset.news.letters(10).range(0, 4)
weights = np.load(path.realpath(sys.argv[1]))
weights = [weights['arr_' + str(file)] for file in range(0, len(weights.files))]

latent_size = 100
print('dataset build')

sutskever = neural.network.Sutskever(
    max_output_size=data.max_output_size, verbose=True, indexed_input=True
)

# Setup layers
sutskever.set_encoder_input(neural.layer.Input(data.n_classes, indexed=True))
sutskever.push_encoder_layer(neural.layer.LSTM(200))
sutskever.push_encoder_layer(neural.layer.LSTM(latent_size))

sutskever.set_decoder_input(neural.layer.Input(data.n_classes))
sutskever.push_decoder_layer(neural.layer.LSTM(latent_size))
sutskever.push_decoder_layer(neural.layer.LSTM(200))
sutskever.push_decoder_layer(neural.layer.Softmax(data.n_classes))

# Setup loss function
sutskever.set_loss(neural.loss.NaiveEntropy())
sutskever.set_optimizer(neural.optimizer.RMSgrave())

# Load weights
sutskever.set_weights(weights)

# Compile train, test and predict functions
sutskever.compile()

(x, t) = data.astuple()
predict = sutskever.predict(x, max_output_size=t.shape[1])
output = np.argmax(predict, axis=1)

def missclassification(model, data):
    (x, t) = data.astuple()
    if (len(t.shape) > 1):
        prediction = model.predict(x, max_output_size=t.shape[1])
    else:
        prediction = model.predict(x)
    print(np.argmax(prediction, axis=1) != t)
    return np.mean(np.argmax(prediction, axis=1) != t)

print('missclassication rate: %f' % missclassification(sutskever, data))

def code2str(code):
    return ''.join([dataset.news.unique_chars[c] for c in code])

for y_i, t_i in zip(output, t):
    print('')
    print('predict:', code2str(y_i))
    print(' target:', code2str(t_i))
