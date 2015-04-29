
import run

import numpy as np

import dataset
import neural

data = dataset.encoder.mode(300)

encoder = neural.network.SutskeverEncoder(
    eta=0.05, momentum=0.2, verbose=True
)

# Setup theano tap.test_value
encoder.test_value(data.data[0:10, :], data.target[0:10])

# Setup layers
encoder.set_input(neural.layer.Input(data.data.shape[1]))
encoder.push_layer(neural.layer.LSTM(80))
encoder.push_layer(neural.layer.Softmax(data.n_classes, log=True))

# Setup loss function
encoder.set_loss(neural.loss.NaiveEntropy(time=False, log=True))

# Compile train, test and predict functions
encoder.compile()

def missclassification(model, test_dataset):
    (data, target) = test_dataset
    return np.mean(np.argmax(model.predict(data), axis=1) != target)

results = run.simple_learn(encoder, data, 500, missclassification)

np.savez_compressed(run.output_file, **results)
