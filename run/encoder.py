
import run

import numpy as np

import dataset
import neural

data = dataset.encoder.mode(300)

encoder = neural.network.SutskeverEncoder(
    eta=0.05, momentum=0.2, verbose=True
)

# Setup layers
encoder.set_input(neural.layer.Input(data.data.shape[1]))
encoder.push_layer(neural.layer.LSTM(80))
encoder.push_layer(neural.layer.Softmax(data.n_classes, log=True))

# Setup loss function
encoder.set_loss(neural.loss.NaiveEntropy(time=False, log=True))

# Compile train, test and predict functions
encoder.compile()

results = run.simple_learn(encoder, data, 100, 500)
np.savez_compressed(run.output_file, **results)
