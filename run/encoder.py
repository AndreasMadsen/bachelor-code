
import run

import numpy as np

import dataset
import neural

data = dataset.encoder.mode(1280)

encoder = neural.network.SutskeverEncoder(verbose=True)

# Setup layers
encoder.set_input(neural.layer.Input(data.data.shape[1]))
encoder.push_layer(neural.layer.LSTM(80))
encoder.push_layer(neural.layer.Softmax(data.n_classes))

# Setup loss function
encoder.set_loss(neural.loss.NaiveEntropy(time=False))

# Compile train, test and predict functions
encoder.compile()

results = run.minibatch_learn(encoder, data, test_size=128, epochs=300,
                          learning_rate=0.07, momentum=0.2)
np.savez_compressed(run.output_file + '.npz', **results)
