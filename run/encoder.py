
import run

import numpy as np

import dataset
import neural

data = dataset.encoder.mode(300)

encoder = neural.network.SutskeverEncoder(verbose=True)

# Setup layers
encoder.set_input(neural.layer.Input(data.data.shape[1]))
encoder.push_layer(neural.layer.LSTM(80))
encoder.push_layer(neural.layer.Softmax(data.n_classes, log=True))

# Setup loss function
encoder.set_loss(neural.loss.NaiveEntropy(time=False, log=True))

# Compile train, test and predict functions
encoder.compile()

results = run.batch_learn(encoder, data, test_size=100, epochs=500,
                          learning_rate=0.05, momentum=0.2)
np.savez_compressed(run.output_file, **results)
