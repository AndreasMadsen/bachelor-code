
import run

import numpy as np

import dataset
import neural

data = dataset.encoder.copy(11 * 128)

encoder = neural.network.SutskeverEncoder(verbose=True, regression=True)

# Setup layers
encoder.set_input(neural.layer.Input(data.data.shape[1]))
encoder.push_layer(neural.layer.LSTM(40))
encoder.push_layer(neural.layer.LSTM(data.target.shape[1]))

# Setup loss and optimizer
encoder.set_loss(neural.loss.MeanSquaredError(time=False))
encoder.set_optimizer(neural.optimizer.RMSgrave(clipping_stratagy='L2'))

# Compile train, test and predict functions
encoder.compile()

results = run.minibatch_learn(encoder, data, test_size=128, epochs=300, regression=True, learning_rate=0.001)
np.savez_compressed(run.output_file + '.npz', **results)
