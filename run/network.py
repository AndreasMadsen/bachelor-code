
import run

import numpy as np

import dataset
import neural

data = dataset.network.copy(11 * 128)

sutskever = neural.network.Sutskever(
    max_output_size=data.target.shape[1], verbose=True
)

# Setup layers
sutskever.set_encoder_input(neural.layer.Input(10))
sutskever.push_encoder_layer(neural.layer.LSTM(40))
sutskever.push_encoder_layer(neural.layer.LSTM(20))

sutskever.set_decoder_input(neural.layer.Input(10))
sutskever.push_decoder_layer(neural.layer.LSTM(20))
sutskever.push_decoder_layer(neural.layer.LSTM(40))
sutskever.push_decoder_layer(neural.layer.Softmax(10))

# Setup loss function
sutskever.set_loss(neural.loss.NaiveEntropy())
sutskever.set_optimizer(neural.optimizer.RMSgrave())

# Compile train, test and predict functions
sutskever.compile()

results = run.minibatch_learn(sutskever, data, test_size=128, epochs=1000)
np.savez_compressed(run.output_file + '.npz', **results)

# show example
(b_enc, t) = (data.data[0:10, :], data.target[0:10, :])
y = sutskever.predict(b_enc)

print(np.argmax(y, axis=1))
print(t)
