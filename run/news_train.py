
import run

import datetime
import numpy as np

import dataset
import neural

data = dataset.news.letters()
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

# Compile train, test and predict functions
sutskever.compile()

results = run.minibatch_learn(sutskever, data, test_size=1000,
                              max_time=datetime.timedelta(hours=23),
                              learning_rate=0.07, momentum=0.2)

np.savez_compressed(run.output_file + '.epoch.npz', **results)
np.savez_compressed(run.output_file + '.weights.npz', *sutskever.get_weights())

print('completed')
