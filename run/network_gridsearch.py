
import run

import json
import numpy as np

import dataset
import neural

data = dataset.network.copy(11 * 1280)

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
sutskever.set_optimizer(neural.optimizer.RMSgrave(clipping_stratagy='L2'))

# Compile train, test and predict functions
sutskever.compile()

search = neural.learn.GridSearch(sutskever, {
    "epochs": 20,
    "momentum": [0, 0.2, 0.9],
    "decay": [0.9, 0.95],
    "clip": [1, 5, 10, 50],
    "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.2]
}, test_size=1280, verbose=True)

results = search.run(data)

print(results)

with open(run.output_file + '.json', 'w') as outfile:
    json.dump(results, outfile, indent=4)
