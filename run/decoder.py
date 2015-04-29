
import run

import numpy as np

import dataset
import neural

data = dataset.decoder.count(300)

decoder = neural.network.SutskeverDecoder(
    eta=0.2, momentum=0.3, maxlength=9, verbose=True
)

# Setup layers
decoder.set_input(neural.layer.Input(6))  # Should match output
decoder.push_layer(neural.layer.LSTM(1))  # Should match b_enc input
decoder.push_layer(neural.layer.LSTM(80))
decoder.push_layer(neural.layer.Softmax(6, log=True))

# Setup loss function
decoder.set_loss(neural.loss.NaiveEntropy(log=True))

# Compile train, test and predict functions
decoder.compile()

def missclassification(model, test_dataset):
    (data, target) = test_dataset
    return np.mean(np.argmax(model.predict(data), axis=1) != target)

test_dataset = (data.data[0:100, :], data.target[0:100])
train_dataset = (data.data[100:300, :], data.target[100:300])

results = run.simple_learn(decoder, data, 100, 500, missclassification)

np.savez_compressed(run.output_file, **results)
