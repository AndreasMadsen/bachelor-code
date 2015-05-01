
import run

import numpy as np

import dataset
import neural

data = dataset.decoder.memorize(300)

decoder = neural.network.SutskeverDecoder(
    eta=0.1, momentum=0.0, maxlength=data.target.shape[1], verbose=True
)

# Setup layers
decoder.set_input(neural.layer.Input(data.n_classes))  # Should match output
decoder.push_layer(neural.layer.LSTM(1))  # Should match b_enc input
decoder.push_layer(neural.layer.LSTM(80))
decoder.push_layer(neural.layer.Softmax(data.n_classes))

# Setup loss function
decoder.set_loss(neural.loss.NaiveEntropy())

# Compile train, test and predict functions
decoder.compile()

results = run.simple_learn(decoder, data, 100, 1000)
np.savez_compressed(run.output_file, **results)

# show example
(b_enc, t) = (data.data[0:10, :], data.target[0:10, :])
y = decoder.predict(b_enc)

print(np.argmax(y, axis=1))
print(t)
