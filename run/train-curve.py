
import run

import numpy as np

import dataset
import neural

EPOCHS = 20
TEST_SIZE = 200
TRAIN_SIZES = [100, 200]

sutskever = neural.network.Sutskever(eta=0.2, momentum=0.3, max_output_size=15, verbose=True)

# Setup layers
sutskever.set_encoder_input(neural.layer.Input(2))
sutskever.push_encoder_layer(neural.layer.LSTM(2))

sutskever.set_decoder_input(neural.layer.Input(6))
sutskever.push_decoder_layer(neural.layer.LSTM(2))
sutskever.push_decoder_layer(neural.layer.LSTM(80))
sutskever.push_decoder_layer(neural.layer.Softmax(6, log=True))

# Setup loss function
sutskever.set_loss(neural.loss.NaiveEntropy(log=True))

# Compile train, test and predict functions
sutskever.compile()

results = {
    'train_loss': np.zeros(len(TRAIN_SIZES)),
    'test_loss': np.zeros(len(TRAIN_SIZES)),

    'train_miss': np.zeros(len(TRAIN_SIZES)),
    'test_miss': np.zeros(len(TRAIN_SIZES)),

    'train_sizes': np.asarray(TRAIN_SIZES),
    'n_classes': dataset.network.count(10).n_classes
}

for i, train_size in enumerate(TRAIN_SIZES):
    np.random.seed(42)
    sutskever.reset_weights()
    data = dataset.network.count(train_size + TEST_SIZE)

    run_results = run.batch_learn(sutskever, data, test_size=TEST_SIZE, epochs=EPOCHS)

    results['train_loss'][i] = run_results['train_loss'][-1]
    results['test_loss'][i] = run_results['test_loss'][-1]
    results['train_miss'][i] = run_results['train_miss'][-1]
    results['test_miss'][i] = run_results['test_miss'][-1]

np.savez_compressed(run.output_file, **results)
