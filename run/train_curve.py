it d
import run

import os
import os.path as path
import neural
import numpy as np

EPOCHS = 4000
TEST_SIZE = 200
TRAIN_SIZES = [100, 200, 400, 800, 1600, 3200, 6400, 12800]
Y_SHAPE = (TEST_SIZE, 6, 15)
OUTNAME = os.environ.get('OUTNAME')

if (OUTNAME is None): raise ValueError('No OUTNAME env')

thisdir = path.dirname(path.realpath(__file__))

def count_network_sequence(items, T=8, classes=5):
    # Create initial value
    X = np.random.uniform(0, classes, size=(items, 2, 2))
    X[:, 0, 0] = 0
    X[:, 0, 1] = 1
    X[:, 1, 1] = 0

    # Create targe by incrementing
    inc = np.tile(np.arange(0, T), (items, 1))
    t = np.mod(X[:, 1, 0][:, None] + inc, classes)
    t = np.floor(t)

    # add <EOS>
    t = t + 1
    t = np.hstack([t, np.zeros((items, 1))])

    # Normalize X
    X[:, 1, 0] = X[:, 1, 0] / classes

    return (X.astype('float32'), t.astype('int32'))

def create_model():
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

    return sutskever

def run_model(train_size, test_dateset):
    np.random.seed(42)

    print('')
    print('### Creating sutskever network, train size: %d ###' % (train_size, ))

    model = create_model()

    train_dataset = count_network_sequence(train_size)

    for i in range(0, EPOCHS):
        train_error = model.train(*train_dataset)
        print('  train: size %d, epoch %d, loss %f' % (train_size, i, train_error))

    return {
        'train_loss': model.test(*train_dataset),
        'test_loss': model.test(*test_dateset),
        'test_predict': model.predict(test_dateset[0])
    }

test_dataset = count_network_sequence(TEST_SIZE)

train_loss = np.zeros(len(TRAIN_SIZES))
test_loss = np.zeros(len(TRAIN_SIZES))
test_predict = np.zeros((len(TRAIN_SIZES), ) + Y_SHAPE)

for i, train_size in enumerate(TRAIN_SIZES):
    results = run_model(train_size, test_dataset)
    train_loss[i] = results['train_loss']
    test_loss[i] = results['test_loss']
    test_predict[i] = results['test_predict']

np.savez_compressed(
    path.join(thisdir, '..', 'outputs', OUTNAME + '.npz'),
    train_loss=train_loss,
    test_loss=test_loss,
    test_predict=test_predict,
    input=test_dataset[0],
    target=test_dataset[1],
    train_size=np.asarray(TRAIN_SIZES)
)
