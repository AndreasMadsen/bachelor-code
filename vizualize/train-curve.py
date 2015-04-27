
import os.path as path
import sys
import math

import numpy as np
import matplotlib.pyplot as plt

thisdir = path.dirname(path.realpath(__file__))

if (len(sys.argv) < 2):
    print('python3 train-curve.py output.npz')
    sys.exit(1)

output = np.load(path.realpath(sys.argv[1]))

test_predict = output['test_predict']
train_loss = output['train_loss']
test_loss = output['test_loss']

train_sizes = output['train_size']
runs = train_sizes.size
dims = test_predict.shape[2]
name = path.basename(sys.argv[1])

# Pad target with <EOS> so it fits the max length
target = np.zeros((test_predict.shape[1], test_predict.shape[3]), dtype='int32')
target[:, :output['target'].shape[1]] = output['target']

# Calculate miss classificuation error
def set_eos_padding(class_predict):
    class_predict = np.copy(class_predict)
    eosi = np.argmin(class_predict, axis=1)
    for obs, obs_eosi in enumerate(eosi):
        class_predict[obs, obs_eosi:] = 0
    return class_predict

miss_error = np.zeros(runs)

for run, run_predict in enumerate(test_predict):
    class_predict = set_eos_padding(np.argmax(run_predict, axis=1))
    miss_error[run] = np.mean(class_predict != target)

# Plot
plt.figure(figsize=(8, 8))
plt.suptitle(name)

plt.subplot(2, 1, 1)
plt.plot(train_sizes, miss_error, color='SteelBlue', label='test')
plt.axhline((dims - 1) / dims, color='gray')
plt.legend()
plt.ylabel('misclassification error [%]')
plt.xlabel('train size')
plt.ylim(0, 1)

plt.subplot(2, 1, 2)
plt.plot(train_sizes, train_loss, color='IndianRed', label='train')
plt.plot(train_sizes, test_loss, color='SteelBlue', label='test')
plt.axhline(-math.log(1 / dims), color='gray')
plt.legend()
plt.ylabel('loss [entropy]')
plt.xlabel('train size')
plt.ylim(0, 8)

plt.savefig(path.join(thisdir, '..', 'outputs', name[:-4] + '.png'))
plt.show()
