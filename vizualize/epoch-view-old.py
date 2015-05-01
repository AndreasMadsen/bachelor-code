
import os.path as path
import sys
import math

import numpy as np
import matplotlib.pyplot as plt

thisdir = path.dirname(path.realpath(__file__))

if (len(sys.argv) < 2):
    print('python3 epoch-view.py output.npz')
    sys.exit(1)

output = np.load(path.realpath(sys.argv[1]))

predict = output['predict']
train = output['train']
test = output['test']

epochs = predict.shape[0]
dims = predict.shape[2]
name = path.basename(sys.argv[1])

# Pad target with <EOS> so it fits the max length
target = np.zeros((predict.shape[1], predict.shape[3]), dtype='int32')
target[:, :output['target'].shape[1]] = output['target']

miss_error = np.zeros(epochs)

def set_eos_padding(class_predict):
    class_predict = np.copy(class_predict)
    eosi = np.argmin(class_predict, axis=1)
    for obs, obs_eosi in enumerate(eosi):
        class_predict[obs, obs_eosi:] = 0
    return class_predict

for epoch, epoch_predict in enumerate(predict):
    class_predict = set_eos_padding(np.argmax(epoch_predict, axis=1))
    miss_error[epoch] = np.mean(class_predict != target)

plt.figure(figsize=(8, 8))
plt.suptitle(name)

plt.subplot(2, 1, 1)
plt.plot(np.arange(0, epochs), miss_error, color='SteelBlue', label='test')
plt.axhline((dims - 1) / dims, color='gray')
plt.legend()
plt.ylabel('misclassification error [%]')
plt.xlabel('epoch')
plt.ylim(0, 1)

plt.subplot(2, 1, 2)
plt.plot(np.arange(0, epochs), train, color='IndianRed', label='train')
plt.plot(np.arange(0, epochs), test, color='SteelBlue', label='test')
plt.axhline(-math.log(1 / dims), color='gray')
plt.legend()
plt.ylabel('loss [entropy]')
plt.xlabel('epoch')
plt.ylim(0, 8)

plt.savefig(path.join(thisdir, '..', 'outputs', name[:-4] + '.png'))
plt.show()
