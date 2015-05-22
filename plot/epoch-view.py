
import plot

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

epochs = int(np.max(output['train_miss_epoch']))
n_classes = output['n_classes'][0]
name = path.basename(sys.argv[1])[:-4]

plt.figure(figsize=(8, 8))
plt.suptitle(name)

miss_fig = plt.subplot(2, 1, 1)
plt.plot(output['train_miss_epoch'], output['train_miss'], color='IndianRed', label='train')
plt.plot(output['test_miss_epoch'], output['test_miss'], color='SteelBlue', label='test')
plt.axhline((n_classes - 1) / n_classes, color='gray')
plt.legend()
plt.ylabel('misclassification error [rate]')
plt.xlabel('epoch')
plt.ylim(0, 1.1)

print(output['train_loss_epoch'])

loss_fig = plt.subplot(2, 1, 2)
plt.plot(output['train_loss_epoch'], output['train_loss'], color='IndianRed', label='train')
plt.plot(output['test_loss_epoch'], output['test_loss'], color='SteelBlue', label='test')
plt.axhline(-math.log(1 / n_classes), color='gray')
plt.legend()
plt.ylabel('loss [entropy]')
plt.xlabel('epoch')

if (epochs <= 20):
    miss_fig.set_xticks(range(0, epochs))
    miss_fig.xaxis.grid(True)
    loss_fig.set_xticks(range(0, epochs))
    loss_fig.xaxis.grid(True)

plt.savefig(path.join(thisdir, '..', 'outputs', name + '.png'))
plt.show()
