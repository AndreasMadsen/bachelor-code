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

epochs = output['train_miss'].shape[0]
n_classes = output['n_classes'][0]
name = path.basename(sys.argv[1])[:-4]

plt.figure(figsize=(8, 8))
plt.suptitle(name)

plt.subplot(2, 1, 1)
plt.plot(np.arange(0, epochs), output['train_miss'], color='IndianRed', label='train')
plt.plot(np.arange(0, epochs), output['test_miss'], color='SteelBlue', label='test')
plt.axhline((n_classes - 1) / n_classes, color='gray')
plt.legend()
plt.ylabel('misclassification error [rate]')
plt.xlabel('epoch')
plt.ylim(0, 1.1)

plt.subplot(2, 1, 2)
plt.plot(np.arange(0, epochs), output['train_loss'], color='IndianRed', label='train')
plt.plot(np.arange(0, epochs), output['test_loss'], color='SteelBlue', label='test')
plt.axhline(-math.log(1 / n_classes), color='gray')
plt.legend()
plt.ylabel('loss [entropy]')
plt.xlabel('epoch')

plt.savefig(path.join(thisdir, '..', 'outputs', name + '.png'))
plt.show()
