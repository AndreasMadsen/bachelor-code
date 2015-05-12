
import plot

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

train_sizes = output['train_sizes']
dims = output['n_classes']
name = path.basename(sys.argv[1])[:-4]

# Plot
plt.figure(figsize=(8, 8))
plt.suptitle(name)

plt.subplot(2, 1, 1)
plt.plot(train_sizes, output['train_miss'], '-x', color='IndianRed', label='train')
plt.plot(train_sizes, output['test_miss'], '-x', color='SteelBlue', label='test')
plt.axhline((dims - 1) / dims, color='gray')
plt.legend()
plt.ylabel('misclassification error [%]')
plt.xlabel('train size')
plt.ylim(0, 1.1)

plt.subplot(2, 1, 2)
plt.plot(train_sizes, output['train_loss'], '-x', color='IndianRed', label='train')
plt.plot(train_sizes, output['test_loss'], '-x', color='SteelBlue', label='test')
plt.axhline(-math.log(1 / dims), color='gray')
plt.legend()
plt.ylabel('loss [entropy]')
plt.xlabel('train size')

plt.savefig(path.join(thisdir, '..', 'outputs', name + '.png'))
plt.show()
