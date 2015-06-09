
import plot

import os.path as path
import sys
import math

import numpy as np
import matplotlib.pyplot as plt

thisdir = path.dirname(path.realpath(__file__))

if (len(sys.argv) < 2):
    print('python3 epoch-view.py output.npz [report]')
    sys.exit(1)

report_plot = (len(sys.argv) >= 3 and sys.argv[2] == 'report')

output = np.load(path.realpath(sys.argv[1]))

epochs = output['train_loss'].shape[0]
n_classes = output['n_classes'][0]
name = path.basename(sys.argv[1])[:-4]

problem_type = 'regression' if ('type' in output and output['type'] == 'regression') else 'classification'

fig = plt.figure(figsize=(7.0, 5.5) if report_plot else (8, 8))
if (not report_plot): plt.suptitle(name)

if (problem_type == 'classification'):
    miss_fig = plt.subplot(2, 1, 1)
    plt.plot(range(0, epochs), output['train_miss'], color='IndianRed', label='train')
    plt.plot(range(0, epochs), output['test_miss'], color='SteelBlue', label='test')
    plt.axhline((n_classes - 1) / n_classes, color='gray')
    plt.legend(prop={'size': 12})
    plt.ylabel('misclassification error [rate]')
    plt.xlabel('epoch')
    plt.ylim(0, 1.1)

loss_fig = plt.subplot(2, 1, 2)
if ('train_loss_minibatch' in output):
    plt.plot(output['train_loss_minibatch_epoch'], output['train_loss_minibatch'],
             color='IndianRed', alpha=0.5, label='train minibatch')

plt.plot(range(0, epochs), output['train_loss'], color='IndianRed', label='train')
plt.plot(range(0, epochs), output['test_loss'], color='SteelBlue', label='test')
if (problem_type == 'classification'): plt.axhline(-math.log(1 / n_classes), color='gray')
plt.legend(prop={'size': 12})
plt.ylabel('loss [%s]'  % ('entropy' if problem_type == 'classification' else 'MSE'))
plt.xlabel('epoch')
plt.ylim(0, plt.ylim()[1])

if (epochs <= 20):
    miss_fig.set_xticks(range(0, epochs))
    miss_fig.xaxis.grid(True)
    loss_fig.set_xticks(range(0, epochs))
    loss_fig.xaxis.grid(True)

fig.set_tight_layout(True)
plt.show()

plt.savefig(path.join(thisdir, '..', 'outputs', name + '.png'))
if (report_plot):
    plt.savefig(path.join(thisdir, '..', '../report/graphics/results/', 'sutskever-' + name + '.pdf'))
