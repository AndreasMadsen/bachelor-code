
import plot

import json
import sys
import os.path as path
import numpy as np
import matplotlib.pyplot as plt

with open(path.realpath(sys.argv[1]), 'r') as infile:
    results = json.load(infile)

# Build grid_search params object
params = {k: [] for k, v in results[0]['params'].items()}
test_loss = np.zeros(len(results), dtype='float32')
train_loss = np.zeros(len(results), dtype='float32')

for i, result in enumerate(results):
    for k, v in result['params'].items():
        params[k].append(v)

    test_loss[i] = result['test_loss']
    train_loss[i] = result['train_loss']

params = {k: np.unique(v) for k, v in params.items()}

# Print best and worst
def print_example(index):
    print('\ttest loss: %f' % test_loss[index])
    print('\ttrain loss: %f' % train_loss[index])
    print('\tParameters:')
    for k, v in results[index]['params'].items():
        print('\t\t%s: %f' % (k, v))

print('Best result (by test loss):')
print_example(np.argmin(test_loss))

print('Worst result (by test loss):')
print_example(np.argmax(test_loss))

# Specify x,y axis to show
x_axis = 'learning_rate'
y_axis = 'momentum'

# Create datamatrix
test_grid = np.zeros((params[x_axis].size, params[y_axis].size))
train_grid = np.zeros((params[x_axis].size, params[y_axis].size))

for x_i, x_val in enumerate(params[x_axis]):
    for y_i, y_val in enumerate(params[y_axis]):
        for result in results:
            if (result['params'][x_axis] == x_val and result['params'][y_axis] == y_val):
                test_grid[x_i, y_i] += result['test_loss']
                train_grid[x_i, y_i] += result['train_loss']

mean_factor = len(results) / (test_grid.shape[0] * test_grid.shape[1])
test_grid = test_grid / mean_factor
train_grid = train_grid / mean_factor

# Show
# cmap_min = min(np.min(test_loss), np.min(train_loss))
# cmap_max = min(np.max(test_loss), np.max(train_loss))
cmap_min = min(np.min(test_grid), np.min(train_grid))
cmap_max = max(np.max(test_grid), np.max(train_grid))

def plot_grid(grid):
    im = plt.imshow(grid.T, vmax=cmap_max, vmin=cmap_min, interpolation='nearest')
    im.set_cmap('binary_r')

    plt.xlabel(x_axis)
    plt.xticks(range(0, len(params[x_axis])), params[x_axis])
    plt.ylabel(y_axis)
    plt.yticks(range(0, len(params[y_axis])), params[y_axis])

    return im

fig = plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.title('test loss')
plot_grid(test_grid)

plt.subplot(1, 2, 2)
plt.title('train loss')
im = plot_grid(train_grid)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

plt.show()
