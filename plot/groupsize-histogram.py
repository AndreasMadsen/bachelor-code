
import plot
import model

import sys
import numpy as np
import os.path as path
import matplotlib.pyplot as plt

group_size = model.load(path.realpath(sys.argv[1]))['group_size']
sizes = np.unique(group_size)

print("%7s | %3s" % ("size", "count"))

for size in sizes:
    print("%7d | %3d" % (size, np.sum(group_size == size)))

print('\nLATEX:')
print('\\begin{table}[H]')
print('\\centering')
print('\\begin{tabular}{r|' + ('l ' * sizes.shape[0]) + '}')
print('size & ' + ' & '.join([str(size) for size in sizes]) + ' \\\\ \\hline')
print('amount & ' + ' & '.join([str(np.sum(group_size == size)) for size in sizes]))
print('\\end{tabular}')
print('\\caption{%s}' % (path.basename(sys.argv[1])))
print('\\end{table}')

print('\ngood groups:')
groups = np.where(np.logical_and(group_size < 14, group_size > 4))[0]

print('id | size')
for id in groups:
    print('%d | %d' % (id, group_size[id]))
