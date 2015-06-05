import plot
import model
import dataset

import sys
import numpy as np
import os.path as path

node_id = int(sys.argv[1])
cluster = model.load(path.realpath(sys.argv[2]))

group_id = cluster['node_to_group'][node_id]
articles = dataset.news.fetch(100000)
nodes = cluster['group'][group_id, 0:cluster['group_size'][group_id]]

print("\\begin{table}[H]")
print("\\centering")
print("\\begin{tabular}{r|l}")
print("id & title \\\\ \\hline")

for i, node_id in enumerate(nodes):
    print("%6d & %s %s" % (node_id, articles[node_id]['title'], '' if len(nodes) - 1 == i else '\\\\'))

print("\\end{tabular}")
print("\\caption{TODO}")
print("\\end{table}")
