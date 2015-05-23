
import os
import os.path as path

import numpy as np
import scipy

class Cluster:
    def __init__(self, threshold=0.11, verbose=False):
        """Constructs an model object capable of finding similar documents
        in a network graph perspectiv.

        Parameters
        ----------
        threshold : float
            Observation pairs with distance bellow this value will be joined
        verbose : boolean
            If true verbose text will be printed.
        """
        # Initialize verbose flags
        self._verbose = verbose

        # Store model parameters
        self._threshold = threshold

        if (self._verbose): print("Initialized new cluster model")

    def transform(self, distance):
        """Learn network grouping from distance matrix.
        """

        if (self._verbose):
            print("Learning groups from data")
            print("\tCalculating sparse connects matrix")
        # Build join matrix, this will effectivly calculate:
        # for row, col in zip(connectivity.row, connectivity.col):
        #    if distance[row, col] < self._threshold:
        #        connects[row, col] = True
        # connects += connects.T # make it symmetric
        mask = distance.data < self._threshold
        connects_row = distance.row[mask]
        connects_col = distance.col[mask]
        row_index = np.hstack([connects_row, connects_col])
        col_index = np.hstack([connects_col, connects_row])

        def connected_nodes(node_id):
            return set(col_index[row_index == node_id])

        # Allocate group_ list, this is actually nessearry as there would
        # otherwise be a risk of getting an IndexError
        node_to_group = np.empty(distance.shape[0], dtype='int32')

        # Create non-hieratical groups from connects matrix
        ungrouped_nodes = set(range(0, distance.shape[0]))
        group = []
        group_size = []
        current_group_id = 0

        # Continue for as long as there are ungrouped node.
        # Remember a group can contain only one node
        while len(ungrouped_nodes) != 0:
            source_node = ungrouped_nodes.pop()
            # Create a new group
            group_nodes = {source_node}
            # Set node_id -> group_id
            node_to_group[source_node] = current_group_id

            lookup = connected_nodes(source_node)
            while len(lookup) != 0:
                # Set node_id -> group_id
                source_node = lookup.pop()
                node_to_group[source_node] = current_group_id
                # Add node to group
                group_nodes.add(source_node)
                # Add new connected nodes
                lookup.update(connected_nodes(source_node) - group_nodes)

            # Save the group and remove its nodes from the ungrouped set
            group.append(np.fromiter(group_nodes, dtype='int32'))
            ungrouped_nodes.difference_update(group_nodes)
            group_size.append(len(group_nodes))

            # Print verbose text
            current_group_id += 1
            if (self._verbose and current_group_id % 500 == 0):
                print("\tProgress: %d groups created, %d nodes remains" % (
                    current_group_id, len(ungrouped_nodes)
                ))

        if (self._verbose): print("\tBuilding output datastructures")
        group_size_array = np.asarray(group_size, dtype='int32')

        group_array = np.zeros((len(group), np.max(group_size_array)), dtype='int32')
        for i, group_i in enumerate(group):
            group_array[i, 0:group_i.shape[0]] = group_i

        if (self._verbose): print("\tGroup creation done")

        return {
            'connects_row': connects_row,
            'connects_col': connects_col,
            'group_size': group_size_array,
            'group': group_array,
            'node_to_group': node_to_group
        }
