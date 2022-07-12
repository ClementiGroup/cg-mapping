import numpy as np
from typing import List
from itertools import combinations
import networkx as nx

def _mapping_combinations(n_atoms : int, n_beads: int) -> List:
    '''
        Defines all possible combinatorial possibilities 
    '''
    mapping_list = []
    combs = list(combinations(range(n_atoms), n_beads))
    for comb in combs:
        mapping_matrix = np.zeros((n_beads,n_atoms))
        for ix in range(n_beads):
            mapping_matrix[ix,comb[ix]] = 1.
        mapping_list.append(mapping_matrix)
    return mapping_list

def _create_networkx(kirchoff_matrix : np.ndarray):
    '''
        Given a kirchoff matrix, return a networkx representation
    '''
    graph = nx.Graph()
    n_atoms = np.shape(kirchoff_matrix)[0]
    nodes = []

    for ind in range(n_atoms):
        node = {}
        center = {
            'x':0+ind,
            'y':0+ind,
            'z':0+ind,
        }
        node['center'] = center
        nodes.append(node)
        
    for ind,node in enumerate(nodes):
        graph.add_node(ind,**node)

    edges = []
    for i_r,row in enumerate(kirchoff_matrix):
        inds = np.nonzero(row)[0]
        for ind in inds:
            if ind == i_r: continue
            edges.append([i_r,ind])
    graph.add_edges_from(edges)

    return graph
