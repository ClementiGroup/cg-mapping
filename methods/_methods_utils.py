import numpy as np
from typing import List
from itertools import combinations
import networkx as nx


def _slicing_mapping_combinations(n_atoms: int, n_beads: int) -> List:
    """
    Defines all possible combinatorial possibilities using a slice mapping
    """
    mapping_list = []
    combs = list(combinations(range(n_atoms), n_beads))
    for comb in combs:
        mapping_matrix = np.zeros((n_beads, n_atoms))
        for ix in range(n_beads):
            mapping_matrix[ix, comb[ix]] = 1.0
        mapping_list.append(mapping_matrix)
    return mapping_list


def _symmetric_mapping_combinations(n_atoms: int, n_beads: int) -> List:
    """
    Defines a mapping averaging beads together
    """
    mapping_list = []

    partition = (n_atoms / n_beads) * np.ones(n_beads)
    mapping_matrix = np.zeros((n_beads, n_atoms))
    for ix in range(len(partition)):
        ind_left = int(np.sum(partition[:ix]))
        ind_right = int(np.sum(partition[: ix + 1]))
        mapping_matrix[ix, ind_left:ind_right] = 1.0 / partition[ix]
        mapping_list.append(mapping_matrix)
    return mapping_list


def _find_all_partitions(n_atoms: int, n_beads: int) -> List:
    """
    Define how many atoms to give each bead
    """
    partitions = []
    combs = combinations(range(n_atoms - 1), n_beads - 1)
    for comb in list(combs):
        partition = np.ones(n_beads, dtype=int)
        # Set count at first index
        partition[0] += comb[0]
        # Sum of atoms left to be placed
        for jx in range(1, n_beads - 1):
            n_atoms_to_place = comb[jx] - comb[jx - 1] - 1
            partition[jx] += n_atoms_to_place
        n_atoms_left_to_assign = n_atoms - np.sum(partition)
        partition[-1] += n_atoms_left_to_assign
        partitions.append(partition)
    return partitions


def _find_all_averaged_mapping_combinations(
    n_atoms: int, n_beads: int, weights: np.ndarray = None
) -> List:
    """
    Defines all possible combinatorial possibilities using an averaging mapping.
    Averages are unweighted.

    Inputs:
        n_atoms -- Number of fine grain atoms
        n_beads -- Number of coarse grain beads
        weights -- weight to assign each atom

    Outputs:
        mapping_list -- all possible mapping matrices


    E.g. Given 5 atoms with 2 beads
        1 atom  to bead 1, 4 atoms to bead 2
        2 atoms to bead 1, 3 atoms to bead 2
        3 atoms to bead 1, 2 atoms to bead 2
        4 atoms to bead 1, 1 atoms to bead 2

    """
    mapping_list = []
    partitions = _find_all_partitions(n_atoms, n_beads)
    for part in partitions:
        mapping_matrix = np.zeros((n_beads, n_atoms))
        # Selection of where to stop and where to end
        for ix in range(len(part)):
            ind_left = int(np.sum(part[:ix]))
            ind_right = int(np.sum(part[: ix + 1]))
            if weights is None:
                mapping_matrix[ix, ind_left:ind_right] = 1.0 / part[ix]
        mapping_list.append(mapping_matrix)
    return mapping_list


def _create_networkx(kirchoff_matrix: np.ndarray):
    """
    Given a kirchoff matrix, return a networkx representation.
    """
    graph = nx.Graph()
    n_atoms = np.shape(kirchoff_matrix)[0]
    nodes = []

    for ind in range(n_atoms):
        node = {}
        center = {
            "x": 0 + ind,
            "y": 0 + ind,
            "z": 0 + ind,
        }
        node["center"] = center
        nodes.append(node)

    for ind, node in enumerate(nodes):
        graph.add_node(ind, **node)

    edges = []
    for i_r, row in enumerate(kirchoff_matrix):
        inds = np.nonzero(row)[0]
        for ind in inds:
            if ind == i_r:
                continue
            edges.append([i_r, ind])
    graph.add_edges_from(edges)

    return graph
