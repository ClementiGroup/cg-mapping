import sys

sys.path.insert(0, "/import/a12/users/clarkt/mapping-entropy/cg-mapping/simulation")
sys.path.insert(0, "/import/a12/users/clarkt/mapping-entropy/cg-mapping/methods")

import numpy as np
from VAMPClass import VAMP
from AnalyticalMappingEntropyClass import AME
from SimulationClass import Simulation
from TicaTimescalesClass import TICA
import networkx as nx

path = '/import/a12/users/clarkt/mapping-entropy/cg-mapping/examples/'
kirchoff_matrix = np.load(path+'2erl-10A-kirchoff-matrix.npy')
kirchoff_matrix = [[ 10,-10,  0,  0], 
                   [-10, 11, -1,  0], 
                   [  0, -1, 11,-10], 
                   [  0,  0,-10, 10]]
# simulation.run_simulation(10000000)

ame = AME(kirchoff_matrix, n_beads = 2)
ame.compute_ame()
ame_results = ame.return_ame_scores()
vp_results = ame.return_vp_scores()

n_atoms = len(kirchoff_matrix)
tica = TICA(kirchoff_matrix=kirchoff_matrix,n_beads=n_atoms)
lagtime = 100
mapping_matrix = np.eye(n_atoms)
tica.compute_timescale_modes(mapping_matrix, lagtime)
