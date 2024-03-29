import numpy as np
from typing import List
from _methods_utils import (
    _slicing_mapping_combinations,
    _create_networkx,
    _symmetric_mapping_combinations,
    _find_all_averaged_mapping_combinations,
)


class VAMP:
    """
    TODO: Documentation
    """

    _mapping_stratgies = {
        "slice": _slicing_mapping_combinations,
        "center": _symmetric_mapping_combinations,
        "average": _find_all_averaged_mapping_combinations,
    }

    def __init__(
        self,
        kirchoff_matrix: np.ndarray = None,
        kbT: int = 1,
        n_beads: int = 2,
        lagtimes: List = None,
        dt: float = 0.001,
        mapping: str = "slice",
    ):
        """ """
        assert len(np.shape(kirchoff_matrix)) == 2
        assert np.shape(kirchoff_matrix)[0] == np.shape(kirchoff_matrix)[1]
        self.n_atoms = np.shape(kirchoff_matrix)[0]
        self.n_beads = n_beads
        self.kirchoff_matrix = kirchoff_matrix
        self.inv_kirchoff_matrix = np.linalg.pinv(self.kirchoff_matrix)
        self.lagtimes = lagtimes
        self.dt = dt
        self.graph = _create_networkx(self.kirchoff_matrix)
        self.mapping_strategy = self._mapping_stratgies[mapping]

        eigvals, eigvecs = np.linalg.eigh(self.kirchoff_matrix)
        self.eigvals = eigvals
        self.eigvecs = eigvecs.T
        self.eigval0 = self.eigvals[0]
        self.eigvec0 = self.eigvecs[0]

    def compute_vamp(self, lagtime):
        """
        Compute the VAMP score for all possible mapping combinations at a given lagtime.
        keys of _vamp_score are the associated indices

        Inputs:
            lagtime --

        """
        mapping_matrices = self.mapping_strategy(self.n_atoms, self.n_beads)
        self._mapping_matrices = mapping_matrices
        self._vamp_score = {}
        self._lagtime = lagtime
        for i_m, mapping_matrix in enumerate(mapping_matrices):

            # Filter out free translation
            Jn = np.sqrt(self.n_atoms) * self.eigvec0.reshape([-1, 1])
            JN = mapping_matrix @ Jn
            Qn = np.identity(self.n_beads) - 1.0 / float(self.n_beads) * np.outer(
                JN, JN
            )

            C00 = Qn @ mapping_matrix @ self.inv_kirchoff_matrix @ mapping_matrix.T @ Qn
            inv_C00 = np.linalg.pinv(C00)
            omega_tau = self.compute_omega_tau()
            C0tau = (
                Qn
                @ mapping_matrix
                @ self.inv_kirchoff_matrix
                @ omega_tau
                @ mapping_matrix.T
                @ Qn
            )
            self._vamp_score[i_m] = np.trace(inv_C00 @ C0tau)

    def return_vamp_scores(self):
        """
        Return vamp score for mappings and associated lagtime
        """
        return self._vamp_score, self._lagtime

    def return_optimal_vamp_score(self):
        """
        Pass back inds and score for best vamp mapping
        """
        return max(self._vamp_score.items(), key=lambda x: x[1])

    def compute_omega_tau(self):
        """
        Expected value of propagator for GNM
        """
        omega_tau = np.zeros((self.n_atoms, self.n_atoms))
        for (eigval, eigvec) in zip(self.eigvals, self.eigvecs):
            omega_tau += np.outer(eigvec, eigvec) * np.exp(
                -eigval * self._lagtime * self.dt
            )
        return omega_tau
