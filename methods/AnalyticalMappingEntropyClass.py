import numpy as np
from typing import List
from _methods_utils import (
    _slicing_mapping_combinations,
    _create_networkx,
    _symmetric_mapping_combinations,
    _find_all_averaged_mapping_combinations,
)


class AME:
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
        volume: float = 10,
        mapping: str = "slice",
    ):
        """ """
        assert len(np.shape(kirchoff_matrix)) == 2
        assert np.shape(kirchoff_matrix)[0] == np.shape(kirchoff_matrix)[1]
        self.n_atoms = np.shape(kirchoff_matrix)[0]
        self.n_beads = n_beads
        self.kirchoff_matrix = kirchoff_matrix
        self.inv_kirchoff_matrix = np.linalg.pinv(self.kirchoff_matrix)
        self.graph = _create_networkx(self.kirchoff_matrix)
        self.kbT = kbT
        self.volume = volume
        self.mapping_strategy = self._mapping_stratgies[mapping]

        eigvals, eigvecs = np.linalg.eigh(self.kirchoff_matrix)
        self.eigvals = eigvals
        self.eigvecs = eigvecs.T
        self.eigval0 = self.eigvals[0]
        self.eigvec0 = self.eigvecs[0]

    def compute_ame(self, tol=1e-6):
        """
        Compute the VAMP score for all possible mapping combinations at a given lagtime.
        keys of _vamp_score are the associated indices

        Inputs:

        """

        # Compute for all atom case
        tk = np.product(self.eigvals[np.abs(self.eigvals) > 1e-5])
        omega = np.trace(self.compute_square_fluc(self.kirchoff_matrix))

        mapping_matrices = self.mapping_strategy(self.n_atoms, self.n_beads)
        self._mapping_matrices = mapping_matrices
        self._ame_score = {}
        self._vp_score = {}
        for i_m, mapping_matrix in enumerate(mapping_matrices):
            
            # Filter out free translation
            Jn = np.sqrt(self.n_atoms) * self.eigvec0.reshape([-1, 1])
            JN = mapping_matrix @ Jn
            Qn = np.identity(self.n_beads) - 1.0 / float(self.n_beads) * np.outer(
                JN, JN
            )
            cg_kirchoff = (
                Qn @ mapping_matrix @ self.inv_kirchoff_matrix @ mapping_matrix.T @ Qn
            )
            cg_eigvals, cg_eigvecs = np.linalg.eigh(cg_kirchoff)
            cg_eigvecs = cg_eigvecs.T

            cg_kirchoff = np.zeros((self.n_beads, self.n_beads))
            # Filter out translation modes (eignvalues < tol)
            for i_e, (cg_eigval, cg_eigvec) in enumerate(zip(cg_eigvals, cg_eigvecs)):
                if np.abs(cg_eigval) < tol:
                    continue
                cg_eigvec = cg_eigvec.reshape([-1, 1])
                cg_kirchoff += 1 / cg_eigval * cg_eigvec @ cg_eigvec.T
            w, _ = np.linalg.eigh(cg_kirchoff)
            Tk = np.product(w[np.abs(w) > tol])

            # Smap <= 0. Taken from Foley/Noid
            #   First term is a constant that depends on bead resolution
            #   Second is what determines the best mapping
            #   Larger values imply better mapping schemes
            ame_score = 0.5 * (self.n_atoms - self.n_beads) * (
                1 + np.log(2 * np.pi / self.kbT / self.volume**2)
            ) + 0.5 * (np.log(Tk) - np.log(tk))

            # Vibrational Power from Foley/Noid

            #for average strategy include mass weighting (MW)
            if self.mapping_strategy == self._mapping_stratgies['average']:
                   MW=np.zeros((self.n_beads,self.n_beads))
                   for i in range(0,len(mapping_matrix)):
                       for j in range (0,len(mapping_matrix[i])):
                           if mapping_matrix[i][j] >0:
                              MW[i][i]+=1
                       MW[i][i]=np.sqrt(1/MW[i][i])        
                   vp_score = np.trace(self.compute_square_fluc(np.dot(MW,np.dot(cg_kirchoff,MW)))) / omega
            
           #for all other mapping strategies no mass weighting
            else:
               vp_score = np.trace(self.compute_square_fluc(cg_kirchoff)) / omega
           
            self._ame_score[i_m] = ame_score
            self._vp_score[i_m] = vp_score

    def return_vp_scores(self):
        """
        Return vp score for mappings
        """
        return self._vp_score

    def return_ame_scores(self):
        """
        Return ame score for mappings
        """
        return self._ame_score

    def return_optimal_ame_score(self):
        """
        Pass back inds and score for best ame mapping
        """
        return max(self._ame_score.items(), key=lambda x: x[1])

    def return_optimal_vp_score(self):
        """
        Pass back inds and score for best vp mapping
        """
        return max(self._vp_score.items(), key=lambda x: x[1])

    @staticmethod
    def compute_square_fluc(kirchoff_matrix):
        """ """
        inv_kappa = 0
        eigvals, eigvecs = np.linalg.eigh(kirchoff_matrix)
        eigvecs = eigvecs.T
        for i_e, (eigval, eigvec) in enumerate(zip(eigvals, eigvecs)):
            if eigval < 1e-6:
                continue
            eigvec = eigvec.reshape([-1, 1])
            inv_kappa += (1 / eigval) * eigvec @ eigvec.T

        return inv_kappa
