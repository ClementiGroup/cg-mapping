import numpy as np
from VAMPClass import VAMP


class TICA(VAMP):
    def __init__(self, **kwargs):
        super(TICA, self).__init__(**kwargs)

    def compute_timescale_modes(self, mapping_matrix, lagtime):
        self._lagtime = lagtime
        # Filter out free translation
        Jn = np.sqrt(self.n_atoms) * self.eigvec0.reshape([-1, 1])
        JN = mapping_matrix @ Jn
        Qn = np.identity(self.n_beads) - 1.0 / float(self.n_beads) * np.outer(JN, JN)
        cg_kirchoff = (
            Qn @ mapping_matrix @ self.inv_kirchoff_matrix @ mapping_matrix.T @ Qn
        )
        inv_cg_kirchoff = np.linalg.pinv(cg_kirchoff)
        omega_tau = self.compute_omega_tau()
        C0tau = (
            Qn
            @ mapping_matrix
            @ self.inv_kirchoff_matrix
            @ omega_tau
            @ mapping_matrix.T
            @ Qn
        )
        vamp_score = inv_cg_kirchoff @ C0tau
        eigvals, eigvecs = np.linalg.eig(vamp_score)
        eigvecs = eigvecs.T
        sorted_inds = np.argsort(eigvals)[::-1]
        eigvals_sorted = eigvals[sorted_inds]
        eigvecs_sorted = eigvecs[sorted_inds]

        self._eigvals_sorted = eigvals_sorted
        self._eigves_sorted = eigvecs_sorted

    def return_eigvals_eigvecs(self):
        """
        Return eigvals and eigvecs for an associated lagtime
        """
        return self._eigvals_sorted, self._eigves_sorted, self._lagtime

    def return_timescales(self):
        """ """
        eigvals, _, lagtime = self.return_eigvals_eigvecs()
        self._timescales = -lagtime / np.log(np.abs(eigvals))
        return self._timescales
