import numpy as np
from typing import List
from _utils import _mapping_combinations, _create_networkx

class AME():
    '''
    TODO: Documentation
    '''

    def __init__(
        self, kirchoff_matrix : np.ndarray = None, kbT : int = 1, n_beads : int = 2, lagtimes : List = None
    ):
        '''
        '''
        assert len(np.shape(kirchoff_matrix)) == 2
        assert np.shape(kirchoff_matrix)[0] == np.shape(kirchoff_matrix)[1]
        self.n_atoms = np.shape(kirchoff_matrix)[0]
        self.n_beads = n_beads
        self.kirchoff_matrix = kirchoff_matrix
        self.inv_kirchoff_matrix = np.linalg.pinv(self.kirchoff_matrix)
        self.graph = _create_networkx(self.kirchoff_matrix)

        eigvals,eigvecs = np.linalg.eigh(self.kirchoff_matrix)
        self.eigvals = eigvals
        self.eigvecs = eigvecs.T
        self.eigval0 = self.eigvals[0]
        self.eigvec0 = self.eigvecs[0]

    def compute_ame(self, tol = 1e-6):
        '''
            Compute the VAMP score for all possible mapping combinations at a given lagtime.
            keys of _vamp_score are the associated indices

            Inputs:
            
        '''
        mapping_matrices = _mapping_combinations(self.n_atoms, self.n_beads)
        self._ame_score = {}
        self._vp_score = {}
        for mapping_matrix in mapping_matrices:
            inds = tuple(np.nonzero(mapping_matrix)[1])

            # Filter out free translation
            Jn = np.sqrt(self.n_atoms) * self.eigvec0.reshape([-1,1])
            JN = mapping_matrix @ Jn
            Qn = np.identity(self.n_beads)- 1./float(self.n_beads)*np.outer(JN, JN)
            cg_kirchoff = Qn @ mapping_matrix @ self.inv_kirchoff_matrix @ mapping_matrix.T @ Qn
            cg_eigvals, cg_eigvecs = np.linalg.eigh(cg_kirchoff)
            cg_eigvecs = cg_eigvecs.T

            cg_kirchoff = np.zeros((self.n_beads, self.n_beads))
            # Filter out translation modes (eignvalues < tol)
            for i_e,(cg_eigval, cg_eigvec) in enumerate(zip(cg_eigvals, cg_eigvecs)):
                if np.abs(cg_eigval) < tol: continue
                cg_eigvec = cg_eigvec.reshape([-1,1])
                cg_kirchoff += 1/cg_eigval * cg_eigvec @ cg_eigvec.T
            w,_ = np.linalg.eigh(cg_kirchoff)
            Tk = np.product(w[np.abs(w)>tol])

            ame_score = np.log(Tk)
            vp_score = np.sum(self.compute_square_fluc(cg_kirchoff))
            self._ame_score[inds] = ame_score
            self._vp_score[inds] = vp_score
    
    def return_vp_scores(self):
        '''
            Return vp score for mappings
        '''
        return self._vp_score

    def return_ame_scores(self):
        '''
            Return ame score for mappings
        '''
        return self._ame_score

    @staticmethod
    def compute_square_fluc(cg_kirchoff_matrix):
        '''

        '''
        inv_kappa = 0
        eigvals,eigvecs = np.linalg.eigh(cg_kirchoff_matrix)
        for i_e,(eigval, eigvec) in enumerate(zip(eigvals, eigvecs)):
            if eigval < 1e-6: continue
            inv_kappa += 1/eigval*eigvec*eigvec.T
        return inv_kappa