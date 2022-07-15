import numpy as np
from typing import List
from _simulation_utils import tqdm


class Simulation:
    """
    TODO: Documentation
    """

    def __init__(
        self,
        kirchoff_matrix: np.ndarray = None,
        kbT: int = 1,
        gamma: float = 0.1,
        dt: float = 0.001,
        save_frequency: int = 100,
        fout: str = 'coords.npy',
        coords: np.ndarray = None,
        n_steps: int = 1000,
    ):
        '''

        '''
        assert len(np.shape(kirchoff_matrix)) == 2
        assert np.shape(kirchoff_matrix)[0] == np.shape(kirchoff_matrix)[1]
        self.n_atoms = np.shape(kirchoff_matrix)[0]
        self.kirchoff_matrix = kirchoff_matrix
        self.inv_kirchoff_matrix = np.linalg.pinv(self.kirchoff_matrix)
        self.dt = dt
        self.kbT = kbT
        self.gamma = gamma
        self.save_frequency = save_frequency
        self.fout = fout
        self.n_steps = n_steps
        if coords is not None:
            self.coords = coords
        else:
            self.coords = np.zeros((self.n_atoms,3))

    def run_simulation(self):
        '''

        '''
        coords = []
        for step in tqdm(range(self.n_steps), desc="Simulation timestep"):
            self.coords = self.langevin_step()
            if step % self.save_frequency == 0:
                coords.append([])
                coords[-1].append(self.coords)
        coords = np.concatenate(coords,axis=0)
        print(coords.shape)
        np.save(self.fout,coords)

    def langevin_step(self):
        '''
            Take a single langevin step
        '''
        x0 = self.coords
        force = -np.dot(self.kirchoff_matrix, x0)
        dB = np.random.normal(0.,np.sqrt(2.*self.kbT*self.dt),(self.n_atoms,3))
        xt = x0 + force*self.dt + dB
        # Recenter
        xt = xt - np.mean(xt,axis=0)
        return xt