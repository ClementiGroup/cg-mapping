import numpy as np
import torch
from torch.nn import Parameter
from torch import nn, optim
import torch.nn.functional as F
from _methods_utils import (
    _slicing_mapping_combinations,
    _create_networkx,
    _symmetric_mapping_combinations,
    _find_all_averaged_mapping_combinations,
    decoder_linear,
    decoder_non_linear_relu,
)

class RE:
    """
    Structure reconstruction using pytorch
    """
    
    _mapping_stratgies = {
        "slice": _slicing_mapping_combinations,
        "center": _symmetric_mapping_combinations,
        "average": _find_all_averaged_mapping_combinations,
    }

    def __init__(
        self,
        n_beads: int = 2,
        mapping: str = "slice",
        mode: str = "non linear",
        traj: np.ndarray = None,
        device: str = "cpu"
    ):
        assert len(np.shape(traj)) == 3
        self.n_beads = n_beads
        self.device = device
        self.mode = mode
        np.random.shuffle(traj)
        self.traj = traj
        self.n_atoms = np.shape(traj)[1]
        self.dim = np.shape(traj)[2]
        self.mapping_strategy = self._mapping_stratgies[mapping]
        
    def init_decoder(
        self
    ):
        decoders = []
        decoders.append(decoder_linear(in_dim=self.n_beads,
                                       out_dim=self.n_atoms,
                                       device=self.device))
        if self.mode == "non linear":
            decoders.append(decoder_non_linear_relu(in_dim=self.n_atoms,
                                                    out_dim=self.n_atoms,
                                                    device=self.device))
        else:
            pass
        
        return decoders
    
    def data_prepare(
        self,
        batch_size: int = 512,
        crossvalid: float = 0.2,      # ratio of data as test set 
    ):
        batch_size_train = batch_size
        batch_size_test = int(batch_size_train * crossvalid / (1 - crossvalid))

        cut_point = int(self.traj.shape[0] * (1 - crossvalid))
        traj_train = self.traj[0:cut_point,:,:]
        traj_test = self.traj[cut_point::,:,:]

        n_batch = int(traj_train.shape[0] // batch_size_train)
        n_sample_train = n_batch * batch_size_train
        n_sample_test = n_batch * batch_size_test
        xyz_train = traj_train[:n_sample_train].reshape(-1, batch_size_train, self.n_atoms, self.dim)
        xyz_test = traj_test[:n_sample_test].reshape(-1, batch_size_test, self.n_atoms, self.dim)
        
        return xyz_train, xyz_test
    
    def train_RE(
        self,
        lr_linear: float = 1e-3,      # learning rate for the linear part
        lr_non_linear: float = 2e-4,  # learning rate for the non-linear part
        max_epoch: int = 451,         
        epoch_nonlinear: int = 200,   # the epoch start to include non-linear decoder
        batch_size: int = 512,
        crossvalid: float = 0.2,      # ratio of data as test set 
        loss_print: bool = True,      # whether print out the loss values during training
    ):
        if self.mode == "linear":
            assert max_epoch == epoch_nonlinear
            
        mapping_matrices = self.mapping_strategy(self.n_atoms, self.n_beads)
        self._mapping_matrices = mapping_matrices
        self.loss_trains = {}
        self.loss_tests = {}
        
        # Get batched data and initialize the optimizers
        xyz_train, xyz_test = self.data_prepare(batch_size, crossvalid) 
        criterion = torch.nn.MSELoss()
        
        for i_m, mapping_matrix in enumerate(mapping_matrices):
            optimizers = []
            decoders = self.init_decoder()
            encoder = torch.Tensor(mapping_matrix).to(self.device)
            for i, decoder in enumerate(decoders):
                if i < 1:
                    optimizers.append(optim.Adam(list(decoder.parameters()), lr=lr_linear))
                else:
                    optimizers.append(optim.Adam(list(decoder.parameters()), lr=lr_non_linear))
        
            # Train the model
            loss_log_train = []
            loss_log_test = []
    
            for epoch in range(max_epoch):  
                loss_epoch_train = 0.0
                loss_epoch_test = 0.0
            
                for k in range(xyz_train.shape[0]):
                    batch_train  = torch.Tensor(xyz_train[k]).to(self.device) 
                    CGs_train = torch.matmul(encoder.expand(batch_train.shape[0], self.n_beads, self.n_atoms), batch_train)
        
                    batch_test = torch.Tensor(xyz_test[k]).to(self.device) 
                    CGs_test = torch.matmul(encoder.expand(batch_test.shape[0], self.n_beads, self.n_atoms), batch_test)

                    decoded_linear_train = decoders[0](CGs_train)
                    decoded_linear_test = decoders[0](CGs_test)
                
                    if epoch < epoch_nonlinear:
                        decoded_train = decoded_linear_train
                        decoded_test = decoded_linear_test
                    else:
                        decoded_nonlinear_train = decoders[1](decoded_linear_train)
                        decoded_nonlinear_test = decoders[1](decoded_linear_test)
                        decoded_train = decoded_nonlinear_train + decoded_linear_train
                        decoded_test = decoded_nonlinear_test + decoded_linear_test          
            
                    loss_train = criterion(decoded_train, batch_train)
                    loss_test = criterion(decoded_test, batch_test)
    
                    if epoch < epoch_nonlinear:
                        optimizers[0].zero_grad()
                        loss_train.backward()
                        optimizers[0].step()
                    else:
                        optimizers[1].zero_grad()
                        loss_train.backward()
                        optimizers[1].step()
            
                    loss_epoch_train += loss_train.item()
                    loss_epoch_test += loss_test.item()
        
                loss_epoch_train = loss_epoch_train/xyz_train.shape[0]
                loss_epoch_test = loss_epoch_test/xyz_test.shape[0]

                loss_log_train.append(loss_epoch_train)
                loss_log_test.append(loss_epoch_test)
        
                if epoch%50 == 0 and loss_print:
                    print("epoch %d reconstruction %.3f validation %.3f" % (epoch, loss_epoch_train, loss_epoch_test))
        
            self.loss_trains[i_m] = loss_log_train
            self.loss_tests[i_m] = loss_log_test
        
       