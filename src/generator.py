import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from potential import Potential
from util import ress

class CudaNotAvailableError(Exception):
    pass

class GeneratorBuilder:
    """Helper class to build a Boltzmann Generator. Takes care of initializing blocks and coupling layers as well as the prior"""

    def __init__(self, potential, n_blocks, n_hidden_layers, n_nodes, dimensions, dimensions_conditions=0, normal_std=1, conditions_min=0, conditions_max=1):
        """Initializes a new Boltzmann Generator builder

        Parameters
        ----------
        potential : Potential
            The potential to use in configuration space
        n_blocks : int
            Number of invertible neural blocks. Each block contains two affine coupling layers
        n_hidden_layers : int
            Number of neural network hidden layers per coupling layer
        n_nodes : int
            Number of nodes per neural network hidden layer
        dimensions : int
            Dimensionality of the data (excluding extra conditions)
        dimensions_conditions : int, optional
            The dimensions of the additional conditions passed to the RealNVP blocks, by default 0 (no additional conditions)
        normal_std : int, optional
            Standard deviation of the gaussian prior, by default 1
        conditions_min : int, optional
            The lower value that the extra conditions can take, by default 0
        conditions_max : int, optional
            The upper value that the extra conditions can take, by default 1
        """
        assert isinstance(potential, Potential), "Potential must be an instance of a class that has potential.Potential as parent class"
        self.potential = potential
        self.n_blocks = n_blocks
        self.n_hidden_layers = n_hidden_layers
        self.n_nodes = n_nodes
        self.dimensions = dimensions
        self.dimensions_conditions = dimensions_conditions
        self.normal_std = normal_std
        self.conditions_min = conditions_min
        self.conditions_max = conditions_max

    def __get_layers(self, activation_function):
        """Build the layers for a single network (e.g. s or t network). Each affine layer contains two such networks.

        Parameters
        ----------
        activation_function : Class
            PyTorch activation function class (e.g. torch.nn.ReLU, torch.nn.Tanh)

        Returns
        -------
        list    
            A list containing the generated input, output and hidden layers along with activation functions. Can be passed
            to torch.nn.Sequential to build network.
        """
        layers = [nn.Linear(self.dimensions + self.dimensions_conditions, self.n_nodes), activation_function()]

        for _ in range(self.n_hidden_layers - 1):
            layers.append(nn.Linear(self.n_nodes, self.n_nodes))
            layers.append(activation_function())

        layers.append(nn.Linear(self.n_nodes, self.dimensions + self.dimensions_conditions))

        return layers
    

    def __get_single_affine_layer(self):
        """Build a single affine layer containing two networks: One for scaling (s) and one for translation (t)

        Returns
        -------
        tuple[torch.nn.Sequential, torch.nn.Sequential]
            The affine layer, i.e., two neural networks
        """
        single_s = nn.Sequential(*self.__get_layers(nn.Tanh))
        single_t = nn.Sequential(*self.__get_layers(nn.ReLU))

        return single_s, single_t


    def __init_weights(self, module):
        """Initializes the generator's weights and biases with random numbers drawn from a normal distribution centered around 1/d,
        where d is the dimensionality of the system.

        Parameters
        ----------
        module : torch.nn.Module
            The module for which to set the weights and biases
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)

            if module.bias != None:
                module.bias.data.normal_(mean=0.0, std=0.01)
    

    def build_generator(self, cuda=True, init_weights_dimensionality=False):
        """Builds a Boltzmann Generator with the specified parameters

        Parameters
        ----------
        cuda : bool, optional
            Tries to send the generator to GPU, by default True
        init_weights_dimensionality : bool, optional
            Whether or not to initialize the weights using normal distribution centered at 1/dimensions (better convergence)
            If False, weights will be initialized with PyTorch default method

        Returns
        -------
        BoltzmannGenerator
            The generated Boltzmann Generator
        """

        # Build masks in checkerboard pattern (e.g. [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]] for dimensions=4, n_blocks=2)
        # If self.dimensions_conditions > 0, the masks will be padded with self.dimensions_conditions ones at the end in order to always
        # include them in the inputs for the s and t networks (e.g. [[0, 1, 0, 1], [1, 0, 1, 1], [0, 1, 0, 1], [1, 0, 1, 1]] for dimensions=3, 
        # n_blocks=2, dimensions_conditions = 1)
        masks = torch.zeros((self.n_blocks*2, self.dimensions + self.dimensions_conditions), dtype=torch.float32, requires_grad=False)
        masks[::2, 1::2] = 1.0
        masks[1::2, ::2] = 1.0
        masks[:, self.dimensions:] = 1.0 # Fill the condition part of the mask with ones

        # Initialize a Gaussian prior centered at zero
        prior = torch.distributions.MultivariateNormal(torch.zeros(self.dimensions), torch.eye(self.dimensions)*self.normal_std)

        # Build the neural blocks (two affine coupling layers per block)
        s_networks = []
        t_networks = []
        for _ in range(self.n_blocks*2):
            single_s, single_t = self.__get_single_affine_layer()
            s_networks.append(single_s)
            t_networks.append(single_t)

        generator = BoltzmannGenerator(s_networks, t_networks, masks, prior, self.potential, self.normal_std, self.dimensions, self.dimensions_conditions,
                                       self.conditions_min, self.conditions_max)
        if init_weights_dimensionality:
            generator.apply(self.__init_weights)

        if cuda: # Send entire model to the GPU if possible
            if not torch.cuda.is_available() and torch.cuda.device_count > 0:
                raise CudaNotAvailableError("Cuda is not available")
            generator = generator.to(torch.device("cuda"))

        return generator
    

class SampleLoader:
    """Custom data loader for samples
    """

    def __init__(self, z_samples, x_samples, batch_size, shuffle=False):
        """Initializes new sample loader

        Parameters
        ----------
        z_samples : torch.Tensor
            Latent space samples, may be None if only x_samples are to be used
        x_samples : torch.Tensor
            Configuration space samples
        batch_size : int
            Batch size to use
        shuffle : bool
            If True, the data is shuffled after each epoch, by default False
        """
        if z_samples != None and len(z_samples) != len(x_samples):
            raise ValueError("Sizes of input tensors don't match")
        if len(x_samples)%batch_size != 0:
            raise ValueError("Size of input tensors is no multiple of batch size")
        self.z_samples = z_samples
        self.x_samples = x_samples
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.n_batches = len(self.x_samples)//self.batch_size

        self.current = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current == self.n_batches:
            self.current = 0
            if self.shuffle:
                shuffle_indices = np.random.choice(len(self.x_samples), len(self.x_samples), replace=False)
                self.x_samples = self.x_samples[shuffle_indices]
                if self.z_samples != None:
                    self.z_samples = self.z_samples[shuffle_indices]
            raise StopIteration
        if self.z_samples != None:
            z_batch = self.z_samples[self.current*self.batch_size:(self.current + 1)*self.batch_size]
        x_batch = self.x_samples[self.current*self.batch_size:(self.current + 1)*self.batch_size]

        self.current += 1
        if self.z_samples != None:
            return z_batch, x_batch
        else:
            return x_batch

    def __len__(self):
        return self.n_batches

class BoltzmannGenerator(nn.Module):
    """Implements a Boltzmann Generator"""

    def __init__(self, s_networks, t_networks, masks, prior, potential, normal_std, dimensions, dimensions_conditions, conditions_min, conditions_max):
        """Initializes the Boltzmann Generator

        Parameters
        ----------
        s_networks : list
            List of the scaling networks
        t_networks : list
            List of the translational networks
        masks : torch.tensor
            Binary masks to use in forward/backward process
        prior : torch.distributions.MultivariateNormal
            The Gaussian prior to use
        potential : Object
            The potential in configuration space
        normal_std : float
            Standard deviation of Gaussian prior
        dimensions : int
            Dimensionality of the data, excluding extra conditions
        dimensions_conditions : int
            Dimensionality of the extra conditions
        conditions_min : int
            The lower value that the extra conditions can take
        conditions_max : int
            The upper value that the extra conditions can take
        """
        super().__init__()
        
        self.s_networks = nn.ModuleList(s_networks)
        self.t_networks = nn.ModuleList(t_networks)
        self.masks = nn.Parameter(masks, requires_grad=False)
        self.prior = prior
        assert isinstance(potential, Potential), "Potential must be an instance of a class that has potential.Potential as parent class"
        self.potential = potential
        self.normal_std = normal_std
        self.dimensions = dimensions
        self.dimensions_conditions = dimensions_conditions
        self.conditions_min = conditions_min
        self.conditions_max = conditions_max

        self.n_networks = len(self.masks)

    @property
    def device(self):
        return "cuda" if self.masks.device.type == "cuda" else "cpu"

    def F_zx(self, z):
        """Maps a sample from latent space to configuration space (generative direction)

        Parameters
        ----------
        z : torch.tensor
            The sample(s) in latent space. Can be either 1-dimensional for a single sample or 2-dimensional for multiple samples

        Returns
        -------
        tuple[torch.tensor, torch.tensor]
            The generated sample in configuration space along with the logarithm of the absolute value of the transformation's Jacobian
        """
        log_R_zx = 0 if z.ndim == 1 else z.new_zeros(z.shape[0])
        x = z

        for i in range(self.n_networks):
            mask = self.masks[i]
            masked_x = x*mask

            s = self.s_networks[i](masked_x) * (1 - mask)
            t = self.t_networks[i](masked_x) * (1 - mask)
            x = masked_x + (1 - mask) * torch.exp(-s)*(x - t)
            
            log_R_zx -= s.sum(dim=-1)

        return x, log_R_zx
    

    def F_xz(self, x):
        """Maps a sample from configuration space to latent space (inference)

        Parameters
        ----------
        x : torch.tensor
            The sample(s) in configuration space. Can be either 1-dimensional for a single sample or 2-dimensional for multiple samples

        Returns
        -------
        tuple[torch.tensor, torch.tensor]
            The inferred sample in latent space along with the logarithm of the absolute value of the transformation's Jacobian
        """
        log_R_xz = 0 if x.ndim == 1 else x.new_zeros(x.shape[0])
        z = x

        for i in reversed(range(self.n_networks)):
            mask = self.masks[i]
            masked_z = z*mask

            s = self.s_networks[i](masked_z) * (1 - mask)
            t = self.t_networks[i](masked_z) * (1 - mask)
            z = masked_z + (1 - mask) * (z * torch.exp(s) + t)

            log_R_xz += s.sum(dim=-1)

        return z, log_R_xz

    
    def generate_z_samples(self, n_samples):
        """Generates data in latent space by sampling prior

        Parameters
        ----------
        n_samples : int
            Number of individual samples to generate

        Returns
        -------
        torch.Tensor
            The generated samples
        """
        return self.prior.sample((n_samples, ))
    

    def __regularize_energy(self, energy, t=None, decay="none"):
        """Energy regularization

        Parameters
        ----------
        energy : torch.tensor
            The energies to be regularized
        t : float, optional
            Number between 0 and 1 indicating training progress. This determines the E_HIGH value, which
            decays in the training process to some plateau value E_HIGH_PLATEAU, by default None
        decay : str, optional
            Either "none", "linear" or "exponential". Determines the type of decay. "none" means no decay,
            i.e., E_HIGH = E_PLATEAU for all t, by default "none"

        Returns
        -------
        torch.tensor
            The regularized energies
        """
        E_MAX = 1e20
        E_HIGH_INITIAL = 1e10
        E_HIGH_PLATEAU = 1e5

        decay = decay.lower()

        if decay == "exponential":
            assert 0 <= t <= 1, "t must be between 0 and 1"
            e_high = E_HIGH_INITIAL*np.exp(-np.log(2)/0.03 * t) + E_HIGH_PLATEAU
        elif decay == "linear":
            assert 0 <= t <= 1, "t must be between 0 and 1"
            e_high = (E_HIGH_PLATEAU - E_HIGH_INITIAL)/0.3 * t + E_HIGH_INITIAL if t < 0.3 else E_HIGH_PLATEAU
        elif decay == "none":
            e_high = E_HIGH_PLATEAU
        else:
            raise ValueError("Decay must be either 'none', 'linear' or 'exponential'")

        return torch.where(energy < e_high, energy, torch.where(energy > E_MAX, e_high + np.log(E_MAX - e_high + 1), e_high + torch.log(energy - e_high + 1)))


    def compute_energy(self, sample, space, t=None, decay="none"):
        """Computes the energy of a sample in latent or configuration space

        Parameters
        ----------
        sample : torch.tensor
            The sample(s) for which to compute the energy
        space : str
            The space in which the samples live, either "latent" or "configuration"
        t : float, optional
            Number between 0 and 1 indicating training progress. This determines the E_HIGH value, which
            decays in the training process to some plateau value E_HIGH_PLATEAU, by default None
        decay : str, optional
            Either "none", "linear" or "exponential". Determines the type of decay. "none" means no decay,
            i.e., E_HIGH = E_PLATEAU for all t, by default "none"

        Returns
        -------
        torch.tensor
            The computed energies
        """

        space = space.lower()
        if space == "configuration":
            return self.__regularize_energy(self.potential.evaluate(sample), t, decay)
        elif space == "latent":
            return 1/(2*self.normal_std**2) * torch.norm(sample[..., :self.dimensions], dim=-1)**2 # Energy of normal prior
        else:
            raise ValueError("Space must be either 'latent' or 'configuration'")
        

    def get_weights(self, x, log_R_zx, z, normalize=False):
        """Get the weights for a given sample that can be used for re-weighting into Boltzmann distribution

        Parameters
        ----------
        x : torch.tensor
            The samples for which to compute the weights in configuration space
        log_R_zx : torch.tensor
            The logarithm of the absolute value of the Jacobian of the transformation between latent and configuration space
        z : torch.tensor
            The corresponding points in latent space
        normalize : bool, optional
            Whether to normalize the weights such that they sum to 1, by default False

        Returns
        -------
        torch.tensor
            The weights for each sample
        """
        weights = torch.exp(-self.potential.evaluate(x) - self.prior.log_prob(z[..., :self.dimensions].detach().cpu()).to(x.device) + log_R_zx)
        weights = torch.where(weights > 1e10, 0, weights) # Prevent overflow, is there a better solution?
        return weights / torch.sum(weights) if normalize else weights
    

    def ress(self, x, log_R_zx, z):
        """Computes the relative effective sample size given a sample generated by the generator

        Parameters
        ----------
        x : torch.Tensor
            The samples in sample space for which to compute the relative effective sample size
        log_R_zx : torch.Tensor
            The logarithm of the absolute value of the Jacobian of the transformation between latent and configuration space
        z : torch.Tensor
            The points in latent space corresponding to the points in configuration space

        Returns
        -------
        float
            The relative effective sample size
        """
        weights = self.get_weights(x, log_R_zx, z, normalize=True)
        return ress(weights)


    def train(self, batch_size, z_samples = None, x_samples = None, epochs = 1, lr = 1e-5, w_KL = 1.0, w_ML = 1.0, n_out = 10, out_loss = False, decay="none", 
              push_to_gpu=False, pbar=None, conditions_per_batch=None, weight_decay=0.0):
        """Train the Boltzmann generator

        Parameters
        ----------
        batch_size : int
            Batch size to use for training
        z_samples : torch.Tensor, optional
            The number of samples to use for training by energy. If None, training will only be done by example, by default None
        x_samples : torch.Tensor, optional
            The samples in configuration space. If None, training will only be done by energy, by default None
        epochs : int, optional
            Number of epochs to train for, by default 1
        lr : float, optional
            Learning rate to pass to Adam optimizer, by default 1e-4
        w_KL : float, optional
            The weight to assign to the KL loss term (training by energy), by default 1.0
        w_ML : float, optional
            The weight to assign to the ML loss term (training by example), by default 1.0
        n_out : int, optional
            Output frequency of training progress to console. If 0, no output will be shown, by default 10
        out_loss : bool, optional
            Whether or not to return a trajectory for the loss function, by default False
        decay : str, optional
            Either "none", "linear" or "exponential". Determines the type of energy regularization to use, i.e., in what way the energy bound is to decay
            to a plateau value during training, by default "none"
        push_to_gpu : bool, optional
            Whether to push the entire data set to the GPU before training (increases performance, but be careful of memory), False by default
        pbar : tqdm.tqdm | list[tqdm.tqdm], optional
            tqdm object or list of tqdm objects to update after each epoch. Total must be set to the number of epochs specified. If None, no
            updates are done, by default None.
        conditions_per_batch : int, optional
            Determines how many unique extra conditions should be generated per batch. If None, conditions_per_batch = batch_size, by default None
        weight_decay : float, optional
            The value for the weigth decay to pass to the Adam optimizer, by default 0.0
            
        Returns
        -------
        torch.tensor
            The loss function trajectory if out_loss is True
        """
        assert z_samples == None or isinstance(z_samples, int), "z_samples must be of type int"
        assert x_samples == None or isinstance(x_samples, torch.Tensor), "x_samples must be of type torch.Tensor"
        assert not (z_samples == None and x_samples == None), "At least one of (z_samples, x_samples) must be given"
        assert self.device == "cuda" if push_to_gpu else True, "Cannot push to GPU if device is not CUDA"

        n_samples = len(x_samples) if x_samples != None else z_samples
        n_batches = n_samples//batch_size

        if conditions_per_batch == None:
            conditions_per_batch = batch_size
            assert batch_size%conditions_per_batch == 0, "Batch size must be a multiple of conditions per batch"

        if z_samples != None:
            assert z_samples%batch_size == 0, "z_samples is no multiple of batch size"

        optim = torch.optim.Adam([param for param in self.parameters() if param.requires_grad], lr=lr, weight_decay=weight_decay) # Also used by Noé

        if z_samples == None:
            w_KL = 0

        if x_samples == None:
            x_samples = torch.empty((n_batches, batch_size, z_samples))
            w_ML = 0

        if out_loss or n_out:
            loss_traj = torch.empty((epochs, n_batches), requires_grad=False, device=self.device)

        if pbar == None:
            pbar = []
        elif not isinstance(pbar, (list, tuple)):
            pbar = [pbar]

        if n_out:
            print("Epoch\tLoss")

        decay = decay.lower()
    
        if push_to_gpu:
            x_samples = x_samples.to("cuda")
            samples = SampleLoader(None, x_samples, batch_size, shuffle=True)
        else:
            samples = DataLoader(x_samples, batch_size, num_workers=8, pin_memory=True, shuffle=True)

        total_t = epochs*n_batches - 1
        for epoch in range(epochs):
            batch = 0
            for x_batch in samples:
                
                if decay != "none":
                    t = (batch + epoch*n_batches)/total_t
                else:
                    t = None
                
                if self.device == "cuda" and not push_to_gpu:
                    x_batch = x_batch.to(self.device)

                if w_KL != 0:

                    z_batch = self.generate_z_samples(batch_size)
                    conditions = torch.rand([conditions_per_batch, self.dimensions_conditions], dtype=torch.float32)*(self.conditions_max - self.conditions_min) + self.conditions_min
                    conditions = conditions.repeat(batch_size//conditions_per_batch, 1)

                    z_batch = torch.hstack([z_batch, conditions]).to(self.device)

                    x, log_R_zx = self.F_zx(z_batch)
                    u = self.compute_energy(x, space="configuration", decay=decay, t=t)
                    J_KL = torch.mean(u - log_R_zx)
                else:
                    J_KL = 0
                    
                if w_ML != 0:
                    z, log_R_xz = self.F_xz(x_batch)
                    u = self.compute_energy(z, space="latent")
                    J_ML = torch.mean(u - log_R_xz)
                else:
                    J_ML = 0
                
                loss = w_KL*J_KL + w_ML*J_ML

                optim.zero_grad()
                loss.backward()
                optim.step()

                if out_loss:
                    loss_traj[epoch, batch] = loss
                batch += 1

            if n_out and (epoch+1) % n_out == 0:
                print(f"{epoch+1}\t{loss_traj[epoch].mean().item()}")

            for bar in pbar:
                bar.update()
                bar.refresh()

        if out_loss:
            return loss_traj
        