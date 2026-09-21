import numpy as np
import torch
from multiprocessing import get_context
import copy
import json
from zipfile import ZipFile
import os
from io import BytesIO
from pathlib import Path

from torch import nn
from tqdm.auto import tqdm

import time

import potential
from generator import SampleLoader, GeneratorBuilder
from tps import generate_path_to_state_brownian, generate_path_to_state_mcmc_polymer
from md import brownian_propagate
from util import graceful_keyboard_interrupt_class
from mcmc import polymer_biased_mcmc_2D

MP_CONTEXT = get_context("fork")


class AIMMDNetwork(nn.Sequential):
    """Simple feed forward network with Tanh activation functions after each hidden layer and optional
    sigmoid function in output layer. Models the committor of a given system.
    """

    def __init__(self, n_input, n_hidden, n_nodes, sigmoid=False, init_weights_zero=False, transform_layers=[], cv_function_without_sigmoid=False):
        """Init function of the AIMMDNetwork class. Initializes the network with given parameters.

        Parameters
        ----------
        n_input : int
            Number of input nodes
        n_hidden : int
            Number of hidden layers
        n_nodes : int | list[int] | tuple [int]
            Number of nodes per hidden layer. If int, all hidden layers have the same number of nodes. If list of ints
            or tuple of ints, length must be equal to n_hidden and each entry corresponds to the number of nodes in
            that layer.
        sigmoid : bool, optional
            If True, sigmoid activation function is automatically applied in output layer, by default False
        init_weights_zero : bool, optional
            If True, the last hidden layer is initialized with weights and biases equal to zero, resulting in the network
            initially outputting 0.5 for every input (after sigmoid is applied, either automatically or manually), 
            by default False
        transform_layers : torch.nn.ModuleList, optional
            The transform layers that should be applied to the input in the xz direction before going into the AIMMD network
        cv_function_without_sigmoid : bool, optional
            If this is True, the cv_function and cv_gradient_function will not apply a sigmoid function, even if sigmoid is set to True, by default False
        """

        if not isinstance(n_nodes, (list, tuple)):
            assert isinstance(n_nodes, int), "n_nodes must be an integer or a list or a tuple"
            n_nodes = [n_nodes for _ in range(n_hidden)]
        
        assert len(n_nodes) == n_hidden, "Length of n_nodes must match n_hidden"

        n_nodes.append(1)

        layers = [transform_layer for transform_layer in transform_layers]
        layers += [nn.Linear(n_input, n_nodes[0])]
        for i in range(n_hidden):
            layers.append(nn.Tanh())
            layers.append(nn.Linear(n_nodes[i], n_nodes[i+1]))

        if sigmoid:
            layers.append(nn.Sigmoid())

        super().__init__(*layers)

        if init_weights_zero:
            nn.init.constant_(self[-1 - sigmoid].weight, 0)
            nn.init.constant_(self[-1 - sigmoid].bias, 0)

        n_nodes.pop(-1)
        self.n_nodes = n_nodes
        self.n_hidden = n_hidden
        self.n_input = n_input
        self.sigmoid = sigmoid
        self.cv_function_without_sigmoid = cv_function_without_sigmoid

    def apply_sigmoid(self, output, warn_auto_applied=True):
        """Manually applies sigmoid activation function. Only to be used if object was initialized with sigmoid=False

        Parameters
        ----------
        output : torch.Tensor
            Output of the network before sigmoid is applied
        warn_auto_applied : bool, optional
            If True, a warning is printed if the network auto-applies a sigmoid in the last layer and this method is
            called, aiming to prevent accidental multiple application of the function, by default True

        Returns
        -------
        torch.Tensor
            The resulting output with sigmoid activation function applied
        """
        if warn_auto_applied and self.sigmoid:
            print("WARNING: Applying sigmoid although it is the last activation function is this network")
        return torch.sigmoid(output)

    def forward_with_sigmoid(self, input):
        """Feeds given input through network and applies sigmoid at the end. Only to be used if object was 
        initialized with sigmoid=False

        Parameters
        ----------
        input : torch.Tensor
            Input coordinates to be passed through the network

        Returns
        -------
        torch.Tensor
            Resulting output with applied sigmoid
        """
        return self.apply_sigmoid(self(input), True)

    def cv_function(self, coords, squeeze=True):
        """Wrapper function for the network. Takes care of proper output shape and dtype. Passes given input
        through network, applies sigmoid if necessary and returns same dtype as input array.

        Parameters
        ----------
        coords : torch.Tensor | np.ndarray
            Input coordinates to be passed through the network
        squeeze : bool, optional
            If True, the output is squeezed into a 1D array, by default True

        Returns
        -------
        torch.Tensor | np.ndarray
            Output of the network with applied sigmoid function. Data type is the same as the data type of coords.
        """
        if isinstance(coords, torch.Tensor):
            output = self(coords) if (self.sigmoid or self.cv_function_without_sigmoid) else self.apply_sigmoid(self(coords))
            return output.squeeze(axis=-1) if squeeze else output
        elif isinstance(coords, np.ndarray):
            output = self(torch.tensor(coords, dtype=torch.float32).to(next(self.parameters()).device))
            if not (self.sigmoid or self.cv_function_without_sigmoid):
                output = self.apply_sigmoid(output)
            output = output.detach().cpu().numpy()
            return output.squeeze(axis=-1) if squeeze else output
    
    def cv_gradient_function(self, coords):
        """Calculates the gradient of the network output with respect to the input coordinates by applying PyTorch's autograd function.

        Parameters
        ----------
        coords : torch.Tensor | np.ndarray
            Coordinates for which to calculate the gradient

        Returns
        -------
        torch.Tensor | np.ndarray
            Calculates gradients. Data type is the same as the data type of coords.
        """
        np_array = isinstance(coords, np.ndarray)
        if np_array:
            coords = torch.tensor(coords, dtype=torch.float32, device=next(self.parameters()).device, requires_grad=True)
        else:
            coords = coords.detach()

        output = self.cv_function(coords, squeeze=False)

        grad = torch.autograd.grad(output, coords, grad_outputs=torch.ones_like(output), create_graph=True)[0]

        if np_array:
            grad = grad.detach().cpu().numpy()

        return grad
    

class AIMMDCommittorTrainer:
    """Implementation of a training algorithm that simultaneously trains an AIMMD committor model and conditions a Boltzmann Generator on generating
    shooting points at arbitrary committor bias centers.
    """

    def __init__(self, aimmd_network, generator, committor_potential, committor_q_A, committor_q_B, committor_target, md_potential, md_timestep, 
                 md_diffusion_coeff, order_param, order_param_kwargs={}, committor_timestep=1e-3, committor_diffusion_coeff=1.0, mcmc_max_displacement=0.4):
        """Initializes the trainer class

        Parameters
        ----------
        aimmd_network : AIMMDNetwork
            The AIMMD network object that will be trained
        generator : generator.BoltzmannGenerator
            The Boltzmann Generator that will be trained
        committor_potential : potential.Potential
            The potential used to estimate the committor using fleeting trajectories (should be unbiased)
        committor_q_A : float
            Value of the order parameter below which samples are regarded as being in stable state A
        committor_q_B : float
            Value of the order parameter above which samples are regarded as being in stable state B
        committor_target : str
            Target for the committor, einer "A" or "B"
        md_potential : potential.Potential
            Potential in which short MD simulations are done in order to relax generated samples during training (should be biased)
        md_timestep : float
            Timestep to be used in short MD simulations
        md_diffusion_coeff : float
            Diffusion coefficient to be used in short MD simulations
        order_param : callable
            Order parameter of the system. Should be callable that takes one or more configurations and returns their order parameter
        order_param_kwargs : dict, optional
            Keyword arguments to pass to the order parameter function if needed, by default {}
        committor_timestep : float, optional
            Timestep for the fleeting trajectories used to train committor model, by default 1e-3
        committor_diffusion_coeff : float, optional
            Diffusion coefficient for the fleeting trajectories used to train committor model, by default 1.0
        mcmc_max_displacement : float, optional
            Max displacement for trial moves in MCMC propagation of samples (if this is chosen over md), by default 0.4
        """

        self.aimmd_network = aimmd_network
        self.generator = generator
        self.committor_potential = committor_potential
        self.committor_q_A = committor_q_A
        self.committor_q_B = committor_q_B
        self.committor_target = committor_target
        self.md_potential = md_potential
        self.md_timestep = md_timestep
        self.md_diffusion_coeff = md_diffusion_coeff
        self.order_param = order_param
        self.order_param_kwargs = order_param_kwargs
        self.committor_timestep = committor_timestep
        self.committor_diffusion_coeff = committor_diffusion_coeff
        self.mcmc_max_displacement = mcmc_max_displacement
        
        self.processes = []
        
        
    def __del__(self):
        """Makes sure that the sub-processes are killed when the object is destroyed
        """
        self._kill_processes()

    @property
    def device(self):
        """Returns the device on which the training of the AIMMD committor model and the Boltzmann Generator is done

        Returns
        -------
        str
            Either "cpu" or "cuda", depending on device
        """
        return self.generator.device
    
    def _spawn_processes(self, n_processes, committor_max_steps, x_sample_beta):
        """Spawns a given number of sub-processes that await necessary data in order to start fleeting trajectories and record their
        outcome. Used to train the AIMMD committor model.

        Parameters
        ----------
        n_processes : int
            Number of sub-processes to spawn
        committor_max_steps : int
            Cutoff value for the fleeting trajectories. If a trajectory takes more timesteps than this value, it is considered to not have
            reached the target state.
        x_sample_beta : float
            The beta value corresponding to the points to propagate
        """
        self._kill_processes()

        committor_potential = self._committor_potential_on_cpu()

        for _ in range(n_processes):
            parent_conn, child_conn = MP_CONTEXT.Pipe()
            process = MP_CONTEXT.Process(target=self._run_fleeting_trajectory_job, args=(child_conn, committor_max_steps, x_sample_beta, committor_potential))

            process.start()
            self.processes.append([process, parent_conn, True])

    def _committor_potential_on_cpu(self):
        """Creates the copy of the committor potential that the sub-processes propagate in. The collective variable of the committor
        potential is the committor model, which lives on the gpu during training. Moving the copy to the cpu keeps the sub-processes
        from touching cuda, which a forked process cannot do and which would need one cuda context per sub-process.

        Returns
        -------
        potential.Potential
            A copy of the committor potential whose collective variable is evaluated on the cpu
        """
        committor_potential = copy.deepcopy(self.committor_potential)

        for attribute in ["cv_function", "cv_gradient_function"]:
            owner = getattr(getattr(committor_potential, attribute, None), "__self__", None)
            if isinstance(owner, nn.Module):
                owner.to("cpu")

        return committor_potential

    def _kill_processes(self):
        """Kills all sub-processes spawned by this object
        """
        for state in self.processes:
            state[0].terminate()
        self.processes = []

    @graceful_keyboard_interrupt_class()
    def _run_fleeting_trajectory_job(self, child_conn, committor_max_steps, x_sample_beta, committor_potential):
        """Runs fleeting trajectories from given initial conditions and records their outcome (which states they have reached first) in a
        two-way shooting move. This method is run by the sub-processes spawned in _spawn_processes and communicates with the main process.

        Parameters
        ----------
        child_conn : multiprocessing.connection.Connection
            Child connection used to receive/send initial conditions/outcomes from/to the main process
        committor_max_steps : int
            Cutoff value for the fleeting trajectories. If a trajectory takes more timesteps than this value, it is considered to not have
            reached the target state.
        x_sample_beta : float
            The beta value corresponding to the points to propagate
        committor_potential : potential.Potential
            The copy of the committor potential this sub-process propagates in, see _committor_potential_on_cpu
        """
        child_conn.send("ready")
        while True:
            if not child_conn.poll():
                time.sleep(0.01)
                continue
            batch = child_conn.recv()
            outcomes = np.zeros((len(batch), 2), dtype=float)
            for i, point in enumerate(batch):
                for trial in range(2):
                    outcome = generate_path_to_state_brownian(point[1:], self.order_param, self.committor_q_A, self.committor_q_B, committor_potential, 
                                                                self.committor_timestep, self.committor_diffusion_coeff, self.committor_target, self.order_param_kwargs, 
                                                                max_steps=committor_max_steps, beta=x_sample_beta)
                    outcomes[i, trial] = 2*(not outcome) - 1 # Path enters target first -> -1, path enters other state first -> 1
            child_conn.send([batch[:, 0], outcomes])

    def _fleeting_trajectories(self, n_samples, x_samples, committor_batch_size, n_out, current_pbar, total_pbar):
        """Handles the fleeting trajectories by spawning sub-processes, communicating with them and recording outcomes.

        Parameters
        ----------
        n_samples : int
            Number of samples in data set
        x_samples : torch.Tensor
            Samples in sample space from which to start fleeting trajectories in a two-way shooting move
        committor_batch_size : int
            Size of batches of initial conditions to send to each sub-process
        n_out : int
            Output frequency of training to console
        current_pbar : tqdm.tqdm
            Progress bar for the current training state (committor estimation)
        total_pbar : tqdm.tqdm
            Progress bar for the total training progress

        Returns
        -------
        np.ndarray
            Outcomes as a 2D NumPy array, where each row corresponds to the two-way shooting outcomes of each entry in x_samples. Shooting
            outcomes are either -1 if the trajectory reached the target state or 1 otherwise.
        """
        outcomes = np.empty((n_samples, 2))
        shooting_points = np.hstack([np.arange(n_samples)[..., None], x_samples[..., :-1].detach().cpu().numpy()])

        n_batches_total = int(np.ceil(n_samples/committor_batch_size))
        batches_done = 0
        next_batch = 0

        while batches_done < n_batches_total:
            for process_index, state in enumerate(self.processes):
                process, parent_conn, is_idle = state
                if parent_conn.poll():
                    signal = parent_conn.recv()
                    if isinstance(signal, list):
                        batches_done += 1
                        shooting_point_indices, outcome = signal
                        outcomes[shooting_point_indices.astype(int)] = outcome
                        is_idle = True
                        if n_out:
                            if batches_done < n_batches_total:
                                current_pbar.update(committor_batch_size)
                                total_pbar.update(committor_batch_size)
                                current_pbar.refresh()
                            else:
                                current_pbar.update(n_samples - (n_batches_total - 1)*committor_batch_size)
                                total_pbar.update(n_samples - (n_batches_total - 1)*committor_batch_size)
                                current_pbar.refresh()
                if next_batch < n_batches_total and is_idle:
                    if next_batch != n_batches_total - 1:
                        batch = shooting_points[next_batch*committor_batch_size:(next_batch + 1)*committor_batch_size]
                    else:
                        batch = shooting_points[next_batch*committor_batch_size:]
                    parent_conn.send(batch)
                    next_batch += 1
                    is_idle = False
                self.processes[process_index][2] = is_idle
            time.sleep(0.1)

        return outcomes
    
    def _estimate_aimmd_efficiency(self, outcomes, x_samples):
        """Estimates the efficiency factor of the AIMMD training by comparing currently predicted transitions with actually observed transitions.
        The higher this value is, the more the AIMMD network is in need of further training. This can be used in order to avoid overfitting.

        Parameters
        ----------
        outcomes : np.ndarray
            Observed outcomes of two-way shooting moves (the output from _fleeting_trajectories)
        x_samples : torch.Tensor
            The samples in sample space from which the fleeting trajectories were started

        Returns
        -------
        float
            The efficiency factor (between 0 and 1)
        """
        current_p_B = self.aimmd_network.forward_with_sigmoid(x_samples).detach().cpu().numpy().flatten()
        n_TPs_expected = 2*np.sum((1 - current_p_B)*current_p_B)
        n_TPs_generated = np.sum(1 - (np.abs(outcomes[:, 0] + outcomes[:, 1]))/2)

        return min(1, (1 - n_TPs_generated/n_TPs_expected)**2)

    def _train_aimmd(self, n_epoch_aimmd, outcomes, x_samples, batch_size, optim, n_out, current_pbar, total_pbar):
        """Handles the AIMMD committor model training by training on observed fleeting trajectory outcomes using "coin flip" loss function

        Parameters
        ----------
        n_epoch_aimmd : int
            Number of epochs to train the AIMMD model for
        outcomes : torch.Tensor
            Observed outcomes from fleeting trajectories (output of _fleeting_trajectories)
        x_samples : torch.Tensor
            The samples in sample space from which the fleeting trajectories were started
        batch_size : int
            Batch size for the training of the network
        optim : torch.optim.Optimizer
            Optimizer to be used in training
        n_out : int
            Training progress output frequency to console
        current_pbar : tqdm.tqdm
            Progress bar of current training state (training AIMMD network)
        total_pbar : tqdm.tqdm
            Progress bar of total training progress

        Returns
        -------
        list[float]
            Losses of the last training epoch
        """
        sample_loader = SampleLoader(torch.column_stack([x_samples, x_samples]).reshape((2*len(x_samples), -1)).detach(), outcomes, batch_size*2, shuffle=True)
        losses = []
        for _ in range(n_epoch_aimmd):
            losses = []
            for x_sample, outcome in sample_loader:
                loss = torch.sum(torch.log(1 + torch.exp(outcome[..., None] * self.aimmd_network(x_sample))))
                optim.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optim.step()
                if n_out:
                    current_pbar.update()
                    total_pbar.update()
                    current_pbar.refresh()

        return losses

    def _run_md(self, x_samples, md_steps, n_out, current_pbar, total_pbar, x_sample_beta):
        """Propagates given samples in sample space within the md_potential for a given number of timesteps in order to relax
        them in bias potential

        Parameters
        ----------
        x_samples : torch.Tensor
            Samples to propagate
        md_steps : int
            Number of timesteps to propagate for
        n_out : int
            Training output frequency to console
        current_pbar : tqdm.tqdm
            Progress bar of current training state (propagating samples)
        total_pbar : tqdm.tqdm
            Progress bar for total progress
        x_sample_beta : float
            The beta values corresponding to the x_samples

        Returns
        -------
        torch.Tensor
            The propagated samples
        """
        x_samples = x_samples.detach().cpu().numpy()

        for _ in range(md_steps):
            current_forces = self.md_potential.evaluate_force(x_samples)
            x_samples[:, :-1] = brownian_propagate(x_samples[:, :-1], current_forces, self.md_timestep, self.md_diffusion_coeff, beta=x_sample_beta)

            if n_out:
                current_pbar.update()
                total_pbar.update()
                current_pbar.refresh()

        return torch.tensor(x_samples, dtype=torch.float32, device=self.generator.device)
    

    def _run_mcmc(self, x_samples, md_steps, n_out, current_pbar, total_pbar, x_sample_beta):
        """Propagates given samples in sample space within the md_potential for a given number of timesteps in order to relax
        them in bias potential using mcmc. This currently only works for the 2D polymer potential.

        Parameters
        ----------
        x_samples : torch.Tensor
            Samples to propagate
        md_steps : int
            Number of timesteps to propagate for
        n_out : int
            Training output frequency to console
        current_pbar : tqdm.tqdm
            Progress bar of current training state (propagating samples)
        total_pbar : tqdm.tqdm
            Progress bar for total progress
        x_sample_beta : float
            The beta values corresponding to the x_samples

        Returns
        -------
        torch.Tensor
            The propagated samples
        """
        if not isinstance(self.md_potential, potential.Polymer):
            raise NotImplementedError("MCMC propagation currently only works for 2D polymer potential")
        
        x_samples = x_samples.detach().cpu().numpy()
        configs, bias_centers = x_samples[..., :-1], x_samples[..., -1]

        configs, _ = polymer_biased_mcmc_2D(configs, self.md_potential, n_samples=0, bias_center=bias_centers, enhanced_sampling=True, 
                                            max_displacement=self.mcmc_max_displacement, beta=x_sample_beta, n_out=0, n_equil=md_steps,
                                            progress=[current_pbar, total_pbar] if n_out else False)

        return torch.tensor(np.hstack([configs, bias_centers[..., None]]), dtype=torch.float32, device=self.generator.device)

    @graceful_keyboard_interrupt_class(_kill_processes, True, "\n\nAborting training.")
    def train(self, initial_x_samples, n_processes, n_cycles, generator_training_scheduler, n_epoch_aimmd, aimmd_batch_size, aimmd_efficiency_threshold,
            committor_batch_size, md_steps, pool_size, conditions_min, conditions_max, initial_outcomes=None, aimmd_lr=1e-4, n_out=1, committor_max_steps=100000, n_out_samples=None, 
            out_samples_bias_centers=np.arange(0.0, 1.1, 0.1), weight_decay_aimmd=0.0, x_sample_beta=1.0, propagation_method="md", preprocess_training_data=None):
        """Trains the AIMMD committor model and the Boltzmann Generator simultaneously in a self-consistent way

        Parameters
        ----------
        initial_x_samples : torch.Tensor
            Initial samples in sample space to use for training (without conditions)
        n_processes : int
            Number of sub-processes to spawn for committor estimation
        n_cycles : int
            Number of total training cycles to perform
        generator_training_scheduler : generator.GeneratorTrainingScheduler
            The training scheduler for the Boltzmann Generator
        n_epoch_aimmd : int
            Number of epochs per cycle to train the AIMMD network for
        aimmd_batch_size : int
            Batch size for the AIMMD training
        aimmd_efficiency_threshold : float
            Threshold value below which AIMMD training is skipped for the current cycle
        committor_batch_size : int
            Size of batches of initial conditions that the sub-processes work on at the same time in MD propagation
        md_steps : int
            Number of MD steps to propagate generated samples for in each cycle
        pool_size : int
            Size of the pool of training data, given in cycles. For example, if the pool has size 3, then the training data from
            the current cycle and the two cycles before will be used in training. Minimum is 1.
        conditions_min : float
            The minimum value for the conditions to generate
        conditions_max : float
            The maximum value for the conditions to generate
        initial_outcomes : np.ndarray, optional
            Initial outcomes of two-way shooting moves for given initial_samples. If this is None, the fleeting trajectories will also
            be run for the initial data in order to obtain outcome data for AIMMD training, by default None
        aimmd_lr : float, optional
            Learning rate for the AIMMD training, by default 1e-4
        n_out : int, optional
            Output frequency to console of current training progress/stats. If this is 0, there will be no output, by default 1
        committor_max_steps : int, optional
            Cutoff value for the fleeting trajectories. If a trajectory takes more timesteps than this value, it is considered to not have
            reached the target state, by default 100000
        n_out_samples : int, optional
            If this is set to a positive integer, the specified number of samples in sample space will be generated and stored after each cycle,
            enabling analysis of training progress after the training, by default None
        out_samples_bias_centers : Iterable, optional
            Bias centers to generate samples (described in out_samples) at, by default np.arange(0.0, 1.1, 0.1)
        weight_decay_aimmd : float, optional
            The weight decay value to pass to Adam optimizer for the training of the AIMMD committor model, by default 0.0
        x_sample_beta : float, optional
            The value for beta at which to generate new x_samples
        propagation_method : str, optional
            Either "md" or "mcmc", determines the method by which samples are propagated. "mcmc" currently only works for the 2D Polymer model,
            by default "md"
        preprocess_training_data : callable, optional
            A callable that takes the training data to be used to train the generator (the x_samples), pre-processes it in some way and returns the
            pre-processed data back. It will be called directly before generator training starts, by default None

        Returns
        -------
        torch.Tensor
            If n_out_samples is set, this is the 4-dim array of generated samples with
            shape (len(out_samples_bias_centers), n_cycles, n_out_samples, initial_x_samples.shape[-1] + 1)
        np.ndarray
            If n_out_samples is set, this is the array of weights corresponding to the samples generated at the end of each cycle,
            computed using the currently available committor model. This means that these weights reflect the instantaneous training
            status of the generator.
        np.ndarray
            If n_out_samples is set, this is the array of bias centers used to generate the samples after each cycle
        """
        
        assert pool_size > 0, "Pool size must be at least 1 in order to have training data"
        n_samples = len(initial_x_samples)
        n_samples_in_pool = n_samples * pool_size

        if propagation_method == "md":
            propagation_function = self._run_md
        elif propagation_method == "mcmc":
            propagation_function = self._run_mcmc
        else:
            raise ValueError("Propagation method must be either \"md\" or \"mcmc\"")

        self.aimmd_network = self.aimmd_network.to(self.device)

        x_samples = torch.zeros((n_samples_in_pool, initial_x_samples.shape[-1] + 1), dtype=torch.float32, device=self.device)
        x_samples[:n_samples, :-1] = initial_x_samples

        z_samples = n_samples

        outcomes = np.empty((n_samples_in_pool, 2))
        if initial_outcomes is not None:
            outcomes[:n_samples] = initial_outcomes

        current_pool_start = 0
        current_pool_end = n_samples
        pool_filled = n_samples

        optim = torch.optim.Adam([param for param in self.aimmd_network.parameters() if param.requires_grad], lr=aimmd_lr, weight_decay=weight_decay_aimmd)
        
        if n_out:
            current_pbar = tqdm(total=1, position=0, desc="Initializing", leave=True, smoothing=0.1)
            pbar_total_aimmd_steps = n_samples_in_pool*(n_cycles - (pool_size - 1))
            for i in range(1, pool_size):
                pbar_total_aimmd_steps += i*n_samples
            pbar_total_aimmd_steps //= aimmd_batch_size
            pbar_total_aimmd_steps *= n_epoch_aimmd
            total_pbar = tqdm(total=n_cycles*(2*md_steps + generator_training_scheduler.n_epochs_total) + pbar_total_aimmd_steps + (n_cycles*n_samples if initial_outcomes is None else (n_cycles - 1)*n_samples), smoothing=0, position=1, desc="Total progress", leave=True)
            tqdm.write(f"Running for {n_cycles} cycles, output frequency is {n_out}\n")
            tqdm.write(f"Cycle\t\tAIMMD efficiency\tAIMMD loss\t\tGenerator losses (per task)")

        self._spawn_processes(n_processes, committor_max_steps, x_sample_beta)


        if n_out_samples:
            out_samples = torch.empty((len(out_samples_bias_centers), n_cycles, n_out_samples, x_samples.shape[-1]), requires_grad=False)
            out_weights = np.empty((len(out_samples_bias_centers), n_cycles, n_out_samples))

        for cycle in range(n_cycles):

            if n_out:
                total_pbar.set_description(f"Total progress ({cycle + 1}/{n_cycles})")
                if ((cycle + 1) % n_out == 0) or (cycle == n_cycles - 1):
                    tqdm.write(f"{cycle + 1}\t\t", end="")
                if cycle > 0 or initial_outcomes is None:
                    current_pbar.set_description_str("Estimating committors")
                    current_pbar.reset(n_samples)
                    current_pbar.refresh()
            
            if cycle > 0 or initial_outcomes is None:
                outcomes[current_pool_start:current_pool_end] = self._fleeting_trajectories(n_samples, x_samples[current_pool_start:current_pool_end], committor_batch_size, n_out, current_pbar, total_pbar)

            x_samples_for_aimmd = x_samples[:pool_filled, :-1] # Cut away conditions

            current_efficiency = self._estimate_aimmd_efficiency(outcomes[:pool_filled], x_samples_for_aimmd)

            if n_out and (((cycle + 1) % n_out == 0) or (cycle == n_cycles - 1)):
                tqdm.write(f"{current_efficiency:0.6f}\t\t", end="")

            if current_efficiency > aimmd_efficiency_threshold:
                if n_out:
                    current_pbar.set_description_str("Training AIMMD")
                    current_pbar.reset(n_epoch_aimmd*pool_filled//aimmd_batch_size)
                    current_pbar.refresh()

                outcomes_aimmd = torch.tensor(outcomes[:pool_filled].flatten(), dtype=torch.float32, device=x_samples.device)

                losses = self._train_aimmd(n_epoch_aimmd, outcomes_aimmd, x_samples_for_aimmd, aimmd_batch_size, optim, n_out, current_pbar, total_pbar)

            elif n_out:
                total_pbar.update(n_epoch_aimmd*pool_filled//aimmd_batch_size) # Make sure the total progress bar is updated with the skipped AIMMD training
            
            if n_out and (((cycle + 1) % n_out == 0) or (cycle == n_cycles - 1)):
                if current_efficiency > aimmd_efficiency_threshold:
                    tqdm.write(f"{np.mean(losses):0.6e}\t\t", end="")
                else:
                    tqdm.write(f"skipped (eff)\t\t", end="")

            if not self.aimmd_network.cv_function_without_sigmoid: # This is a dirty hack for if we want to use q(x) (pre-sigmoid log-committor) as a bias coordinate
                x_samples[:pool_filled, -1] = torch.flatten(self.aimmd_network.forward_with_sigmoid(x_samples_for_aimmd)).detach()
            else:
                x_samples[:pool_filled, -1] = torch.round(torch.flatten(self.aimmd_network(x_samples_for_aimmd))).detach()

            if n_out and md_steps > 0:
                current_pbar.set_description_str("Propagating samples")
                current_pbar.reset(md_steps)
                current_pbar.refresh()
            
            x_samples[:pool_filled] = propagation_function(x_samples[:pool_filled], md_steps, n_out, current_pbar, total_pbar, x_sample_beta)

            if preprocess_training_data is not None:
                with torch.no_grad():
                    x_samples[:pool_filled, :-1] = preprocess_training_data(x_samples[:pool_filled, :-1])

            if n_out:
                current_pbar.set_description_str("Training generator")
                current_pbar.reset(generator_training_scheduler.n_epochs_total)
                current_pbar.refresh()

            def pre_task_hook(task_number):
                current_pbar.set_description_str(f"Training generator ({task_number+1}/{len(generator_training_scheduler)})")
                current_pbar.refresh()

            generator_losses = generator_training_scheduler.train(self.generator, z_samples=z_samples, x_samples=x_samples[:pool_filled].detach(), 
                                                                  pre_task_hook=pre_task_hook, pbar=[current_pbar, total_pbar], out_losses=True)

            if n_out:
                if ((cycle + 1) % n_out == 0) or (cycle == n_cycles - 1):
                    for generator_loss in generator_losses:
                        tqdm.write(f"{torch.mean(generator_loss[-1]).item():0.3e} ", end="")
                    tqdm.write("\n", end="")
            
            if generator_losses is not None:
                for generator_loss in generator_losses:
                    del generator_loss
                del generator_losses

            current_pool_start += n_samples
            current_pool_end += n_samples
            current_pool_start %= n_samples_in_pool
            current_pool_end %= n_samples_in_pool
            if current_pool_end == 0:
                current_pool_end = n_samples_in_pool

            if pool_filled < n_samples_in_pool:
                pool_filled += n_samples

            new_conditions = torch.round(torch.rand(z_samples, dtype=torch.float32, device=self.generator.device) * (conditions_max - conditions_min) + conditions_min)
            current_z_samples = torch.hstack([self.generator.generate_z_samples(z_samples, betas=x_sample_beta), new_conditions[..., None]])
            x_samples[current_pool_start:current_pool_end] = self.generator.F_zx(current_z_samples)[0]
            x_samples = x_samples.detach()

            if n_out:
                current_pbar.set_description_str("Generating samples")
                current_pbar.reset(md_steps)
                current_pbar.refresh()

            x_samples[:pool_filled] = propagation_function(x_samples[:pool_filled], md_steps, n_out, current_pbar, total_pbar, x_sample_beta)

            if n_out_samples:
                for i, out_samples_bias_center in enumerate(out_samples_bias_centers):
                    with torch.inference_mode():
                        out_z_samples = torch.hstack([self.generator.generate_z_samples(n_out_samples, betas=x_sample_beta), torch.ones(n_out_samples, device=self.generator.device)[:, None]*out_samples_bias_center])
                        out_sample, out_log_R_zx = self.generator.F_zx(out_z_samples)
                        out_weight = self.generator.get_weights(out_sample, out_log_R_zx, out_z_samples, normalize=True, beta=x_sample_beta).detach().cpu().numpy()

                        out_sample = out_sample.detach().cpu()
                        out_samples[i, cycle] = out_sample
                        out_weights[i, cycle] = out_weight

                    del out_sample, out_z_samples, out_log_R_zx, out_weight        


        self._kill_processes()
        if n_out_samples:
            return out_samples, out_weights, out_samples_bias_centers

    def save(self, filename, on_exist="ask", ignore_deprecation=False):
        """Saves the AIMMD model and Boltzmann Generator weights and biases along with relevant metadata to a zip file

        Parameters
        ----------
        filename : str
            Name of the file to save the models to
        on_exist : str, optional
            Determines what to do if the file exists (one of "ask", "overwrite", "error"), by default "ask"
        """
        if not ignore_deprecation:
            raise DeprecationWarning("This save function is deprecated. Use torch.save instead. To ignore this message, pass ignore_deprecation=True.")

        if filename.endswith(".zip"):
            filename = filename[:-4]

        on_exist = on_exist.lower()

        filename_aimmd = filename + ".aimmd.pt"
        filename_generator = filename + ".generator.pt"
        filename_metadata = filename + ".metadata.json"

        filename_zipfile = filename + ".zip"

        for f in [filename_aimmd, filename_generator, filename_metadata, filename_zipfile]:
            if os.path.exists(f):
                if on_exist == "ask":
                    if input(f"File {f} exists. Replace? [y/N]: ").lower() != "y":
                        print("Aborting.")
                        return
                elif on_exist == "error":
                    raise FileExistsError(f)
                elif on_exist == "overwrite":
                    pass
                else:
                    raise ValueError(f"Invalid option encountered for on_exist: {on_exist}")
            
            Path(f).parent.absolute().mkdir(parents=True, exist_ok=True)

        metadata_dict = {
            "device": self.generator.device,
            "aimmd": {
                "n_input": self.aimmd_network.n_input,
                "n_hidden": self.aimmd_network.n_hidden,
                "n_nodes": self.aimmd_network.n_nodes,
                "sigmoid": self.aimmd_network.sigmoid
            },
            "generator": {
                "n_blocks": self.generator.masks.shape[0]//2,
                "n_hidden_layers": next(self.generator.s_networks.parameters()).shape[1],
                "n_nodes": next(self.generator.s_networks.parameters()).shape[0],
                "dimensions": self.generator.dimensions,
                "dimensions_conditions": self.generator.masks.shape[1] - self.generator.dimensions,
                "normal_std": self.generator.normal_std
            },
            "trainer": {
                "committor_q_A": self.committor_q_A,
                "committor_q_B": self.committor_q_B,
                "committor_target": self.committor_target,
                "md_timestep": self.md_timestep,
                "md_diffusion_coeff": self.md_diffusion_coeff,
                "committor_timestep": self.committor_timestep,
                "committor_diffusion_coeff": self.committor_diffusion_coeff
            }
        }
        
        torch.save(self.aimmd_network.state_dict(), filename_aimmd)
        torch.save(self.generator.state_dict(), filename_generator)

        with open(filename_metadata, "w") as file:
            json.dump(metadata_dict, file, indent=2)

        with ZipFile(filename_zipfile, "w") as zip_object:
            for f in [filename_aimmd, filename_generator, filename_metadata]:
                zip_object.write(f, Path(f).name)

        if os.path.exists(filename_zipfile):
            for f in [filename_aimmd, filename_generator, filename_metadata]:
                os.remove(f)
        else:
            raise EOFError("ZIP file creation failed")       

    @classmethod
    def from_file(clx, filename, potential, committor_potential, md_potential, order_param, order_param_kwargs={}, ignore_deprecation=False):
        """Loads a trainer object from a zip file

        Parameters
        ----------
        filename : str
            Name of the file to load
        potential : potential.Potential
            Potential to pass to the Boltzmann Generator init function
        committor_potential: potential.Potential
            Potential to use for running the fleeting trajectories in committor estimation (should not be biased)
        md_potential : potential.Potential
            Potential to use for propagating generated samples using short MD simulations (should be biased)
        order_param : callable
            Function that calculates the order paramater for given configuration(s)
        order_param_kwargs : dict, optional
            Keyword arguments to pass to the order parameter function if needed, by default {}

        Returns
        -------
        AIMMDCommittorTrainer
            The loaded trainer object
        """
        if not ignore_deprecation:
            raise DeprecationWarning("This load function is deprecated. Use torch.load instead. To ignore this message, pass ignore_deprecation=True.")
        if filename.endswith(".zip"):
            filename = filename[:-4]

        filename_aimmd = Path(filename + ".aimmd.pt").name
        filename_generator = Path(filename + ".generator.pt").name
        filename_metadata = Path(filename + ".metadata.json").name

        filename_zipfile = filename + ".zip"

        if not os.path.exists(filename_zipfile):
            raise FileNotFoundError(f"File {filename_zipfile} does not exist")
        
        with ZipFile(filename_zipfile, "r") as zip_object:
            metadata_dict = json.loads(zip_object.read(filename_metadata).decode("utf-8"))

            new_aimmd = AIMMDNetwork(*metadata_dict["aimmd"].values())
            new_aimmd.load_state_dict(torch.load(BytesIO(zip_object.read(filename_aimmd)), weights_only=True))
            new_aimmd = new_aimmd.to(metadata_dict["device"])

            potential.cv = new_aimmd.cv_function
            potential.cv_gradient = new_aimmd.cv_gradient_function

            new_generator = GeneratorBuilder(potential, metadata_dict["generator"]["n_blocks"], metadata_dict["generator"]["n_hidden_layers"], 
                                            metadata_dict["generator"]["n_nodes"], metadata_dict["generator"]["dimensions"], metadata_dict["generator"]["dimensions_conditions"], 
                                            metadata_dict["generator"]["normal_std"]).build_generator(cuda = metadata_dict["device"] == "cuda")
            new_generator.load_state_dict(torch.load(BytesIO(zip_object.read(filename_generator)), weights_only=True))
            new_generator = new_generator.to(metadata_dict["device"])

            return clx(new_aimmd, new_generator, committor_potential=committor_potential, md_potential=md_potential, order_param=order_param,
                       order_param_kwargs=order_param_kwargs, **metadata_dict["trainer"])