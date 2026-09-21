import numpy as np
from tqdm.auto import tqdm

from potential import Polymer

def mcmc_propagate(coords, n_timesteps, max_displacement, potential, initial_energy = None, n_equil=0, n_out=0, beta=1.0, progress=False):
    """Propagate a configuration using Markov Chain Monte Carlo and Metropolis rule

    Parameters
    ----------
    coords : np.ndarray
        A pair of x_1 and x_2 coordinates to propagate
    n_timesteps : int
        Number of timesteps to propagate for
    max_displacement : float
        Maximum displacement to generate. Particles are displaced within a square of side length
        max_displacement around current position with uniform probability.
    potential : object
        The potential for which to calculate energies. Must implement function evaluate(coords)
    initial_energy : float
        Energy of the initial configuration. If None, initial energy will be computed by evaluating potential
    n_equil : int, optional
        Number of steps to discard as equilibration steps, by default 0
    n_out : int, optional
        Record positions at every n_out step. If 0, only the final configuration will be returned, by default 0
    beta : float, optional
        Inverse simulation temperature, by default 1
    progress : bool, optional
        Whether to print a progress bar to console
    
    Returns
    -------
    Union[np.ndarray, tuple[np.ndarray, ndarray]]
        The final configuration and the trajectory if n_out > 0
    """

    trajectory = None

    if n_out > 0:
        trajectory = np.zeros(((n_timesteps - n_equil)//n_out, 2), dtype=float)

    x = coords.copy()
    energy = potential.evaluate(coords) if initial_energy == None else initial_energy

    for timestep in range(n_timesteps) if not progress else tqdm(range(n_timesteps)):
        proposed_x = x + np.random.uniform(-max_displacement/2, max_displacement/2, size=2)
        new_energy = potential.evaluate(proposed_x)

        if new_energy < energy or np.random.random() < np.exp(-beta*(new_energy - energy)):
            energy = new_energy
            x = proposed_x
        
        if n_out > 0 and (timestep - n_equil) % n_out == 0 and timestep >= n_equil:
            trajectory[(timestep - n_equil)//n_out] = x

    if n_out > 0:
        return x, trajectory
    return x

def polymer_biased_mcmc_2D(config, potential, n_samples, bias_center, enhanced_sampling=True, max_displacement=0.4, beta=1.0, n_out=1, n_equil=0, 
                           exchange_rate=0, progress=False):
    """Performes an MCMC simulation in the biased 2D polymer potential for M polymers with N monomers each. Optinally, enhanced sampling moves such
    as bond rerolling, mirroring and reversing are performed, as well as hamiltonian replica exchange when sampling in multiple windows.

    Parameters
    ----------
    config : np.ndarray
        The configuration(s) in flattened cartesian coordinates, i.e., with shape (M, N) or (N, ) for one polymer
    potential : potential.Polymer
        The polymer potential object
    n_samples : int
        Number of samples to produce. This is the effective sample number that will be returned in the end, so if n_out is larger than 1 and/or
        n_equil is larger than 0, more MC steps will be performed in order to achieve this sample number
    bias_center : float | np.ndarray
        Bias center(s) in which to simulate. If this is a float, the same bias cener will be used for all configurations
    enhanced_sampling : bool, optional
        If this is True, bond rerolling, mirroring and reversal moves will be performed in addition to particle displacement moves, by default True
    max_displacement : float, optional
        Maximum displacement value for particle displacement moves (half of the side length of the square in which particles are displaced), by default 0.4
    beta : float, optional
        Beta value to use in simulation/computation of energies, by default 1.0
    n_out : int, optional
        Output frequency. If this is 0, only final configuration(s) will be returned, by default 1
    n_equil : int, optional
        Number of MC steps to perform as equilibration before samples are being collected, by default 0
    exchange_rate : int, optional
        Rate at which to attempt hamiltonian replica exchange moves. This is in units of n_out, so a value of, e.g., 2 means that the exchange is 
        attempted after every second output of samples. If this is zero, no exchange is attempted, by default 0
    progress : bool | tqdm.tqdm | List[tqdm], optional
        If this is True, a progress bar will be shown during sampling
        If this is a tqdm progress bar object or a list of such, the given progress bars will be updated, by default False

    Returns
    -------
    np.ndarray
        The final configuration(s) after the performed MC steps
    list[np.ndarray(shape=(M, )), np.ndarray(shape=(M, )), np.ndarray(shape=(M, ))]
        Statistics about the sampling

        First element is the acceptance rate for each polymer
        Second element is the rejection rate for each polymer
        Third element is the rate of accepted exchanges for each polymer
    np.ndarray, optional
        The trajectory of the simulation (the generated samples) if n_out is larger than 0
    """
    
    assert not ((exchange_rate > 0) and (config.ndim == 1 or (config.ndim == 2 and len(config) == 1))), "Need at least two replicas in order to perform Hamiltonian Replica Exchange"
    config = config.copy()

    flatten = config.ndim == 1
    if flatten:
        config = config[None, :]

    if isinstance(bias_center, (int, float)):
        bias_center = np.array(config.shape[0]*[bias_center], dtype=float)

    n_timesteps = int(n_samples*n_out + n_equil)

    trajectory = np.zeros((n_samples, *config.shape)) if n_out else None
    acc = np.zeros(config.shape[0])
    rej = np.zeros_like(acc)
    exchanged = np.zeros_like(acc)
    attempted_exchanges = np.zeros_like(acc)

    current_bias_U = potential.evaluate_bias_potential(config, bias_center)
    current_U = potential.evaluate_unbiased(config) + current_bias_U

    if isinstance(progress, tqdm):
        pbars = [progress]
    elif isinstance(progress, list):
        pbars = progress
    else:
        pbars = None

    for step in range(n_timesteps) if ((isinstance(progress, bool) and progress == False) or isinstance(progress, list)) else tqdm(range(n_timesteps), smoothing=0):
        new_config = config.copy()

        if enhanced_sampling:
            configs_to_reroll = np.random.random(config.shape[0]) < 1/200

            if np.sum(configs_to_reroll) > 0:
                new_config[configs_to_reroll] = Polymer.reroll_bonds(new_config[configs_to_reroll], in_place=False)

            n_configs_not_rerolled = int(np.sum(~configs_to_reroll))

            if n_configs_not_rerolled > 0:
                particles_chosen = np.random.randint(0, config.shape[-1]//2, n_configs_not_rerolled)
                particles_chosen = np.stack((particles_chosen*2, particles_chosen*2 + 1), axis=1).flatten()
                dr = np.random.random(n_configs_not_rerolled*potential.dimensions) * max_displacement
                dr *= np.random.choice([-1, 1], dr.shape)

                sub_configs = new_config[~configs_to_reroll]
                sub_configs[np.repeat(np.arange(n_configs_not_rerolled, dtype=int), potential.dimensions), particles_chosen] += dr
                new_config[~configs_to_reroll] = sub_configs

                new_config[~configs_to_reroll] = Polymer.reverse(new_config[~configs_to_reroll], probability=0.1, in_place=False)
                new_config[~configs_to_reroll] = Polymer.mirror(new_config[~configs_to_reroll], probability=0.1, in_place=False)

        else:
            particles_chosen = np.random.randint(0, config.shape[-1]//2, config.shape[0])
            particles_chosen = np.stack((particles_chosen*2, particles_chosen*2 + 1), axis=1).flatten()
            dr = np.random.random(config.shape[0]*potential.dimensions) * max_displacement
            dr *= np.random.choice([-1, 1], dr.shape)

            new_config[np.repeat(np.arange(config.shape[0], dtype=int), potential.dimensions), particles_chosen] += dr

        x_mean = np.mean(new_config[:, ::2], axis=-1)[:, None]
        y_mean = np.mean(new_config[:, 1::2], axis=-1)[:, None]

        new_config[:, ::2] -= x_mean
        new_config[:, 1::2] -= y_mean

        new_bias_U = potential.evaluate_bias_potential(new_config, bias_center)
        new_U = potential.evaluate_unbiased(new_config) + new_bias_U

        configs_to_accept = np.random.random(len(new_U)) < np.exp(-beta*(new_U - current_U))

        config[configs_to_accept] = new_config[configs_to_accept].copy()
        current_bias_U[configs_to_accept] = new_bias_U[configs_to_accept].copy()
        current_U[configs_to_accept] = new_U[configs_to_accept].copy()

        acc += configs_to_accept
        rej += ~configs_to_accept

        if n_out and step >= n_equil and (step - n_equil)%n_out == 0:
            trajectory[(step - n_equil)//n_out] = config.copy()
            
        if n_out and exchange_rate and step >= n_equil and (step - n_equil + 1)%(exchange_rate*n_out) == 0:
            n_partners = config.shape[0] if config.shape[0]%2 == 0 else config.shape[0] - 1
            exchange_partners = np.random.choice(np.arange(config.shape[0]), replace=False, size=n_partners).reshape((-1, 2))

            i = exchange_partners[:, 0]
            j = exchange_partners[:, 1]

            exchange_energies = beta*(current_bias_U[j] + current_bias_U[i] - potential.evaluate_bias_potential(config[j], bias_center[i]) - potential.evaluate_bias_potential(config[i], bias_center[j]))
            accepted_exchanges = exchange_partners[np.random.random() < np.exp(exchange_energies)]

            if len(accepted_exchanges) > 0:
                inv_accepted_exchanges = accepted_exchanges[:, ::-1]
                config[accepted_exchanges] = config[inv_accepted_exchanges]

                accepted_exchanges = accepted_exchanges.flatten()

                current_bias_U[accepted_exchanges] = potential.evaluate_bias_potential(config[accepted_exchanges], bias_center[accepted_exchanges])
                current_U[accepted_exchanges] = potential.evaluate_unbiased(config[accepted_exchanges]) + current_bias_U[accepted_exchanges]

            exchanged[accepted_exchanges] += 1
            attempted_exchanges[exchange_partners.flatten()] += 1

        if pbars is not None:
            for pbar in pbars:
                pbar.update()
                pbar.refresh()


    if flatten:
        config = config.flatten()
        acc = acc[0]
        rej = rej[0]
        if n_out:
            trajectory = trajectory.reshape((trajectory.shape[0], trajectory.shape[2]))

    if n_out:
        return config, [acc/n_timesteps, rej/n_timesteps, exchanged/(attempted_exchanges if exchange_rate else 1)], trajectory
    return config, [acc/n_timesteps, rej/n_timesteps, exchanged/(attempted_exchanges if exchange_rate else 1)]

    

    