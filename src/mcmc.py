import numpy as np
from tqdm import tqdm

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