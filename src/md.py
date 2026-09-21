import numpy as np
from numba import jit

@jit(nopython=True)
def brownian_propagate(positions : np.ndarray, forces : np.ndarray, timestep : float, diffusion_coeff : float, beta : float = 1.0) -> np.ndarray:
    """Propagates particle position(s) according to Brownian (overdamped Langevin) dynamics for a single timestep.

    Parameters
    ----------
    positions : np.ndarray
        The position(s) to be propagated. Can be a 1D or 2D numpy array
    forces : np.ndarray
        The current forces acting on the particles at their current positions. Must be the same shape as positions.
    timestep : float
        Timestep to use in propagation
    diffusion_coeff : float
        Diffusion coefficient
    beta : float, optional
        Simulation temperature, by default 1.0

    Returns
    -------
    np.ndarray
        The updated positions
    """
    dr = np.random.randn(*positions.shape).astype(np.float32)

    return positions + diffusion_coeff * timestep * beta * forces + np.sqrt(2 * diffusion_coeff * timestep) * dr