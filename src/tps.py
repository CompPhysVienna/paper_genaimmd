import numpy as np
from tqdm.auto import tqdm
from md import brownian_propagate


def h_A_B(A_B, coords, order_parameter, q_A, q_B, order_parameter_kwargs = {}):
    """Characteristic function for the order parameter. Returns True if the coordinates are in state A_B and
    False otherwise.

    Parameters
    ----------
    A_B : str
        Either "A" for state A or "B" for state B
    coords : np.ndarray
        The coordinates for which to evaluate the order parameter
    order_parameter : callable
        A function that computes the order parameter given some coordinates
    q_A : float
        Threshold below which order parameter values are considered to belong to state A
    q_B : _type_
        Threshold above which order parameter values are considered to belong to state B
    order_parameter_kwargs : dict, optional
        Keyword arguments to pass to the order parameter, {} by default

    Returns
    -------
    Union[bool, np.ndarray]
        The result of the characteristic function. If coords is 1D, a boolean is returned,
        otherwise a 1D numpy array is returned.
    """
    A_B = A_B.upper()
    assert A_B in ["A", "B"], "A_B must be either A or B"
    q = order_parameter(coords, **order_parameter_kwargs)

    return q < q_A if A_B == "A" else q > q_B


def generate_path_to_state_brownian(shooting_point, order_parameter, q_A, q_B, potential, timestep, diffusion_coeff, target=None, order_parameter_kwargs = {}, 
                                    beta=1.0, max_steps=10000, mute_warn=False, trajectory_out_frequency=0):
    """Propagates a trajectory from a given point using Brownian dynamics and observes what state is reached by the trajectory

    Parameters
    ----------
    coords : np.ndarray
        Coordinates of the initial point
    order_parameter : callable
        A function that computes the order parameter given some coordinates
    q_A : float
        Threshold below which order parameter values are considered to belong to state A
    q_B : _type_
        Threshold above which order parameter values are considered to belong to state B
    potential : Potential
        An instance of a Potential class that implements the evaluate_force function
    timestep : float
        Timestep to use for propagation
    diffusion_coeff : float
        Diffusion coefficient to use for propagation
    target : str, optional
        Target state, either "A" for state A or "B" for state B. If None, only trajectory will be returned, by default None
    order_parameter_kwargs : dict, optional
        Keyword arguments to pass to the order parameter, {} by default
    beta : float, optional
        Temperature to use for propagation, by default 1.0
    max_steps : int, optional
        Number of timesteps to propagate for at maximum, by default 10000
    mute_warn : bool, optional
        If True, the warning message that is printed to console if max_steps is reached is muted, by default False
    trajectory_out_frequency : int, optional
        Determines output frequency of points along trajectory. If 0, no trajectory is recorded and returned. Always includes
        end of simulation, by default 0

    Returns
    -------
    bool
        True if target state was reached, False if other state was reached or max_steps was reached.
        This is only returned if target is not None

    np.ndarray
        The trajectory with output frequency trajectory_out_frequency. This is only returned if trajectory_out_frequency > 0.
    """
    if target != None:
        target = target.upper()
        assert target in ["A", "B"], "target must be either A or B"
        other = "A" if target == "B" else "B"

    if trajectory_out_frequency:
        assert trajectory_out_frequency > 0, "trajectory_out_frequency must be >= 0"
        trajectory = []

    position = shooting_point
    for step in range(max_steps):
        force = potential.evaluate_force(position)
        position = brownian_propagate(position, force, timestep, diffusion_coeff, beta)

        if target != None:
            h_target = h_A_B(target, position, order_parameter, q_A, q_B, order_parameter_kwargs)
            h_other = h_A_B(other, position, order_parameter, q_A, q_B, order_parameter_kwargs)
        else:
            h_target = h_A_B("A", position, order_parameter, q_A, q_B, order_parameter_kwargs)
            h_other = h_A_B("B", position, order_parameter, q_A, q_B, order_parameter_kwargs)

        if h_target or h_other:
            if trajectory_out_frequency and target != None:
                trajectory.append(position)
                return (True, np.array(trajectory)) if h_target else (False, np.array(trajectory))
            elif trajectory_out_frequency:
                trajectory.append(position)
                return np.array(trajectory)
            elif target != None:
                return True if h_target else False
            else:
                return None

        if trajectory_out_frequency and step % trajectory_out_frequency == 0:
            trajectory.append(position)
        
    if not mute_warn:
        print(f"WARNING: Max iterations reached at point {position}")

    return False


def generate_reactive_paths_brownian(shooting_points, potential, order_parameter, q_A, q_B, timestep, diffusion_coeff, trajectory_out_frequency, 
                                     order_parameter_kwargs = {}, progress=False, max_attempts=1, max_steps=10000, mute_warn=False, return_sp_indices=False):
    """Given a set of shooting points, tries to generate single reactive path from each shooting point for a given maximum number of attempts using Brownian dynamics

    Parameters
    ----------
    shooting_points : np.ndarray
        The shooting points from which to start generating paths
    potential : potential.Potential
        The potential to use for propagation
    order_parameter : callable
        The order parameter to use for evaluating whether a stable state was reached
    q_A : float
        Maximum value of order parameter below which a configuration is considered to be in state A
    q_B : float
        Minimum value of order parameter above which a configuration is considered to be in state B
    timestep : float
        Timestep to use for propagation
    diffusion_coeff : float
        Diffusion coefficient to use for propagation
    trajectory_out_frequency : int
        Frequency with which to output steps to trajectory
    order_parameter_kwargs : dict, optional
        Keyword arguments to pass to order parameter function, by default {}
    progress : bool, optional
        If True, progress bar will be shown, by default False
    max_attempts : int, optional
        Maximum number of times to try to find a reactive path per shooting point, by default 1
    max_steps : int, optional
        Maximum number of steps to do before reaching a stable state, by default 10000
    mute_warn : bool, optional
        If True, no warning will be printed if max_attempts is exceeded. If max_attempts = 1, this is automatically True, by default False
    return_sp_indices : bool, optional
        If this is True, indices of the input shooting points instead of actual shooting points of reactive paths are returned as the second
        return value, by default False

    Returns
    -------
    list
        Generated trajectories, possibly smaller than the shooting points given, since not all shooting points must result in reactive paths
    list
        Shooting points corresponding to the generated trajectories if return_sp_indices is False, otherwise indices of shooting points corresponding
        to the generated trajectories
    """

    if shooting_points.ndim == 1: # Only a single shooting point is given, reshape into 2D array of single shooting point
        shooting_points = shooting_points.reshape(1, -1)

    trajectory_ensemble = []
    reactive_shooting_points = []

    for shooting_point_index, shooting_point in enumerate(tqdm(shooting_points, smoothing=0)) if progress else enumerate(shooting_points):
        for _ in range(max_attempts):
            forward_traj = generate_path_to_state_brownian(shooting_point, order_parameter, q_A, q_B, potential, timestep, diffusion_coeff, trajectory_out_frequency=trajectory_out_frequency, order_parameter_kwargs=order_parameter_kwargs, max_steps=max_steps, mute_warn=mute_warn)
            backward_traj = generate_path_to_state_brownian(shooting_point, order_parameter, q_A, q_B, potential, timestep, diffusion_coeff, trajectory_out_frequency=trajectory_out_frequency, order_parameter_kwargs=order_parameter_kwargs, max_steps=max_steps, mute_warn=mute_warn)

            fw_in_A = h_A_B("A", forward_traj[-1], order_parameter, q_A, q_B, order_parameter_kwargs)
            fw_in_B = h_A_B("B", forward_traj[-1], order_parameter, q_A, q_B, order_parameter_kwargs)

            bw_in_A = h_A_B("A", backward_traj[-1], order_parameter, q_A, q_B, order_parameter_kwargs)
            bw_in_B = h_A_B("B", backward_traj[-1], order_parameter, q_A, q_B, order_parameter_kwargs)

            if (fw_in_A and bw_in_B) or (fw_in_B and bw_in_A):
                trajectory_ensemble.append(np.vstack([backward_traj[::-1], forward_traj[1:]]))
                reactive_shooting_points.append(shooting_point if not return_sp_indices else shooting_point_index)
                break # Found reactive path, go to next shooting point
        if max_attempts > 1 and not mute_warn:
            print(f"WARNING: Failed to find reactive path after {max_attempts} attempts for shooting point {shooting_point}")
        
    return trajectory_ensemble, reactive_shooting_points

def load_umbrella_sampling_samples(filename, sort=True):
    """Loads samples from a file that where created using Umbrella Sampling. File must be a csv file
    with N columns, the first (N-1) being cartesian coordinates of the samples and the last one being
    the bias center they were generated at.

    Parameters
    ----------
    filename : str
        The path to the file to load
    sort : bool, optional
        Whether to sort the data by bias center, by default True

    Returns
    -------
    np.ndarray
        An array of shape (N_bias_centers, N_samples_per_bias_center, N-1) containing the simulation data
        grouped by bias center, where the bias center is in the first axis and the simulation data is in the
        second and third axis.
    np.ndarray
        The (sorted) bias centers
    """

    loaded_samples = np.loadtxt(filename, delimiter=",")
    bias_centers = list(set(center.item() for center in loaded_samples[:, -1]))
    if sort:
        bias_centers.sort()
    bias_centers = np.array(bias_centers)

    samples = np.empty((len(bias_centers), len(loaded_samples)//len(bias_centers), loaded_samples.shape[1] - 1))

    for i in range(len(bias_centers)):
        samples[i] = loaded_samples[loaded_samples[:, 2] == bias_centers[i]][:, :-1]

    return samples, bias_centers