import functools

import numpy as np
import torch

def correlate(A, B, normalize=True):
    """Get correlation function for input vectors A and B

    Parameters
    ----------
    A : np.ndarray
        Input vector A
    B : np.ndarray
        Input vector B (same length as A)
    normalize : bool, optional
        Whether to normalize the correlation function using sample variance, by default True

    Returns
    -------
    np.ndarray
        Correlation function of A and B
    """
    assert len(A) == len(B), f"Inputs have different lengths ({len(A)} != {len(B)})"
    M = len(A)

    # Remove center of mass motion
    com_A = A - np.mean(A)
    com_B = B - np.mean(B)

    C_AB = np.correlate(com_A, com_B, mode="full")[-M:] / np.arange(M, 0, -1)

    if normalize:
        C_AB /= np.std(com_A)*np.std(com_B)

    return C_AB


def effective_sample_size(sample, max_lag, tolerance = 0.01, relative=True):
    """Computes (relative) effective sample size of a given sample

    Parameters
    ----------
    sample : np.ndarray
        The input sample for which to calculate relative effective sample size
    max_lag : float
        Maximum lag to use for autocorrelation time
    tolerance : float
        Minimum value to be considered unequal to zero, by default 0.01
    relative : bool, optional
        Whether to generate relative or absolute effective sample size, by default True

    Returns
    -------
    float
        (Relative) effective sample size of sample
    """
    autocorrelation = correlate(sample, sample)[1:max_lag+1] # Exclude 0 lag
    relative_effective_sample_size = 1/(1 + 2*np.sum(np.where(autocorrelation > tolerance, autocorrelation, 0)))
    return relative_effective_sample_size if relative else len(sample) * relative_effective_sample_size



def wham_position_bias(samples_cv, bias_centers, bias_potential_function, bins, max_steps=100000, max_eps=1e-5, mute_warn=False):
    """Combines multiple biased histograms into unbiased single histogram by solving WHAM equations. The bias must be of
    the form of a bias potential that depends on a scalar collective variable and the data must also be that collective
    variable.

    Parameters
    ----------
    samples_cv : np.ndarray | list
        A list containing NumPy arrays or a 2D NumPy array where each row is the data from a simulation belonging to the
        same bias center
    bias_centers : np.ndarray
        The bias centers at which the simulations were performed, in the same order as the simulation data in samples_cv
    bias_potential_function : callable
        A function that computes the bias potential given an array of collective variable values and a bias center at which
        to evaluate the potential. The first positional argument must accept an array of collective variable values and the
        second must accept a single bias center common to all the collective variable values.
    bins : int | float | np.ndarray | list
        Information about the bins of the resulting single histogram. If an int is passed, it is interpreted as the bin count
        and automatic bin spacing will be applied. If a float is passed, it will be interpreted as the bin width and the
        appropriate number of bins will be selected. In both of these cases, the first bin starts at the smallest value in
        samples_cv and the last bin ends at the highest value in samples_cv. If a NumPy array or a list is passed, it will
        be used for the bin edges, so if N bins are desired, the array/list must have a length of N + 1. Also, bin widths
        must be constant.
    max_steps : int, optional
        Max steps after which to stop self-consistent solving of WHAM equations, by default 100000
    max_eps : float, optional
        The maximum relative difference between subsequent iterations of the self-consistent solving scheme used to solve the
        WHAM equations, by default 1e-5
    mute_warn : bool, optional
        If this is False, a warning will be printed if the maximum number of steps is exceeded while the required maximum relative
        difference is not yet reached, by default False

    Returns
    -------
    np.ndarray
        The optimal unbiased histogram created from the individual histograms, normalized to a probability density. Here, optimal
        means that the variance of the best estimator for the probability density in each bin is minimized.
    np.ndarray
        The bin centers of the histogram
    """

    min_cv, max_cv = np.min(samples_cv), np.max(samples_cv)

    if isinstance(bins, float):
        delta_cv = bins
        n_bins = int(np.ceil((max_cv - min_cv)/delta_cv))
        bin_edges = np.arange(min_cv, min_cv + (n_bins + 1)*delta_cv, delta_cv)
    elif isinstance(bins, int):
        n_bins = bins
        delta_cv = (max_cv - min_cv)/n_bins
        bin_edges = np.linspace(min_cv, max_cv, n_bins + 1)
    elif isinstance(bins, list):
        bin_edges = np.ndarray(bins)
        delta_cv = bin_edges[1] - bin_edges[0]
        n_bins = len(bin_edges) - 1
    elif isinstance(bins, np.ndarray):
        bin_edges = bins
        delta_cv = bin_edges[1] - bin_edges[0]
        n_bins = len(bin_edges) - 1

    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
    n_simulations = len(bias_centers)

    h = np.array([np.histogram(samples_cv[i], bin_edges)[0] for i in range(n_simulations)])
    N = np.array([len(samples_cv[i]) for i in range(n_simulations)]) # Number of samples in each simulation

    c = np.array([np.exp(-bias_potential_function(bin_centers, bias_centers[i])) for i in range(n_simulations)])

    f = np.ones(n_simulations)
    p = np.ones(n_bins)

    h_T_sum = np.sum(h.T, axis=1) # Prevent repeated calculation of this constant

    step = 0

    while step < max_steps:
        new_f = 1/np.dot(c, p)
        eps = np.max(np.abs((new_f - f)/f))
        f = new_f
        
        p = h_T_sum / np.dot(c.T, N*f)

        if eps < max_eps:
            break
        step += 1
    
    if step == max_steps and not mute_warn:
        print(f"WARNING: Max steps ({max_steps}) exceeded but eps = {eps} > max_eps ({max_eps})")

    return p / (np.sum(p) * delta_cv), bin_centers


def graceful_keyboard_interrupt(callback=None, message=None):
    """
    A decorator to handle `KeyboardInterrupt` exceptions gracefully during the execution of a function.

    Parameters
    ----------
    callback : callable, optional
        A function to be executed when a `KeyboardInterrupt` is caught.
        If `callback_needs_self` is True, the `self` instance will be passed as an argument to the callback,
        by default None

    message : str, optional
        A message to be printed when a `KeyboardInterrupt` is caught, by default None

    Returns
    -------
    callable
        A decorator that wraps the target function with the `KeyboardInterrupt` handling logic.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                if callback != None:
                    callback()
                if message != None:
                    print(message)

        return wrapper
    return decorator


def graceful_keyboard_interrupt_class(callback=None, callback_needs_self=False, message=None):
    """
    A decorator to handle `KeyboardInterrupt` exceptions gracefully during the execution of a method inside of a class.

    Parameters
    ----------
    callback : callable, optional
        A function to be executed when a `KeyboardInterrupt` is caught.
        If `callback_needs_self` is True, the `self` instance will be passed as an argument to the callback,
        by default None

    callback_needs_self : bool, optional
        Indicates whether the `self` instance should be passed to the callback,by default False

    message : str, optional
        A message to be printed when a `KeyboardInterrupt` is caught, by default None

    Returns
    -------
    callable
        A decorator that wraps the target method with the `KeyboardInterrupt` handling logic.
    """
    def decorator(func):
        @functools.wraps(func) # Keeps the name intact, so that decorated methods stay picklable for multiprocessing
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except KeyboardInterrupt:
                if callback != None:
                    callback() if not callback_needs_self else callback(self)
                if message != None:
                    print(message)

        return wrapper
    return decorator

def ress(weights):
    """Computes the relative effective sample size given sample weights

    Parameters
    ----------
    weights : torch.Tensor | np.ndarray
        The normalized weights of the samples

    Returns
    -------
    float
        The relative effective sample size
    """
    if isinstance(weights, torch.Tensor):
        return 1/(torch.sum(weights**2) * weights.shape[0])
    elif isinstance(weights, np.ndarray):
        return 1/(np.sum(weights**2) * weights.shape[0])