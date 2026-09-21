import matplotlib.animation
import torch
from typing import Union
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from numba import jit

from abc import ABC, abstractmethod
from util import TO_CM, vector2D_angle, vector2D_rotate, vector2D_angle_npy

class Potential(ABC):

    @abstractmethod
    def evaluate(self, coords : Union[torch.tensor, np.ndarray]) -> Union[Union[torch.tensor, np.ndarray], float]:
        """Evaluates the potential at given coordinates

        Parameters
        ----------
        coords : Union[torch.tensor, np.ndarray]
            Either a 1D or 2D array of coordinates at which to evaluate the potential

        Returns
        -------
        Union[Union[torch.tensor, np.ndarray], float]
            The potential value(s) that were computed
        """
        pass


class Polymer(Potential):
    """Simple Polymer model, describes interaction between N monomers in a chain polymer through a combination of
    three pairwise additive potentials:

    * Lennard-Jones (LJ) potential U_LJ(r_ij) = 4 eps * ((sigma/r_ij)^12 - (sigma/r_ij)^6)
    * Harmonic bond potential U_bond(r_ij) = k_bond/2 * (r_ij - d_bond)
    * Angular potential U_angle(theta_ijl) = k_angle/2 * (1 - cos(theta_ijl - theta_angle))
    """

    def __init__(self, n_monomers, dimensions, cv_function=None, cv_gradient_function=None, k_bias=100.0, eps_LJ=1.0, sigma_LJ=1.0, d_bond=np.power(2, 1/6), 
                 k_bond=5.0, theta_angle=np.pi, k_angle=1.4):
        """Init function for the Polymer model

        Parameters
        ----------
        n_monomers : int
            Number of monomers that make up the polymer
        dimensions : int
            Number of dimensions
        cv_function : callable, optional
            The cv function used to evaluate the bias potential, by default None
        cv_gradient_function : callable, optional
            The cv gradient function used to evaluate the bias force, by default None
        k_bias : float, optional
            Strength of the bias force
        eps_LJ : float, optional
            Value of parameter epsilon in LJ potential, by default 1.0
        sigma_LJ : float, optional
            Value of parameter sigma in LJ potential, by default 1.0
        d_bond : float, optional
            Target bond length for harmonic bond potential, by default np.power(2, 1/6)
        k_bond : float, optional
            Force constant for harmonic bond potential, by default 5.0
        theta_angle : float, optional
            Target angle for the angle potential, by default np.pi
        k_angle : float, optional
            Force constant for angle potential, by default 1.4
        """
        self.n_monomers = n_monomers
        self.dimensions = dimensions
        self.cv_function = cv_function
        self.cv_gradient_function = cv_gradient_function
        self.k_bias = k_bias
        self.eps_LJ = eps_LJ
        self.sigma_LJ = sigma_LJ
        self.d_bond = d_bond
        self.k_bond = k_bond
        self.theta_angle = theta_angle
        self.k_angle = k_angle

        self.param_array

    
    @property
    def param_array(self):
        """Used for backwards compatibility. Build the parameter array if the object does not have it yet.

        Returns
        -------
        np.ndarray
            The parameter array for this object, containing self.n_monomers, self.dimensions, self.k_bias, self.eps_LJ, self.sigma_LJ,
            self.d_bond, self.k_bond, self.theta_angle, self.k_angle
        """
        if not hasattr(self, "_param_array"):
            self._param_array = np.array([self.n_monomers, self.dimensions, self.k_bias, self.eps_LJ, self.sigma_LJ, self.d_bond, self.k_bond, self.theta_angle, self.k_angle])
        return self._param_array


    def evaluate_unbiased(self, coords):
        """Evaluates the unbiased Polymer potential

        Parameters
        ----------
        coords : np.ndarray | torch.Tensor
            The cartesian coordinates of the sample(s) for which to evaluate the unbiased potential. Coordinates belonging to the same polymer must be in a 1D array.

        Returns
        -------
        np.ndarray | torch.Tensor | float
            The unbiased potential energy/energies of the sample(s)
        """
        if isinstance(coords, np.ndarray):
            return self._evaluate_unbiased_npy(np.ascontiguousarray(coords), self.param_array)
        
        if coords.ndim == 1:
            coords = coords.reshape((self.n_monomers, self.dimensions))
            U_total = 0
        else:
            coords = coords.reshape((-1, self.n_monomers, self.dimensions))
            U_total = torch.zeros(coords.shape[0], dtype=coords.dtype, device=coords.device)
        
        for i in range(self.n_monomers):
            for j in range(i + 1, self.n_monomers):
                r = torch.norm(coords[..., i, :] - coords[..., j, :], dim=-1)
                
                term6 = (self.sigma_LJ/r)**6
                term12 = term6**2

                U_total += 4*self.eps_LJ*(term12 - term6)

        relative_vectors = coords[..., 1:, :] - coords[..., :-1, :]

        distances = torch.linalg.norm(relative_vectors, dim=-1)
        angles = vector2D_angle(relative_vectors[..., :-1, :], relative_vectors[..., 1:, :])
        
        U_total += self.k_bond/2 * torch.sum((distances - self.d_bond)**2, dim=-1)

        U_total += self.k_angle/2 * torch.sum(1 - torch.cos(angles - self.theta_angle - np.pi), dim=-1)

        return U_total

    @staticmethod
    @jit(nopython=True)
    def _evaluate_unbiased_npy(coords, param_array):
        """Pure NumPy version of the evaluate_unbiased function. Not to be called directly, only by the wrapper function.

        Parameters
        ----------
        coords : np.ndarray
            See docstring of evaluate_unbiased
        param_array : np.ndarray
            Polymer.param_array containing the model parameters

        Returns
        -------
        np.ndarray
            See docstring of evaluate_unbiased
        """
        n_monomers, dimensions = int(param_array[0]), int(param_array[1])
        eps_LJ, sigma_LJ, d_bond, k_bond, theta_angle, k_angle = param_array[3], param_array[4], param_array[5], param_array[6], param_array[7], param_array[8]

        flatten = coords.ndim == 1
        coords = coords.reshape((-1, n_monomers, dimensions))
        U_total = np.zeros(coords.shape[0])
        
        for i in range(n_monomers):
            for j in range(i + 1, n_monomers):
                r = np.sqrt(np.sum((coords[..., i, :] - coords[..., j, :])**2, axis=-1))

                term6 = (sigma_LJ/r)**6
                term12 = term6**2

                U_total += 4*eps_LJ*(term12 - term6)

        relative_vectors = coords[..., 1:, :] - coords[..., :-1, :]

        distances = np.sqrt(np.sum(relative_vectors**2, axis=-1))
        angles = vector2D_angle_npy(relative_vectors[..., :-1, :], relative_vectors[..., 1:, :])
        
        U_total += k_bond/2 * np.sum((distances - d_bond)**2, axis=-1)

        U_total += k_angle/2 * np.sum(1 - np.cos(angles - theta_angle - np.pi), axis=-1)

        return U_total if not flatten else U_total[0]


    def evaluate_force_unbiased(self, coords):
        """Evaluates the unbiased Polymer force

        Parameters
        ----------
        coords : np.ndarray | torch.Tensor
            The cartesian coordinates of the sample(s) for which to evaluate the unbiased force. Coordinates belonging to the same polymer must be in a 1D array.

        Returns
        -------
        np.ndarray | torch.Tensor
            The unbiased force(s) of the sample(s)
        """
        if isinstance(coords, np.ndarray):
            return self._evaluate_force_unbiased_npy(np.ascontiguousarray(coords), self.param_array)
        
        input_shape = coords.shape
        if coords.ndim == 1:
            coords = coords.reshape((self.n_monomers, self.dimensions))
            forces = torch.zeros((self.n_monomers, self.dimensions), dtype=coords.dtype, device=coords.device)
        else:
            coords = coords.reshape((-1, self.n_monomers, self.dimensions))
            forces = torch.zeros((coords.shape[0], self.n_monomers, self.dimensions), dtype=coords.dtype, device=coords.device)

        for i in range(self.n_monomers):
            for j in range(i + 1, self.n_monomers):
                relative_vectors = coords[..., i, :] - coords[..., j, :]

                r = torch.norm(relative_vectors, dim=-1)

                term6 = (self.sigma_LJ/r)**6
                term12 = term6**2

                if coords.ndim == 3:
                    term6 = term6[..., None]
                    term12 = term12[..., None]
                    r = r[..., None]

                force = 24*self.eps_LJ * (2*term12 - term6) * relative_vectors/r**2

                forces[..., i, :] += force
                forces[..., j, :] -= force

        relative_vectors = coords[..., 1:, :] - coords[..., :-1, :]

        distances = torch.linalg.norm(relative_vectors, dim=-1)
        angles = vector2D_angle(relative_vectors[..., :-1, :], relative_vectors[..., 1:, :])

        force = (self.k_bond*(distances - self.d_bond)/distances)[..., None] * relative_vectors
        
        for i in range(self.n_monomers - 1):
            forces[..., i, :] += force[..., i, :]
            forces[..., i+1, :] -= force[..., i, :]

        normal_relative_vectors = torch.flip(relative_vectors, (-1, ))
        normal_relative_vectors[..., -1] *= -1

        force = self.k_angle/2 * torch.sin(angles - self.theta_angle)

        for i in range(1, self.n_monomers - 1):
            f = force[..., i-1][:, None] if coords.ndim == 3 else force[..., i-1]
            d_1 = distances[..., i-1][..., None] if coords.ndim == 3 else distances[..., i-1]
            d_3 = distances[..., i][..., None] if coords.ndim == 3 else distances[..., i]

            force_1 = f/d_1**2*normal_relative_vectors[..., i-1, :]
            force_3 = f/d_3**2*normal_relative_vectors[..., i, :]
            force_2 = -force_1 - force_3

            forces[..., i-1, :] += force_1
            forces[..., i+1, :] += force_3
            forces[..., i, :] += force_2

        return forces.reshape(input_shape)

    @staticmethod
    @jit(nopython=True)
    def _evaluate_force_unbiased_npy(coords, param_array):
        """Pure NumPy version of the evaluate_force_unbiased function. Not to be called directly, only by the wrapper function.

        Parameters
        ----------
        coords : np.ndarray
            See docstring of evaluate_force_unbiased
        param_array : np.ndarray
            Polymer.param_array containing the model parameters

        Returns
        -------
        np.ndarray
            See docstring of evaluate_unbiased
        """
        n_monomers, dimensions = int(param_array[0]), int(param_array[1])
        eps_LJ, sigma_LJ, d_bond, k_bond, theta_angle, k_angle = param_array[3], param_array[4], param_array[5], param_array[6], param_array[7], param_array[8]

        input_shape = coords.shape

        coords = coords.reshape((-1, n_monomers, dimensions))
        forces = np.zeros((coords.shape[0], n_monomers, dimensions))

        for i in range(n_monomers):
            for j in range(i + 1, n_monomers):
                relative_vectors = coords[..., i, :] - coords[..., j, :]
                r = np.sqrt(np.sum(relative_vectors**2, axis=-1))

                term6 = (sigma_LJ/r)**6
                term12 = term6**2

                term6 = term6[..., None]
                term12 = term12[..., None]
                r = r[..., None]

                force = 24*eps_LJ * (2*term12 - term6) * relative_vectors/r**2

                forces[..., i, :] += force
                forces[..., j, :] -= force

        relative_vectors = coords[..., 1:, :] - coords[..., :-1, :]

        distances = np.sqrt(np.sum(relative_vectors**2, axis=-1))
        angles = vector2D_angle_npy(relative_vectors[..., :-1, :], relative_vectors[..., 1:, :])

        force = (k_bond*(distances - d_bond)/distances)[..., None] * relative_vectors
        
        for i in range(n_monomers - 1):
            forces[..., i, :] += force[..., i, :]
            forces[..., i+1, :] -= force[..., i, :]

        normal_relative_vectors = relative_vectors[..., ::-1]
        normal_relative_vectors[..., -1] *= -1

        force = k_angle/2 * np.sin(angles - theta_angle)

        for i in range(1, n_monomers - 1):
            f = force[..., i-1][:, None]
            d_1 = distances[..., i-1][..., None]
            d_3 = distances[..., i][..., None]

            force_1 = f/d_1**2*normal_relative_vectors[..., i-1, :]
            force_3 = f/d_3**2*normal_relative_vectors[..., i, :]
            force_2 = -force_1 - force_3

            forces[..., i-1, :] += force_1
            forces[..., i+1, :] += force_3
            forces[..., i, :] += force_2

        return forces.reshape(input_shape)


    def evaluate_bias_potential_from_cv(self, cv_values, cv_bias_centers):
        """Evaluates the bias potential given cv values of samples and bias centers

        Parameters
        ----------
        cv_values : np.ndarray | torch.Tensor | float
            The cv value(s) for which to compute the bias potential
        cv_bias_centers : np.ndarray | torch.Tensor | float
            The bias center(s) belonging to the window(s) in which to evaluate the bias potential

        Returns
        -------
        np.ndarray | torch.Tensor | float
            The evaluated bias potential energy value(s)
        """
        return 0.5*self.k_bias*(cv_values - cv_bias_centers)**2
    

    def evaluate_bias_force_from_cv(self, cv_values, cv_gradient_values, cv_bias_centers):
        """Evaluates the bias force given cv values and cv gradient values of samples and bias centers

        Parameters
        ----------
        cv_values : np.ndarray | torch.Tensor | float
            The cv value(s) for which to compute the bias force
        cv_gradient_values : np.ndarray | torch.Tensor
            The cv gradient values at the coordinates of the sample(s)
        cv_bias_centers : np.ndarray | torch.Tensor | float
            The bias center(s) belonging to the window(s) in which to evaluate the bias force

        Returns
        -------
        np.ndarray | torch.Tensor
            The evaluated bias force(s)
        """
        if cv_values.ndim == 1 and cv_gradient_values.ndim == 2:
            return -self.k_bias * (cv_values - cv_bias_centers)[:, None] * cv_gradient_values
        else:
            return -self.k_bias * (cv_values - cv_bias_centers) * cv_gradient_values


    def evaluate_bias_potential(self, coords, cv_bias_centers):
        """Evaluates the bias potential given cartesian coordinates. The coordinates belonging to the same polymer must be in the same 1D array

        Parameters
        ----------
        coords : np.ndarray | torch.Tensor
            The cartesian coordinates of the sample(s)
        cv_bias_centers : np.ndarray | torch.Tensor | float
            The bias centers of the windows in which to evaluate the bias potential

        Returns
        -------
        np.ndarray | torch.Tensor | float
            The evaluated bias potential
        """
        if self.cv_function == None:
            raise ValueError("No cv function callable was passed to the init function")
        return self.evaluate_bias_potential_from_cv(self.cv_function(coords), cv_bias_centers)
    

    def evaluate_bias_force(self, coords, cv_bias_centers):
        """Evaluates the bias force given cartesian coordinates. The coordinates belonging to the same polymer must be in the same 1D array.

        Parameters
        ----------
        coords : np.ndarray | torch.Tensor
            The cartesian coordinates of the sample(s)
        cv_bias_centers : np.ndarray | torch.Tensor | float
            The bias centers of the windows in which to evaluate the bias force

        Returns
        -------
        np.ndarray | torch.Tensor
            The evaluated bias force
        """
        if self.cv_function == None:
            raise ValueError("No cv function callable was passed to the init function")
        if self.cv_gradient_function == None:
            raise ValueError("No cv gradient function callable was passed to the init function")
        return self.evaluate_bias_force_from_cv(self.cv_function(coords), self.cv_gradient_function(coords), cv_bias_centers)


    def evaluate(self, coords_with_bias_centers, automatic_coordinate_conversion=True):
        """Evaluates the total potential given coordinates of the polymer sample(s) along with their bias centers. If the given coordinates's
        dimensionality does not match the expected dimensionality for cartesian coordinates, the function assumes that they are given
        as internal coordinates and can try to convert them to cartesian coordinates. If the dimensionality also does not match internal coordinates,
        an error is raised.

        Parameters
        ----------
        coords_with_bias_centers : np.ndarray | torch.Tensor
            The coordinates of the sample(s). All coordinates belonging to the same polymer must be given inside the same
            1D array, which may be part of a larger array containing the coordinates of multiple polymers. The bias center(s) must
            be given as the last entry of each polymer's coordinate array.
        automatic_coordinate_conversion : bool, optional
            If this is True, coordinates are automatically converted from internal to cartesian before potential is evaluated if the function detects
            the given coordinates to be internal
            
        Returns
        -------
        np.ndarray | torch.Tensor | float
            The evaluated total potential for the given sample(s)
        """
        coords = coords_with_bias_centers[..., :-1]
        bias_centers = coords_with_bias_centers[..., -1]

        if coords.shape[-1] == self.n_monomers*2-3 and automatic_coordinate_conversion:
            # Input is internal coordinates, automatically convert to cartesian coordinates before evaluating
            coords = Polymer.internal_to_cartesian(coords)
        elif coords.shape[-1] != self.n_monomers*self.dimensions:
            raise ValueError("Dimensionality of coordinates does not match system dimensionality")

        return self.evaluate_unbiased(coords) + self.evaluate_bias_potential(coords, bias_centers)
    
    def evaluate_force(self, coords_with_bias_centers, automatic_coordinate_conversion=True):
        """Evaluates the total force given coordinates of the polymer sample(s) along with their bias centers. If the given coordinates's
        dimensionality does not match the expected dimensionality for cartesian coordinates, the function assumes that they are given
        as internal coordinates and can try to convert them to cartesian coordinates. If the dimensionality also does not match internal coordinates,
        an error is raised.

        Parameters
        ----------
        coords_with_bias_centers : np.ndarray | torch.Tensor
            The coordinates of the sample(s). All coordinates belonging to the same polymer must be given inside the same
            1D array, which may be part of a larger array containing the coordinates of multiple polymers. The bias center(s) must
            be given as the last entry of each polymer's coordinate array.
        automatic_coordinate_conversion : bool, optional
            If this is True, coordinates are automatically converted from internal to cartesian before potential is evaluated if the function detects
            the given coordinates to be internal
            
        Returns
        -------
        np.ndarray | torch.Tensor
            The evaluated total force for the given sample(s)
        """
        coords = coords_with_bias_centers[..., :-1]
        bias_centers = coords_with_bias_centers[..., -1]

        if coords.shape[-1] == self.n_monomers*2-3 and automatic_coordinate_conversion:
            # Input is internal coordinates, automatically convert to cartesian coordinates before evaluating
            coords = Polymer.internal_to_cartesian(coords)
        elif coords.shape[-1] != self.n_monomers*self.dimensions:
            raise ValueError("Dimensionality of coordinates does not match system dimensionality")

        return self.evaluate_force_unbiased(coords) + self.evaluate_bias_force(coords, bias_centers)

    @staticmethod
    def cartesian_to_internal(coords, jacobian_regularization=0.0):
        """Converts cartesian coordinates to internal coordinates. Internal coordinates are bond lengths and angles
        between three consecutive monomers.

        Parameters
        ----------
        coords : np.ndarray | torch.Tensor
            The cartesian coordinates to convert to internal coordinates. All coordinates belonging to the same polymer must be given inside the same
            1D array, which may be part of a larger array containing the coordinates of multiple polymers.
        jacobian_regularization : float, optional
            Number added before evaluating the logarithm during computation of the log determinant of the transformation. This prevents
            overflow in case of very small numbers common during first training epochs, by default 0.0

        Returns
        -------
        np.ndarray | torch.Tensor
            The internal coordinates of the M polymers, each with N monomers. Shape is (M, 2*N-3) and the last axis contains first the N - 1 bonds and
            then the N - 2 angles.
        np.ndarray | torch.Tensor | float
            The log determinant of the transformation
        """
        coords = coords.reshape(coords.shape[0], -1, 2) if coords.ndim == 2 else coords.reshape(-1, 2)

        relative_vectors = coords[..., 1:, :] - coords[..., :-1, :]

        distances = np.linalg.norm(relative_vectors, axis=-1) if isinstance(coords, np.ndarray) else torch.linalg.norm(relative_vectors, dim=-1)
        angles = -vector2D_angle(relative_vectors[..., :-1, :], relative_vectors[..., 1:, :])

        # angles -= np.pi
        # angles[angles <= -np.pi] += 2*np.pi

        if isinstance(coords, np.ndarray):
            logdet = -np.log(distances[..., 1:] + jacobian_regularization).sum(axis=-1)
        else:
            logdet = -torch.log(distances[..., 1:] + jacobian_regularization).sum(dim=-1)

        return (np.hstack([distances, angles]), logdet) if isinstance(coords, np.ndarray) else (torch.hstack([distances, angles]).to(coords.device), logdet)


    @staticmethod
    def internal_to_cartesian(coords, jacobian_regularization=0.0):
        """Converts internal coordinates to cartesian coordinates. Internal coordinates are bond lengths and angles
        between three consecutive monomers. To achieve rotational symmetry, the first monomer of each polymer is placed
        at the origin (0, 0) and the second monomer is placed on the x-axis at the respective distance d given by the first
        bond length, so at (d, 0).

        Parameters
        ----------
        coords : np.ndarray | torch.Tensor
            The internal coordinates to convert to cartesian coordinates. All coordinates belonging to the same polymer must be given inside the same
            1D array, which may be part of a larger array containing the coordinates of multiple polymers. Each 1D array must have 2*N - 3 entries, first
            the N - 1 bonds and then the N - 2 angles.
        jacobian_regularization : float, optional
            Number added before evaluating the logarithm during computation of the log determinant of the transformation. This prevents
            overflow in case of very small numbers common during first training epochs, by default 0.0

        Returns
        -------
        np.ndarray | torch.Tensor
            The cartesian coordinates of the M polymers, each with N monomers. Shape is (M, 2*N).
        np.ndarray | torch.Tensor | float
            The log determinant of the transformation
        """
        npy = isinstance(coords, np.ndarray)

        if npy:
            cartesian = np.zeros(list(coords.shape[:-1]) + [coords.shape[-1] + 3])
        else:
            cartesian = torch.zeros(list(coords.shape[:-1]) + [coords.shape[-1] + 3], dtype=torch.float32, device=coords.device)
        N = (coords.shape[-1]+3)//2

        distances = np.abs(coords[..., :N-1]) if npy else torch.abs(coords[..., :N-1])
        angles = coords[..., N-1:]

        cartesian[..., 2] = distances[..., 0] # Second atom is at (l_1, 0)

        if npy:
            relative_vectors = np.tile([1.0, 0.0], (coords.shape[0], N-2, 1) if coords.ndim == 2 else (N-2, 1))
            relative_vectors = vector2D_rotate(relative_vectors, np.cumsum(angles, axis=-1))*distances[..., 1:][..., None]
        else:
            relative_vectors = torch.tile(torch.tensor([1.0, 0.0], dtype=torch.float32), (coords.shape[0], N-2, 1) if coords.ndim == 2 else (N-2, 1)).to(coords.device)
            relative_vectors = vector2D_rotate(relative_vectors, torch.cumsum(angles, dim=-1))*distances[..., 1:][..., None]

        coordinates = np.cumsum(relative_vectors, axis=-2) if npy else torch.cumsum(relative_vectors, dim=-2)

        logdet = np.log(distances[..., 1:] + jacobian_regularization).sum(axis=-1) if npy else torch.log(distances[..., 1:] + jacobian_regularization).sum(dim=-1)

        if coords.ndim == 1:
            coordinates = coordinates.flatten()
        else:
            coordinates = coordinates.reshape((coordinates.shape[0], -1))

        coordinates[..., ::2] += distances[..., 0] if coords.ndim == 1 else distances[..., None, 0]

        cartesian[..., 4:] = coordinates

        return cartesian, logdet


    @staticmethod
    def reroll_bonds(coords, in_place=False):
        """Rerolls the bonds of the given sample(s) by a random number of permutation such that the resulting polymers are guaranteed to be different
        from the input polymers

        Parameters
        ----------
        coords : np.ndarray
            Coordinates of the sample(s) given as cartesian coordinates
        in_place : bool, optional
            Whether this operation should be performed in-place or a copy should be made, by default False

        Returns
        -------
        np.ndarray
            The resulting polymer(s) with rerolled bonds
        """
        flatten = coords.ndim == 1
        if flatten:
            coords = coords[None, :]

        coords = (coords.copy() if not in_place else coords).reshape((coords.shape[0], -1, 2))
        roll_indices = np.array([np.roll(np.arange(coords.shape[1]), np.random.randint(1, coords.shape[1])) for _ in range(coords.shape[0])])
        for i in range(coords.shape[0]):
            coords[i] = coords[i, roll_indices[i]]

        return coords.reshape((coords.shape[0], -1)) if not flatten else coords.flatten()
    

    @staticmethod
    def mirror(coords, probability=1.0, in_place=False):
        """Mirrors given polymer sample(s) along the x-axis with given probability

        Parameters
        ----------
        coords : np.ndarray
            The cartesian coordinates of the polmer(s) to mirror
        probability : float, optional
            The probability with which to mirror each polymer. Whether or not a polymer is mirrored is decided for each polymer, by default 1.0
        in_place : bool, optional
            Whether this operation should be done in-place or a copy of the coordinates should be made, by default False

        Returns
        -------
        np.ndarray
            The resulting configuation(s)
        """
        if not in_place:
            coords = coords.copy()
        
        flatten = coords.ndim == 1
        if flatten:
            coords = coords[None, :]

        coords_to_mirror = np.random.random(coords.shape[0]) < probability
        coords[coords_to_mirror, ::2] = -coords[coords_to_mirror, ::2]

        return coords if not flatten else coords.flatten()


    @staticmethod
    def reverse(coords, probability=1.0, in_place=False):
        """Reverses given polymer sample(s) coordinates with given probability

        Parameters
        ----------
        coords : np.ndarray
            The cartesian coordinates of the polmer(s) to reverse
        probability : float, optional
            The probability with which to reverse each polymer's coordinates. Whether or not a polymer is reversed is decided for each polymer, by default 1.0
        in_place : bool, optional
            Whether this operation should be done in-place or a copy of the coordinates should be made, by default False

        Returns
        -------
        np.ndarray
            The resulting configuation(s)
        """
        if not in_place:
            coords = coords.copy()
        
        flatten = coords.ndim == 1
        if flatten:
            coords = coords[None, :]

        coords = coords.reshape((coords.shape[0], -1, 2))

        coords_to_reverse = np.random.random(coords.shape[0]) < probability
        coords[coords_to_reverse, :] = coords[coords_to_reverse, ::-1]

        return coords.reshape((coords.shape[0], -1)) if not flatten else coords.flatten()
        

    @staticmethod
    def constrain_first_angle(internal_coords, in_place=False):
        """Constrains the first angle of the given polymer sample(s) to the interval [0, pi) and mirrors polymers if necessary.

        Parameters
        ----------
        internal_coords : np.ndarray
            Internal coordinates for which to enforce the constraint
        in_place : bool, optional
            Whether or not this operation should be performed in-place or a copy of the coordinates should be made, by default False

        Returns
        -------
        np.ndarray
            The resulting constrained internal coordinates
        """
        if not in_place:
            internal_coords = internal_coords.copy()
        
        flatten = internal_coords.ndim == 1
        if flatten:
            internal_coords = internal_coords[None, :]

        N = (internal_coords.shape[-1]+3)//2

        internal_coords[internal_coords[:, N-1] < 0, N-1:] *= -1 # Mirror if first angle is < 0

        return internal_coords if not flatten else internal_coords.flatten()

    @staticmethod
    def constrain_first_angle_cartesian(cartesian_coords):
        """Same as constrain_first_angle but for cartesian coordinates. Will first transform from cartesian to internal,
        apply constraing and then back to cartesian coordinates

        Parameters
        ----------
        cartesian_coords : np.ndarray | torch.Tensor
            The input cartesian coordinates

        Returns
        -------
        np.ndarray | torch.Tensor
            See docstring of constrain_first_angle
        """
        internal, _ = Polymer.cartesian_to_internal(cartesian_coords)
        Polymer.constrain_first_angle(internal, in_place=True)
        return Polymer.internal_to_cartesian(internal)[0]

    @staticmethod
    def radius_of_gyration_2D(coords):
        """Computes the radius of gyration for given polymer(s) in 2D

        Parameters
        ----------
        coords : np.ndarray | torch.Tensor
            The cartesian coordinates of the polymer(s) for which to compute the radius of gyration

        Returns
        -------
        np.ndarray | torch.Tensor | float
            The computed radius of gyration
        """
        npy = isinstance(coords, np.ndarray)
        if coords.ndim == 1:
            coords = coords.reshape((-1, 2))
        else:
            coords = coords.reshape((coords.shape[0], -1, 2))
        com = np.mean(coords, axis=-2) if npy else torch.sum(coords, dim=-2)/coords.shape[-2]

        distances = coords - (com if coords.ndim == 2 else com[:, None])
        distances = np.linalg.norm(distances, axis=-1)**2 if npy else torch.norm(distances, dim=-1)**2

        return np.sqrt(np.mean(distances, axis=-1)) if npy else torch.sqrt(torch.sum(distances, dim=-1)/distances.shape[-1])


    @staticmethod
    def radius_of_gyration_2D_gradient(coords):
        """Computes the gradient of the radius of gyration for given polymer(s) in 2D

        Parameters
        ----------
        coords : np.ndarray | torch.Tensor
            The cartesian coordinates of the polymer(s) for which to compute the gradient of the radius of gyration

        Returns
        -------
        np.ndarray | torch.Tensor
            The computed gradient(s) of the radius of gyration
        """
        npy = isinstance(coords, np.ndarray)
        if coords.ndim == 1:
            coords = coords.reshape((-1, 2))
        else:
            coords = coords.reshape((coords.shape[0], -1, 2))
        com = np.mean(coords, axis=-2) if npy else torch.sum(coords, dim=-2)/coords.shape[-2]

        relative_vectors = coords - (com if coords.ndim == 2 else com[:, None])
        distances = np.linalg.norm(relative_vectors, axis=-1)**2 if npy else torch.norm(relative_vectors, dim=-1)**2

        R_G = np.sqrt(np.mean(distances, axis=-1)) if npy else torch.sqrt(torch.sum(distances, dim=-1)/distances.shape[-1])

        gradients = relative_vectors/(coords.shape[-2]*R_G)

        if coords.ndim == 2:
            return gradients.flatten()
        return gradients.reshape((coords.shape[0], -1))


    @staticmethod
    def plot_2D(configurations, sigma=1.0, fig=None, ax=None, grid=False, xlim=None, ylim=None, title="", figsize=(10*TO_CM, 10*TO_CM), dpi=200, show=True):
        """Creates a plot of given polymer(s) in 2D

        Parameters
        ----------
        configurations : np.ndarray
            Cartesian coordinates of the polymer(s) to plot
        sigma : float, optional
            LJ sigma value, which determines the size of the monomers, by default 1.0
        fig : matplotlib.figure.Figure, optional
            Optional existing figure object to use for plotting, by default None
        ax : matplotlib.axes.Axes, optional
            Optional existing axes object to use for plotting, by default None
        grid : bool, optional
            Whether or not to plot a grid, by default False
        xlim : tuple[float, float], optional
            The xlim to use. If this is None, the xlim will be determined automatically, by default None
        ylim : tuple[float, float], optional
            The ylim to use. If this is None, the ylim will be determined automatically, by default None
        title : str, optional
            Title for the plot, by default ""
        figsize : tuple[float, float], optional
            Figsize to use, by default (10*TO_CM, 10*TO_CM)
        dpi : int, optional
            DPI to use, by default 200
        show : bool, optional
            If this is True, the figure is shown after creation. If this is False, the figure and axes objects are returned, by default True

        Returns
        -------
        matplotlib.figure.Figure
            The figure object created, only returned if show is False
        matplotlib.axes.Axes
            The axes object creates, only returned if show is False
        """
        matplotlib.rcParams.update({'font.size': 9})

        if fig == None and ax == None:
            fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        elif (fig == None) ^ (ax == None):
            raise ValueError("If one of fig, ax is given, the other one must be given too")

        ax.margins(0, 0)

        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        if configurations.ndim == 1:
            configurations = [configurations]

        radius = (np.power(2, 1/6)*sigma)/2

        color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

        lines = []

        for i, configuration in enumerate(configurations):
            color = color_list[i%10]
            reshaped_config = configuration.reshape(-1, 2)

            lines.append(ax.plot(reshaped_config[:, 0], reshaped_config[:, 1], color="gray", zorder=0, marker="o", mfc=color, mec="gray", mew=1, markersize=1, 
                                 scalex= xlim == None, scaley= ylim == None))

        if not xlim:
            low = ax.get_xlim()[0] - radius
            high = ax.get_xlim()[1] + radius
            padding = max(abs(low)*0.05, abs(high)*0.05)
            ax.set_xlim((low - padding, high + padding))
        if not ylim:
            low = ax.get_ylim()[0] - radius
            high = ax.get_ylim()[1] + radius
            padding = max(abs(low)*0.05, abs(high)*0.05)
            ax.set_ylim((low - padding, high + padding))
        
        marker_size = np.linalg.norm(ax.transData.transform((radius, 0)) - ax.transData.transform((0, 0)))/0.75/2
        for line in lines:
            for l in line:
                l.set_markersize(marker_size)

        ax.set_title(title)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

        if grid:
            ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
            ax.grid(alpha=0.5)

        if show:
            plt.show()
        else:
            return fig, ax

    
    @staticmethod
    def animate_2D(trajectories, interval=500, sigma=1.0, xlim=None, ylim=None, title="", figsize=(10*TO_CM, 10*TO_CM), dpi=200, close=True):
        """Creates an animation of given polymer(s) trajectory/trajectories in 2D

        Parameters
        ----------
        trajectories : np.ndarray
            Trajectories of cartesian coordinates of the polymer(s) to animate
        interval : int, optional
            The time interval between two frames of the animation, directly passed to matplotlib's FuncAnimation, by default 500
        sigma : float, optional
            LJ sigma value, which determines the size of the monomers, by default 1.0
        xlim : tuple[float, float], optional
            The xlim to use. If this is None, the xlim will be determined automatically, by default None
        ylim : tuple[float, float], optional
            The ylim to use. If this is None, the ylim will be determined automatically, by default None
        title : str, optional
            Title for the animation. If this is an empty string, the title will be the current frame and progress of the animation, by default ""
        figsize : tuple[float, float], optional
            Figsize to use, by default (10*TO_CM, 10*TO_CM)
        dpi : int, optional
            DPI to use, by default 200
        close : bool, optional
            If this is True, the animation will be closed after it is created, by default True

        Returns
        -------
        matplotlib.animation.FuncAnimation
            FuncAnimation object created
        """
        if trajectories.ndim == 2:
            trajectories = trajectories[:, None]
        elif trajectories.ndim == 1:
            raise ValueError("trajectories must be an array of either 2 or 3 dimensions")

        trajectories = trajectories.reshape((trajectories.shape[0], trajectories.shape[1], trajectories.shape[2]//2, 2))

        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

        ax.margins(0, 0)

        radius = (np.power(2, 1/6)*sigma)/2

        if not xlim:
            low = np.min(trajectories[:, :, :, 0]) - radius
            high = np.max(trajectories[:, :, :, 0]) + radius
            padding = max(abs(low)*0.05, abs(high)*0.05)
            xlim = (low - padding, high + padding)
        ax.set_xlim(xlim)
        
        if not ylim:
            low = np.min(trajectories[:, :, :, 1]) - radius
            high = np.max(trajectories[:, :, :, 1]) + radius
            padding = max(abs(low)*0.05, abs(high)*0.05)
            ylim = (low - padding, high + padding)
        ax.set_ylim(ylim)
        
        marker_size = np.linalg.norm(ax.transData.transform((radius, 0)) - ax.transData.transform((0, 0)))/0.75/2

        color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

        lines = []

        for i, configuration in enumerate(trajectories[0]):
            color = color_list[i%10]
            lines.append(ax.plot(configuration[:, 0], configuration[:, 1], color="gray", zorder=0, marker="o", mfc=color, mec="gray", mew=1, markersize=marker_size, scalex=False, scaley=False)[0])

        ax.set_title(title if len(title) > 0 else f"Frame 1/{len(trajectories)}")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

        def update(frame, trajectories, lines):
            for configuration, line in zip(trajectories[frame], lines):
                line.set_xdata(configuration[:, 0])
                line.set_ydata(configuration[:, 1])
            ax.set_title(f"Frame {frame}/{len(trajectories)} ({frame/len(trajectories)*100:0.1f}%)")
            return lines
        
        animation = matplotlib.animation.FuncAnimation(fig, update, frames=range(0, len(trajectories)), fargs=(trajectories, lines), interval=interval, repeat=True)

        if close:
            plt.close()

        return animation

    @staticmethod
    def get_polymer_class(configs : np.ndarray) -> np.ndarray:
        if flatten := configs.ndim == 1:
            configs = configs[None, :]

        if configs.shape[-1] == 14:
            configs, _ = Polymer.cartesian_to_internal(configs)

        N = (configs.shape[-1]+3)//2
        angles = configs[..., N-1:]*180/np.pi

        angles /= 60
        angles = np.round(angles).astype(int)
        angles[angles == -3] = -2
        angles[angles == 3] = 2
        angles += 2

        lut = np.array([4, 3, 0, 2, 1])

        angles = lut[angles]

        return angles if not flatten else angles.flatten().item()