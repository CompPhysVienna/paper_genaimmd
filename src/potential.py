import torch
from typing import Union
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

from abc import ABC, abstractmethod

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


class WolfeQuapp(Potential):
    """
    Implementation of the rotated Wolfe-Quapp potential
    """

    def __init__(self, h=6.76245):
        """Wolfe-Quapp potential

        Parameters
        ----------
        h : float, optional
            Barrier height in units of k_B T, by default 6.76245
        """ 
        self.h = h

        self.x_4 = 0.969233
        self.x_3_y = -0.480614
        self.x_2_y_2 = 0.184602
        self.x_2 = -2.155285
        self.x_y_3 = 0.480614
        self.x_y = -1.46487
        self.x = -0.285146
        self.y_4 = 0.969233
        self.y_2 = -3.84471
        self.y = 0.136719

        self.A = np.array([-1.30096, -1.3331])
        self.B = np.array([1.349498, 1.31873])

    @property
    def h(self):
        """Setter for barrier height

        Returns
        -------
        self._h : float
            New barrier height
        """
        return self._h
    
    @h.setter
    def h(self, value):
        """Setter for barrier height

        Parameters
        ----------
        value : float
            New barrier height, must be greater than 0
        """
        assert value > 0, "Height must be > 0"
        self._h = value
        self._s = self._h / 6.76245

    def evaluate(self, coords : Union[torch.tensor, np.ndarray]) -> Union[Union[torch.tensor, np.ndarray], float]:
        """Evaluates the Wolfe-Quapp potential with barrier height self.h at given coordinates

        Parameters
        ----------
        coords : Union[torch.tensor, np.ndarray]
            The coordinates at which to evaluate the potential. Can be either a 1D array containing a single pair of x_1 and
            x_2 coordinates or a 2D array where each row is a pair of x_1 and x_2 coordinates.

        Returns
        -------
        Union[Union[torch.tensor, np.ndarray], float]
            The value of the potential at the given coordinates. If input is 1D, returns a float, otherwise returns a 1D array.
        """
        x = coords[..., 0]
        y = coords[..., 1]

        return self._s * (self.x_4 * x**4 + self.x_3_y * x**3 * y + self.x_2_y_2 * x**2 * y**2 + self.x_2 * x**2 + self.x_y_3 * x * y**3 + self.x_y * x * y \
              + self.x * x + self.y_4 * y**4 + self.y_2 * y**2 + self.y * y) + self.h
    
    def evaluate_force(self, coords : Union[torch.tensor, np.ndarray]) -> Union[torch.tensor, np.ndarray]:
        """Evaluates the force acting on (a) particle(s) of unit mass at given position(s) in the potential

        Parameters
        ----------
        coords : Union[torch.tensor, np.ndarray]
            The coordinates (either 1D or 2D numpy array or pytorch tensor) of the particle(s)

        Returns
        -------
        Union[torch.tensor, np.ndarray]
            The evaluated force(s) acting on the particle(s) as an array of same dimensions as coords.
        """
        x = coords[..., 0]
        y = coords[..., 1]

        x_forces = -self._s * (4*self.x_4 * x**3 + 3*self.x_3_y * x**2 * y + 2*self.x_2_y_2 * x * y**2 + 2*self.x_2 * x + self.x_y_3 * y**3 + self.x_y * y \
              + self.x)
        y_forces = -self._s * (self.x_3_y * x**3 + 2*self.x_2_y_2 * x**2 * y + 3*self.x_y_3 * x * y**2 + self.x_y * x \
              + 4*self.y_4 * y**3 + 2*self.y_2 * y + self.y)

        return np.array([x_forces, y_forces], dtype=coords.dtype).T if isinstance(coords, np.ndarray) else torch.tensor([x_forces, y_forces], dtype=coords.dtype).T.to(coords.device)


    def plot_PES(self, fig=None, ax=None, xmin=-2.3, xmax=2.3, ymin=-2.3, ymax=2.3, levels=30, plot_labels=False, title="", fill=True, plot_clabels=True):
        """Creates a plot of the potential energy surface of the Wolfe-Quapp potential

        Parameters
        ----------
        ax : matplotlib.pyplot.axes, optional
            Axes object to create plot in. If None, new figure and axes will be created, by default None
        xmin : float, optional
            Lower bound for x_1, by default -2.3
        xmax : float, optional
            Upper bound for x_1, by default 2.3
        ymin : float, optional
            Lower bound for x_2, by default -2.3
        ymax : float, optional
            Upper bound for x_2, by default -2.3
        levels : int, optional
            Levels for contour plot, by default 30
        plot_labels : bool, optional
            If True, plots markers at the minima of the potential, by default False
        title : str, optional
            Title for the plot, by default ""
        fill : bool, optional
            Whether to color the levels and add a colorbar, by default True

        Returns
        -------
        Tuple[matplotlib.pyplot.figure, matplotlib.pyplot.axes]
            If ax == None, the created figure and axes objects are returned
            Otherwise, the input fig, ax is returned
        """

        if ax == None:
            TO_CM = 1/2.54
            fig, ax = plt.subplots(1, 1, figsize=(12*TO_CM, 10*TO_CM), dpi=200)

        x = np.arange(-xmax, xmax, 0.01)
        y = np.arange(-ymax, ymax, 0.01)
        xv, yv = np.meshgrid(x, y)

        potential_values = self.evaluate(np.vstack([xv.flatten(), yv.flatten()]).T).reshape(xv.shape)

        if fill:
            contour = ax.contourf(xv, yv, potential_values, cmap="RdYlBu_r", levels=levels)
            cbar = fig.colorbar(contour, ax=ax)
            cbar.set_label("Energy ($k_B T$)")

        lines = ax.contour(xv, yv, potential_values, levels=levels, colors="black", linewidths=0.3)

        if plot_clabels:
            ax.clabel(lines, levels=[l for l in lines.levels if l <= 6], inline=True, fontsize=5, fmt="%.0f")

        if plot_labels:
            ax.scatter(*self.A, color="white", marker="x")
            ax.text(self.A[0] + 0.15, self.A[1] -0.05, "A", color="white")

            ax.scatter(*self.B, color="white", marker="x")
            ax.text(self.B[0]+0.15, self.B[1] - 0.05, "B", color="white")

        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title(title)

        ax.set_aspect('equal', adjustable='box')

        ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
        
        return fig, ax
    

class WolfeQuappBiased(WolfeQuapp):
    """Rotated and biased Wolfe-Quapp potential
    """

    def __init__(self, h=6.76245, k_bias=100.0, cv_function=None, cv_gradient_function=None):
        """Initializes potential class

        Parameters
        ----------
        h : float, optional
            Barrier height, by default 6.76245
        k_bias : float, optional
            Bias force constant, by default 100.0
        cv_function : callable, optional
            Collective variable along which the bias centers lie, transforms given cartesian coordinates to collective variable
            Takes one parameter (coords), which is a NumPy array of coordinates and returns the corresponding collective variable values

            If None, the default CV for the WolfeQuapp potential is used, which is a projection of coordinates onto the line f(x) = 0.25*x
        
        cv_gradient_function : callable, optional
            Gradient of collective variable for bias force evaluation. If cv_function is None, this will be the gradient of the default CV
            function. If cv_function is specified and cv_gradient_function is None, gradient of the cv function will be determined automatically
            using central differences.

            If specified, this function takes one argument (the coordinates as a NumPy array) and returns the corresponding gradient of the
            collective variable values.
        """
        super().__init__(h)
        self.k_bias = k_bias
        if cv_function != None:
            self.cv = cv_function
            if cv_gradient_function != None:
                self.cv_gradient = cv_gradient_function
        else:
            self.cv = lambda coords: 0.25*coords[..., 0] + coords[..., 1]
            self.cv_gradient = lambda coords: np.array([0.25, 1]) if coords.ndim == 1 else np.array([[0.25, 1] for _ in range(len(coords))])


    def cv_gradient(self, coords):
        """Computes the gradient of the collective variable self.cv using central differences. This is only used if cv_function is
        specified in the init function but cv_gradient_function is None.

        Parameters
        ----------
        coords : np.ndarray
            The coordinates at which to evaluate the gradient

        Returns
        -------
        np.ndarray
            Gradient of the cv values corresponding to the coordinates
        """
        H = 0.01
        flatten = coords.ndim == 1
        if flatten:
            coords = coords.reshape((1, -1))

        N = coords.shape[1]

        step_matrix = H*np.eye(N)
        coordinate_matrix = np.tile(coords[:, None, :], (N, 1))

        upper = coordinate_matrix + step_matrix
        lower = coordinate_matrix - step_matrix

        gradients = (self.cv(upper) - self.cv(lower))/(2*H)

        return gradients if not flatten else gradients.flatten()   
        

    def evaluate_unbiased(self, coords):
        """Evaluates the unbiased Wolfe-Quapp potential with barrier height self.h at given coordinates

        Parameters
        ----------
        coords : Union[torch.tensor, np.ndarray]
            The coordinates at which to evaluate the potential. Can be either a 1D array containing a single pair of x_1 and
            x_2 coordinates or a 2D array where each row is a pair of x_1 and x_2 coordinates.

        Returns
        -------
        Union[Union[torch.tensor, np.ndarray], float]
            The value of the potential at the given coordinates. If input is 1D, returns a float, otherwise returns a 1D array.
        """
        return super().evaluate(coords)
    
    def evaluate_force_unbiased(self, coords):
        """Evaluates the unbiased force acting on (a) particle(s) of unit mass at given position(s) in the potential

        Parameters
        ----------
        coords : Union[torch.tensor, np.ndarray]
            The coordinates (either 1D or 2D numpy array or pytorch tensor) of the particle(s)

        Returns
        -------
        Union[torch.tensor, np.ndarray]
            The evaluated force(s) acting on the particle(s) as an array of same dimensions as coords.
        """
        return super().evaluate_force(coords)
    
    def evaluate_bias_potential_from_cv(self, cv_values, cv_value_bias_center):
        """Evaluates the harmonic bias potential at given cv values. Bias potential U_b(cv, r) at cv value cv and bias center r is of the form

            U_b(cv, r) = k/2 * (cv - r)^2

        Parameters
        ----------
        cv_values : np.ndarray
            cv values at which to evaluate bias potential
        cv_value_bias_center : np.ndarray
            Bias centers at which to evaluate bias potential transformed to collective variable

        Returns
        -------
        np.ndarray
            Evaluated bias potential
        """
        return 0.5*self.k_bias*(cv_values - cv_value_bias_center)**2

    def evaluate_bias_force_from_cv(self, cv_values, cv_gradient_values, cv_value_bias_center):
        """Evaluates the harmonic bias force at given cv values. Bias force F_b(cv, r') at cv value cv and bias center r' is of the form

            F_b(cv, r) = - k * (cv - r) * cv'

           where cv' is the gradient of the collective variable

        Parameters
        ----------
        cv_values : np.ndarray
            cv values at which to evaluate bias force
        cv_gradient_values : np.ndarray
            Values of the gradient of the cv
        cv_value_bias_center : np.ndarray
            Bias centers at which to evaluate bias force transformed to collective variable

        Returns
        -------
        np.ndarray
            Evaluated bias force
        """
        if cv_values.ndim == 1 and cv_gradient_values.ndim == 2:
            return -self.k_bias * (cv_values - cv_value_bias_center)[:, None] * cv_gradient_values
        else:
            return -self.k_bias * (cv_values - cv_value_bias_center) * cv_gradient_values
    
    def evaluate_bias_potential(self, coords, cv_value_bias_center):
        """Evaluates the harmonic bias potential at given coordinates. Bias potential U_b(x, r) at position x and bias center r is of the form

            U_b(x, r) = k/2 * (cv(x) - r)^2

        Parameters
        ----------
        coords : np.ndarray
            Coordinates at which to evaluate bias potential
        cv_value_bias_center : np.ndarray
            Bias centers at which to evaluate bias potential transformed to collective variable

        Returns
        -------
        np.ndarray
            Evaluated bias potential
        """
        return self.evaluate_bias_potential_from_cv(self.cv(coords), cv_value_bias_center)

    def evaluate_bias_force(self, coords, cv_value_bias_center):
        """Evaluates the harmonic bias force at given coordinates. Bias force F_b(x, r') at position x and bias center r' is of the form

            F_b(x, r) = - k * (cv(x) - r) * cv'(x)

           where cv'(x) is the gradient of the collective variable evaluated at x

        Parameters
        ----------
        coords : np.ndarray
            Coordinates at which to evaluate bias force
        cv_value_bias_center : np.ndarray
            Bias centers at which to evaluate bias force transformed to collective variable

        Returns
        -------
        np.ndarray
            Evaluated bias force
        """
        return self.evaluate_bias_force_from_cv(self.cv(coords), self.cv_gradient(coords), cv_value_bias_center)
    
    
    def evaluate(self, coords_with_bias_centers):
        """Evaluates the full biased potential, which is a sum of the unbiased potential and the biased potential

        Parameters
        ----------
        coords_with_bias_centers : np.ndarray
            An array of either shape (N, 3) or (3, ), containing coordinates in [..., :-1] and bias centers in [..., -1]

        Returns
        -------
        np.ndarray
            Evaluated full potential
        """
        coords = coords_with_bias_centers[..., :-1]
        bias_centers = coords_with_bias_centers[..., -1]
        return self.evaluate_unbiased(coords) + self.evaluate_bias_potential(coords, bias_centers)
    
    def evaluate_force(self, coords_with_bias_centers):
        """Evaluates the full biased force, which is a sum of the unbiased force and the biased force

        Parameters
        ----------
        coords_with_bias_centers : np.ndarray
            An array of either shape (N, 3) or (3, ), containing coordinates in [..., :-1] and bias centers in [..., -1]

        Returns
        -------
        np.ndarray
            Evaluated full force
        """
        coords = coords_with_bias_centers[..., :-1]
        bias_centers = coords_with_bias_centers[..., -1]
        return self.evaluate_force_unbiased(coords) + self.evaluate_bias_force(coords, bias_centers)
    
    def plot_PES(self, fig=None, ax=None, xmin=-2.3, xmax=2.3, ymin=-2.3, ymax=2.3, levels=30, plot_labels=False, title="", fill=True):
        """Creates a plot of the potential energy surface of the unbiased Wolfe-Quapp potential

        Parameters
        ----------
        ax : matplotlib.pyplot.axes, optional
            Axes object to create plot in. If None, new figure and axes will be created, by default None
        xmin : float, optional
            Lower bound for x_1, by default -2.3
        xmax : float, optional
            Upper bound for x_1, by default 2.3
        ymin : float, optional
            Lower bound for x_2, by default -2.3
        ymax : float, optional
            Upper bound for x_2, by default -2.3
        levels : int, optional
            Levels for contour plot, by default 30
        plot_labels : bool, optional
            If True, plots markers at the minima of the potential, by default False
        title : str, optional
            Title for the plot, by default ""
        fill : bool, optional
            Whether to color the levels and add a colorbar, by default True

        Returns
        -------
        Tuple[matplotlib.pyplot.figure, matplotlib.pyplot.axes]
            If ax == None, the created figure and axes objects are returned
            Otherwise, the input fig, ax is returned
        """

        if ax == None:
            TO_CM = 1/2.54
            fig, ax = plt.subplots(1, 1, figsize=(12*TO_CM, 10*TO_CM), dpi=200)

        x = np.arange(-xmax, xmax, 0.01)
        y = np.arange(-ymax, ymax, 0.01)
        xv, yv = np.meshgrid(x, y)

        potential_values = self.evaluate_unbiased(np.vstack([xv.flatten(), yv.flatten()]).T).reshape(xv.shape)

        if fill:
            contour = ax.contourf(xv, yv, potential_values, cmap="RdYlBu_r", levels=levels)
            cbar = fig.colorbar(contour, ax=ax)
            cbar.set_label("Energy ($k_B T$)")

        ax.contour(xv, yv, potential_values, levels=levels, colors="black", linewidths=0.3)

        if plot_labels:
            ax.scatter(*self.A, color="white", marker="x")
            ax.text(self.A[0] + 0.15, self.A[1] -0.05, "A", color="white")

            ax.scatter(*self.B, color="white", marker="x")
            ax.text(self.B[0]+0.15, self.B[1] - 0.05, "B", color="white")

        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title(title)

        ax.set_aspect('equal', adjustable='box')

        ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
        
        return fig, ax