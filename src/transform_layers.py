from potential import Polymer
import torch


class PolymerTransformLayer(torch.nn.Module):
    """Transformation layer in the Polymer model which performs the conversion between cartesian and internal coordinates
    """

    def __init__(self, jacobian_regularization=0.0):
        """Init function for the PolymerTransformLayer

        Parameters
        ----------
        jacobian_regularization : float, optional
            Value that is added before computing the logarithm of the transformation's determinant. Prevents overflow if the
            determinant is very small, which may be the case during the first training epochs, by default 0.0
        """
        super().__init__()
        self.jacobian_regularization = jacobian_regularization
    
    @property
    def relevant_for_reweighting(self):
        return True

    def forward(self, x):
        """This function is only here for the AIMMDNetwork in aimmd.py, since it needs to use this module the intended way. Thus,
        the function is basically just a wrapper for self.F_xz, since this is the direction that transforms from cartesian to
        internal coordinates, which is what the AIMMDNetwork expects.

        Parameters
        ----------
        x : torch.Tensor
            See docstring of self.F_xz

        Returns
        -------
        torch.Tensor
            See docstring of self.F_xz, but only the first return argument
        """
        return self.F_xz(x)[0]

    def F_zx(self, x):
        """Converts internal coordinates to cartesian coordinates

        Parameters
        ----------
        x : torch.Tensor
            Internal cooridnates to convert to cartesian coordinates

        Returns
        -------
        torch.Tensor
            The computed cartesian coordinates
        torch.Tensor
            The logarithm of the transformation's determinant
        """
        cartesian, log_det = Polymer.internal_to_cartesian(x, jacobian_regularization=self.jacobian_regularization)
        return cartesian, log_det
    
    def F_xz(self, x):
        """Converts cartesian coordinates to internal coordinates

        Parameters
        ----------
        x : torch.Tensor
            Cartesian cooridnates to convert to internal coordinates

        Returns
        -------
        torch.Tensor
            The computed internal coordinates
        torch.Tensor
            The logarithm of the transformation's determinant
        """
        internal, logdet = Polymer.cartesian_to_internal(x, jacobian_regularization=self.jacobian_regularization)
        return internal, logdet


class PolymerNormLayer(torch.nn.Module):
    """Normalization layer for the polymer model
    """

    def __init__(self, train_data):
        """Init function for the polymer model normalization layer

        Parameters
        ----------
        train_data : torch.Tensor
            Training data used to compute the mean and standard deviation used for normalization
        """
        super().__init__()
        self.register_buffer("means", train_data.mean(axis=0, keepdim=True))
        self.register_buffer("standard_deviations", train_data.std(axis=0, keepdim=True))

    @property
    def relevant_for_reweighting(self):
        return False
    
    def forward(self, x):
        """This function is only here for the AIMMDNetwork in aimmd.py, since it needs to use this module the intended way. Thus,
        the function is basically just a wrapper for self.F_xz

        Parameters
        ----------
        x : torch.Tensor
            See docstring of self.F_xz

        Returns, but only the first return argument
        -------
        torch.Tensor
            See docstring of self.F_xz, but only the first return argument
        """
        return self.F_xz(x)[0]

    def F_xz(self, x):
        """Subtracts mean and divides by standard deviation in order to normalize in the xz direction

        Parameters
        ----------
        x : torch.Tensor
            Internal coordinates to normalize

        Returns
        -------
        torch.Tensor
            The normalizes internal coordinates
        torch.Tensor
            The log determinant of the transformation
        """
        z = (x -  self.means) / self.standard_deviations 
        logdet = (-torch.log(self.standard_deviations).repeat(x.shape[0], 1)).sum(dim=-1, keepdims=False)
        
        return z, logdet


    def F_zx(self, z):
        """Reverses normalization

        Parameters
        ----------
        z : torch.Tensor
            Normalized internal coordinates

        Returns
        -------
        torch.Tensor
            The un-normalized internal coordinates
        torch.Tensor
            The log determinant of the transformation
        """
            
        x = z * self.standard_deviations + self.means
        logdet = torch.log(self.standard_deviations).repeat(x.shape[0], 1).sum(dim=-1, keepdims=False)
            
        return x, logdet