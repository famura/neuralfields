from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.utils.data
from torch import nn
from torch.nn.utils import convert_parameters

from neuralfields.custom_types import ActivationFunction


class PotentialBased(nn.Module, ABC):
    """Base class for all potential-based recurrent neutral networks."""

    _sqrt_tau: Union[torch.Tensor, nn.Parameter]
    _sqrt_kappa: Union[torch.Tensor, nn.Parameter]

    _potentials_max: Union[float, int] = 100
    """Threshold to clip the potentials symmetrically (at a very large value) for numerical stability."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        activation_nonlin: Union[ActivationFunction, Sequence[ActivationFunction]],
        tau_init: Union[float, int],
        tau_learnable: bool,
        kappa_init: Union[float, int],
        kappa_learnable: bool,
        potentials_init: Optional[torch.Tensor] = None,
        output_size: Optional[int] = None,
        input_embedding: Optional[nn.Module] = None,
        output_embedding: Optional[nn.Module] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Args:
            input_size: Number of input dimensions.
            hidden_size: Number of neurons with potential per hidden layer. For all use cases conceived at this point,
                we only use one recurrent layer. However, there is the possibility to extend the networks to multiple
                potential-based layers.
            activation_nonlin: Nonlinearity used to compute the activations from the potential levels.
            tau_init: Initial value for the shared time constant of the potentials.
            tau_learnable: Whether the time constant is a learnable parameter or fixed.
            kappa_init: Initial value for the cubic decay, pass 0 to disable the cubic decay.
            kappa_learnable: Whether the cubic decay is a learnable parameter or fixed.
            potentials_init: Initial for the potentials, i.e., the network's hidden state.
            output_size: Number of output dimensions. By default, the number of outputs is equal to the number of
                hidden neurons.
            input_embedding: Optional (custom) [Module][torch.nn.Module] to extract features from the inputs.
                This module must transform the inputs such that the dimensionality matches the number of
                neurons of the neural field, i.e., `hidden_size`. By default, a [linear layer][torch.nn.Linear]
                without biases is used.
            output_embedding: Optional (custom) [Module][torch.nn.Module] to compute the outputs from the activations.
                This module must map the activations of shape (`hidden_size`,) to the outputs of shape (`output_size`,)
                By default, a [linear layer][torch.nn.Linear] without biases is used.
            device: Device to move this module to (after initialization).
        """
        # Call torch.nn.Module's constructor.
        super().__init__()

        # For all use cases conceived at this point, we only use one recurrent layer. However, this variable still
        # exists in case somebody in the future wants to try multiple potential-based layers. It will require more
        # changes than increasing this number.
        self.num_recurrent_layers = 1

        self.input_size = input_size
        self._hidden_size = hidden_size // self.num_recurrent_layers  # hidden size per layer
        self.output_size = self._hidden_size if output_size is None else output_size
        self._stimuli_external = torch.zeros(self.hidden_size, device=device)
        self._stimuli_internal = torch.zeros(self.hidden_size, device=device)

        # Create the common layers.
        self.input_embedding = input_embedding or nn.Linear(self.input_size, self._hidden_size, bias=False)
        self.output_embedding = output_embedding or nn.Linear(self._hidden_size, self.output_size, bias=False)

        # Initialize the values of the potentials.
        if potentials_init is not None:
            self._potentials_init = potentials_init.detach().clone().to(device=device)
        else:
            if activation_nonlin is torch.sigmoid:
                self._potentials_init = -7 * torch.ones(1, self.hidden_size, device=device)
            else:
                self._potentials_init = torch.zeros(1, self.hidden_size, device=device)

        # Initialize the potentials' resting level, i.e., the asymptotic level without stimuli.
        self.resting_level = nn.Parameter(torch.randn(self.hidden_size, device=device))

        # Initialize the potential dynamics' time constant.
        self.tau_learnable = tau_learnable
        if tau_init <= 0:
            raise ValueError("The time constant tau must be initialized positive.")
        self._sqrt_tau_init = torch.sqrt(
            torch.as_tensor(tau_init, device=device, dtype=torch.get_default_dtype()).reshape(-1)
        )
        if self.tau_learnable:
            self._sqrt_tau = nn.Parameter(self._sqrt_tau_init)
        else:
            self._sqrt_tau = self._sqrt_tau_init

        # Initialize the potential dynamics' cubic decay.
        self.kappa_learnable = kappa_learnable
        if kappa_init < 0:
            raise ValueError("The cubic decay kappa must be initialized non-negative.")
        self._sqrt_kappa_init = torch.sqrt(
            torch.as_tensor(kappa_init, device=device, dtype=torch.get_default_dtype()).reshape(-1)
        )
        if self.kappa_learnable:
            self._sqrt_kappa = nn.Parameter(self._sqrt_kappa_init)
        else:
            self._sqrt_kappa = self._sqrt_kappa_init

    def extra_repr(self) -> str:
        return f"tau_learnable={self.tau_learnable}, kappa_learnable={self.kappa_learnable}"

    @property
    def param_values(self) -> torch.Tensor:
        """Get the module's parameters as a 1-dimensional array.
        The values are copied, thus modifying the return value does not propagate back to the module parameters.
        """
        return convert_parameters.parameters_to_vector(self.parameters())

    @param_values.setter
    def param_values(self, param: torch.Tensor):
        """Set the module's parameters from a 1-dimensional array."""
        convert_parameters.vector_to_parameters(param, self.parameters())

    @property
    def device(self) -> torch.device:
        """Get the device this model is located on. This assumes that all parts are located on the same device."""
        assert (
            self.input_embedding.weight.device
            == self.resting_level.device
            == self._sqrt_tau.device
            == self._sqrt_kappa.device
        )
        return self.input_embedding.weight.device

    @property
    def hidden_size(self) -> int:
        """Get the number of neurons in the neural field layer, i.e., the ones with the in-/exhibition dynamics."""
        return self.num_recurrent_layers * self._hidden_size

    @property
    def stimuli_external(self) -> torch.Tensor:
        """Get the neurons' external stimuli, resulting from the current inputs.
        This property is useful for recording during a simulation / rollout.
        """
        return self._stimuli_external

    @property
    def stimuli_internal(self) -> torch.Tensor:
        """Get the neurons' internal stimuli, resulting from the previous activations of the neurons.
        This property is useful for recording during a simulation / rollout.
        """
        return self._stimuli_internal

    @property
    def tau(self) -> Union[torch.Tensor, nn.Parameter]:
        r"""Get the timescale parameter, called $\tau$ in the original paper [Amari_77]."""
        return torch.square(self._sqrt_tau)

    @property
    def kappa(self) -> Union[torch.Tensor, nn.Parameter]:
        r"""Get the cubic decay parameter $\kappa$."""
        return torch.square(self._sqrt_kappa)

    @abstractmethod
    def potentials_dot(self, potentials: torch.Tensor, stimuli: torch.Tensor) -> torch.Tensor:
        """Compute the derivative of the neurons' potentials w.r.t. time.

        Args:
            potentials: Potential values at the current point in time, of shape `(hidden_size,)`.
            stimuli: Sum of external and internal stimuli at the current point in time, of shape `(hidden_size,)`.

        Returns:
            Time derivative of the potentials $\frac{dp}{dt}$, of shape `(hidden_size,)`.
        """

    def init_hidden(
        self, batch_size: Optional[int] = None, potentials_init: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, torch.nn.Parameter]:
        """Provide initial values for the hidden parameters. This usually is a zero tensor.

        Args:
            batch_size: Number of batches, i.e., states to track in parallel.
            potentials_init: Initial values for the potentials to override the networks default values with.

        Returns:
            Tensor of shape `(hidden_size,)` if `hidden` was not batched, else of shape `(batch_size, hidden_size)`.
        """
        if potentials_init is None:
            if batch_size is None:
                return self._potentials_init.view(-1)
            return self._potentials_init.repeat(batch_size, 1).to(device=self.device)

        return potentials_init.to(device=self.device)

    @staticmethod
    def _infer_batch_size(inputs: torch.Tensor) -> int:
        """Get the number of batch dimensions from the inputs to the model.
        The batch dimension is assumed to be located at the first axis of the input [tensor][torch.Tensor].

        Args:
            inputs: Inputs to the forward pass, could be of shape `(input_size,)` or `(batch_size, input_size)`.

        Returns:
            The number of batch dimensions a.k.a. the batch size.
        """
        if inputs.dim() == 1:
            return 1
        return inputs.size(0)

    @abstractmethod
    def forward_one_step(
        self, inputs: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the external and internal stimuli, advance the potential dynamics for one time step, and return
        the model's output.

        Args:
            inputs: Inputs of the current time step, of shape `(input_size,)`, or `(batch_size, input_size)`.
            hidden: Hidden state which are for the model in this package the potentials, of shape `(hidden_size,)`, or
                `(batch_size, input_size)`. Pass `None` to leave the initialization to the network which uses
                [init_hidden][neuralfields.PotentialBased.init_hidden] is called.

        Returns:
            The outputs, i.e., the (linearly combined) activations, and the most recent potential values, both of shape
            `(batch_size, input_size)`.
        """

    def forward(self, inputs: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the external and internal stimuli, advance the potential dynamics for one time step, and return
        the model's output for several time steps in a row.

        This method essentially calls [forward_one_step][neuralfields.PotentialBased.forward_one_step] several times
        in a row.

        Args:
            inputs: Inputs of shape `(batch_size, num_steps, dim_input)` to evaluate the network on.
            hidden: Initial values of the hidden states, i.e., the potentials. By default, the network initialized
                the hidden state to be all zeros. However, via this argument one can set a specific initial value
                for the potentials. Depending on the shape of `inputs`, `hidden` is of shape `(hidden_size,)` if
                the input was not batched, else of shape `(batch_size, hidden_size)`.

        Returns:
            The outputs, i.e., the (linearly combined) activations, and all intermediate potential values, both of
            shape `(batch_size, num_steps, dim_output)`.
        """
        # Bring the sequence of inputs into the shape (batch_size, num_steps, dim_input).
        batch_size = PotentialBased._infer_batch_size(inputs)
        inputs = inputs.view(batch_size, -1, self.input_size)  # moved to the desired device by forward_one_step() later

        # If given use the hidden tensor, i.e., the potentials of the last step, else initialize them.
        hidden = self.init_hidden(batch_size, hidden)  # moved to the desired device by forward_one_step() later

        # Iterate over the time dimension. Do this in parallel for all batched which are still along the 1st dimension.
        inputs = inputs.permute(1, 0, 2)  # move time to first dimension for easy iterating
        outputs_all = []
        hidden_all = []
        for inp in inputs:
            outputs, hidden_next = self.forward_one_step(inp, hidden)
            hidden = hidden_next.clone()
            outputs_all.append(outputs)
            hidden_all.append(hidden_next)

        # Return the outputs and hidden states, both stacked along the time dimension.
        return torch.stack(outputs_all, dim=1), torch.stack(hidden_all, dim=1)
