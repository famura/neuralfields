from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from neuralfields.custom_layers import IndependentNonlinearitiesLayer, _is_iterable, init_param_
from neuralfields.custom_types import ActivationFunction, PotentialsDynamicsType
from neuralfields.potential_based import PotentialBased


def _verify_tau(tau: torch.Tensor) -> None:
    r"""Make sure that the time scaling factor is greater than zero.

    Args:
        tau: Time scaling factor to check.

    Raises:
        `ValueError`: If $\tau \le 0$.
    """
    if not all(tau.view(1) > 0):
        raise ValueError(f"The time constant tau must be > 0, but is {tau}!")


def _verify_kappa(kappa: torch.Tensor) -> None:
    r"""Make sure that the cubic decay factor is greater or equal zero.

    Args:
        kappa: Cubic decay factor to check.

    Raises:
        `ValueError`: If $\kappa < 0$.
    """
    if not all(kappa.view(1) >= 0):
        raise ValueError(f"All elements of the cubic decay kappa must be > 0, but they are {kappa}")


def pd_linear(
    p: torch.Tensor, s: torch.Tensor, h: torch.Tensor, tau: torch.Tensor, **kwargs: torch.Tensor
) -> torch.Tensor:
    r"""Basic proportional dynamics.

    $\tau \dot{p} = s - p$

    Args:
        p: Potential, higher values lead to higher activations.
        s: Stimulus, higher values lead to larger changes of the potentials (depends on the dynamics function).
        h: Resting level, a.k.a. constant offset.
        tau: Time scaling factor, higher values lead to slower changes of the potentials (linear dependency).
        kwargs: Additional parameters to the potential dynamics.

    Returns:
        Time derivative of the potentials $\frac{dp}{dt}$.
    """
    _verify_tau(tau)
    return (s + h - p) / tau


def pd_cubic(
    p: torch.Tensor, s: torch.Tensor, h: torch.Tensor, tau: torch.Tensor, **kwargs: torch.Tensor
) -> torch.Tensor:
    r"""Basic proportional dynamics with additional cubic decay.

    $\tau \dot{p} = s + h - p + \kappa (h - p)^3$

    Args:
        p: Potential, higher values lead to higher activations.
        s: Stimulus, higher values lead to larger changes of the potentials (depends on the dynamics function).
        h: Resting level, a.k.a. constant offset.
        tau: Time scaling factor, higher values lead to slower changes of the potentials (linear dependency).
        kwargs: Additional parameters to the potential dynamics.

    Returns:
        Time derivative of the potentials $\frac{dp}{dt}$.
    """
    _verify_tau(tau)
    _verify_kappa(kwargs["kappa"])
    return (s + h - p + kwargs["kappa"] * torch.pow(h - p, 3)) / tau


def pd_capacity_21(
    p: torch.Tensor, s: torch.Tensor, h: torch.Tensor, tau: torch.Tensor, **kwargs: torch.Tensor
) -> torch.Tensor:
    r"""Capacity-based dynamics with 2 stable ($p=-C$, $p=C$) and 1 unstable fix points ($p=0$) for $s=0$

    $\tau \dot{p} =  s - (h - p) (1 - \frac{(h - p)^2}{C^2})$

    Notes:
        Intended to be used with a sigmoid activation function.

    Args:
        p: Potential, higher values lead to higher activations.
        s: Stimulus, higher values lead to larger changes of the potentials (depends on the dynamics function).
        h: Resting level, a.k.a. constant offset.
        tau: Time scaling factor, higher values lead to slower changes of the potentials (linear dependency).
        kwargs: Additional parameters to the potential dynamics.

    Returns:
        Time derivative of the potentials $\frac{dp}{dt}$.
    """
    _verify_tau(tau)
    return (s - (h - p) * (torch.ones_like(p) - (h - p) ** 2 / kwargs["capacity"] ** 2)) / tau


def pd_capacity_21_abs(
    p: torch.Tensor, s: torch.Tensor, h: torch.Tensor, tau: torch.Tensor, **kwargs: torch.Tensor
) -> torch.Tensor:
    r"""Capacity-based dynamics with 2 stable ($p=-C$, $p=C$) and 1 unstable fix points ($p=0$) for $s=0$

    $\tau \dot{p} =  s - (h - p) (1 - \frac{\left| h - p \right|}{C})$

    The "absolute version" of `pd_capacity_21` has a lower magnitude and a lower oder of the resulting polynomial.

    Notes:
        Intended to be used with a sigmoid activation function.

    Args:
        p: Potential, higher values lead to higher activations.
        s: Stimulus, higher values lead to larger changes of the potentials (depends on the dynamics function).
        h: Resting level, a.k.a. constant offset.
        tau: Time scaling factor, higher values lead to slower changes of the potentials (linear dependency).
        kwargs: Additional parameters to the potential dynamics.

    Returns:
        Time derivative of the potentials $\frac{dp}{dt}$.
    """
    _verify_tau(tau)
    return (s - (h - p) * (torch.ones_like(p) - torch.abs(h - p) / kwargs["capacity"])) / tau


def pd_capacity_32(
    p: torch.Tensor, s: torch.Tensor, h: torch.Tensor, tau: torch.Tensor, **kwargs: torch.Tensor
) -> torch.Tensor:
    r"""Capacity-based dynamics with 3 stable ($p=-C$, $p=0$, $p=C$) and 2 unstable fix points ($p=-C/2$, $p=C/2$)
    for $s=0$

    $\tau \dot{p} =  s - (h - p) (1 - \frac{(h - p)^2}{C^2}) (1 - \frac{(2(h - p))^2}{C^2})$

    Notes:
        Intended to be used with a tanh activation function.

    Args:
        p: Potential, higher values lead to higher activations.
        s: Stimulus, higher values lead to larger changes of the potentials (depends on the dynamics function).
        h: Resting level, a.k.a. constant offset.
        tau: Time scaling factor, higher values lead to slower changes of the potentials (linear dependency).
        kwargs: Additional parameters to the potential dynamics.

    Returns:
        Time derivative of the potentials $\frac{dp}{dt}$.
    """
    _verify_tau(tau)
    return (
        s
        + (h - p)
        * (torch.ones_like(p) - (h - p) ** 2 / kwargs["capacity"] ** 2)
        * (torch.ones_like(p) - ((2 * (h - p)) ** 2 / kwargs["capacity"] ** 2))
    ) / tau


def pd_capacity_32_abs(
    p: torch.Tensor, s: torch.Tensor, h: torch.Tensor, tau: torch.Tensor, **kwargs: torch.Tensor
) -> torch.Tensor:
    r"""Capacity-based dynamics with 3 stable ($p=-C$, $p=0$, $p=C$) and 2 unstable fix points ($p=-C/2$, $p=C/2$)
    for $s=0$.

    $\tau \dot{p} =  \left( s + (h - p) (1 - \frac{\left| (h - p) \right|}{C})
    (1 - \frac{2 \left| (h - p) \right|}{C}) \right)$

    The "absolute version" of `pd_capacity_32` is less skewed due to a lower oder of the resulting polynomial.

    Notes:
        Intended to be used with a tanh activation function.

    Args:
        p: Potential, higher values lead to higher activations.
        s: Stimulus, higher values lead to larger changes of the potentials (depends on the dynamics function).
        h: Resting level, a.k.a. constant offset.
        tau: Time scaling factor, higher values lead to slower changes of the potentials (linear dependency).
        kwargs: Additional parameters to the potential dynamics.

    Returns:
        Time derivative of the potentials $\frac{dp}{dt}$.
    """
    _verify_tau(tau)
    return (
        s
        + (h - p)
        * (torch.ones_like(p) - torch.abs(h - p) / kwargs["capacity"])
        * (torch.ones_like(p) - 2 * torch.abs(h - p) / kwargs["capacity"])
    ) / tau


class SimpleNeuralField(PotentialBased):
    """A simplified version of Amari's potential-based recurrent neural network, without the convolution over time.

    See Also:
        [Luksch et al., 2012] T. Luksch, M. Gineger, M. Mühlig, T. Yoshiike, "Adaptive Movement Sequences and
        Predictive Decisions based on Hierarchical Dynamical Systems", International Conference on Intelligent
        Robots and Systems, 2012.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        potentials_dyn_fcn: PotentialsDynamicsType,
        input_embedding: Optional[nn.Module] = None,
        output_embedding: Optional[nn.Module] = None,
        activation_nonlin: Union[ActivationFunction, Sequence[ActivationFunction]] = torch.sigmoid,
        tau_init: Union[float, int] = 10.0,
        tau_learnable: bool = True,
        kappa_init: Union[float, int] = 1e-3,
        kappa_learnable: bool = True,
        capacity_learnable: bool = True,
        potentials_init: Optional[torch.Tensor] = None,
        init_param_kwargs: Optional[dict] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Args:
            input_size: Number of input dimensions.
            output_size: Number of output dimensions. For this simplified neural fields model, the number of outputs
                is equal to the number of neurons in the (single) hidden layer.
            input_embedding: Optional (custom) [Module][torch.nn.Module] to extract features from the inputs.
                This module must transform the inputs such that the dimensionality matches the number of
                neurons of the neural field, i.e., `hidden_size`. By default, a [linear layer][torch.nn.Linear]
                without biases is used.
            output_embedding: Optional (custom) [Module][torch.nn.Module] to compute the outputs from the activations.
                This module must map the activations of shape (`hidden_size`,) to the outputs of shape (`output_size`,)
                By default, a [linear layer][torch.nn.Linear] without biases is used.
            activation_nonlin: Nonlinearity used to compute the activations from the potential levels.
            tau_init: Initial value for the shared time constant of the potentials.
            tau_learnable: Whether the time constant is a learnable parameter or fixed.
            kappa_init: Initial value for the cubic decay, pass 0 to disable the cubic decay.
            kappa_learnable: Whether the cubic decay is a learnable parameter or fixed.
            capacity_learnable: Whether the capacity is a learnable parameter or fixed.
            potentials_init: Initial for the potentials, i.e., the network's hidden state.
            init_param_kwargs: Additional keyword arguments for the policy parameter initialization. For example,
                `self_centric_init=True` to initialize the interaction between neurons such that they inhibit the
                others and excite themselves.
            device: Device to move this module to (after initialization).
        """
        init_param_kwargs = init_param_kwargs if init_param_kwargs is not None else dict()

        # Create the common layers and parameters.
        super().__init__(
            input_size=input_size,
            hidden_size=output_size,
            output_size=output_size,
            activation_nonlin=activation_nonlin,
            tau_init=tau_init,
            tau_learnable=tau_learnable,
            kappa_init=kappa_init,
            kappa_learnable=kappa_learnable,
            potentials_init=potentials_init,
            input_embedding=input_embedding,
            output_embedding=output_embedding,
        )

        # Create the layer that converts the activations of the previous time step into potentials (internal stimulus).
        # For this model, self._hidden_size equals output_size.
        self.prev_activations_embedding = nn.Linear(self._hidden_size, self._hidden_size, bias=False)
        init_param_(self.prev_activations_embedding, **init_param_kwargs)

        # Create the layer that converts potentials into activations which are the outputs in this model.
        # Scaling weights equals beta in eq (4) in [Luksch et al., 2012].
        self.potentials_to_activations = IndependentNonlinearitiesLayer(
            self._hidden_size, nonlin=activation_nonlin, bias=False, weight=True
        )

        # Potential dynamics.
        self.potentials_dyn_fcn = potentials_dyn_fcn
        self.capacity_learnable = capacity_learnable
        if self.potentials_dyn_fcn in [pd_capacity_21, pd_capacity_21_abs, pd_capacity_32, pd_capacity_32_abs]:
            if _is_iterable(activation_nonlin):
                self._init_capacity(activation_nonlin[0])
            else:
                self._init_capacity(activation_nonlin)
        else:
            self._log_capacity = None

        # Initialize cubic decay and capacity if learnable.
        if self.potentials_dyn_fcn == pd_cubic and self.kappa_learnable:
            self._log_kappa.data = self._log_kappa_init
        elif self.potentials_dyn_fcn in [pd_capacity_21, pd_capacity_21_abs, pd_capacity_32, pd_capacity_32_abs]:
            if self.capacity_learnable:
                self._log_capacity.data = self._log_capacity_init

        # Move the complete model to the given device.
        self.to(device=device)

    def _init_capacity(self, activation_nonlin: ActivationFunction) -> None:
        """Initialize the value of the capacity parameter $C$ depending on the activation function.

        Args:
            activation_nonlin: Nonlinear activation function used.
        """
        if activation_nonlin is torch.sigmoid:
            # sigmoid(7.) approx 0.999
            self._log_capacity_init = torch.log(torch.tensor([7.0], dtype=torch.get_default_dtype()))
            self._log_capacity = (
                nn.Parameter(self._log_capacity_init, requires_grad=True)
                if self.capacity_learnable
                else self._log_capacity_init
            )
        elif activation_nonlin is torch.tanh:
            # tanh(3.8) approx 0.999
            self._log_capacity_init = torch.log(torch.tensor([3.8], dtype=torch.get_default_dtype()))
            self._log_capacity = (
                nn.Parameter(self._log_capacity_init, requires_grad=True)
                if self.capacity_learnable
                else self._log_capacity_init
            )
        else:
            raise ValueError(
                "For the potential dynamics including a capacity, only output nonlinearities of type "
                "torch.sigmoid and torch.tanh are supported!"
            )

    def extra_repr(self) -> str:
        return super().extra_repr() + f", capacity_learnable={self.capacity_learnable}"

    @property
    def capacity(self) -> Optional[torch.Tensor]:
        """Get the capacity parameter (exists for capacity-based dynamics functions), otherwise return `None`."""
        return None if self._log_capacity is None else torch.exp(self._log_capacity)

    def potentials_dot(self, potentials: torch.Tensor, stimuli: torch.Tensor) -> torch.Tensor:
        r"""Compute the derivative of the neurons' potentials per time step.

        $/tau /dot{u} = f(u, s, h)$
        with the potentials $u$, the combined stimuli $s$, and the resting level $h$.

        Args:
            potentials: Potential values at the current point in time, of shape `(hidden_size,)`.
            stimuli: Sum of external and internal stimuli at the current point in time, of shape `(hidden_size,)`.

        Returns:
            Time derivative of the potentials $\frac{dp}{dt}$, of shape `(hidden_size,)`.
        """
        return self.potentials_dyn_fcn(
            potentials, stimuli, self.resting_level, self.tau, kappa=self.kappa, capacity=self.capacity
        )

    def forward_one_step(
        self, inputs: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> (torch.Tensor, torch.Tensor):
        # Get the batch size, and prepare the inputs accordingly.
        batch_size = PotentialBased._infer_batch_size(inputs)
        inputs = inputs.view(batch_size, self.input_size).to(device=self.device)

        # If given use the hidden tensor, i.e., the potentials of the last step, else initialize them.
        potentials = self.init_hidden(batch_size, hidden)

        # Don't track the gradient through the hidden state but though the initial potentials.
        if hidden is not None:
            potentials = potentials.detach()

        # Scale the previous potentials, and pass them through a nonlinearity. Could also subtract a bias.
        activations_prev = self.potentials_to_activations(potentials)

        # ----------------
        # Activation Logic
        # ----------------

        # Combine the current input and the hidden variables from the last step.
        self._stimuli_external = self.input_embedding(inputs)
        self._stimuli_internal = self.prev_activations_embedding(activations_prev)

        # Potential dynamics forward integration (dt = 1).
        potentials = potentials + self.potentials_dot(potentials, self._stimuli_external + self._stimuli_internal)

        # Clip the potentials.
        potentials = potentials.clamp(min=-self._potentials_max, max=self._potentials_max)

        # Compute the activations: scale the potentials, subtract a bias, and pass them through a nonlinearity.
        activations = self.potentials_to_activations(potentials)

        # Compute the outputs from the activations. If there is no output embedding, they are the same thing.
        outputs = self.output_embedding(activations)

        return outputs, potentials  # the current potentials are the hidden state
