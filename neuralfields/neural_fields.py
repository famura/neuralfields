import multiprocessing as mp
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import nn

from neuralfields.custom_layers import IndependentNonlinearitiesLayer, MirroredConv1d, init_param_
from neuralfields.custom_types import ActivationFunction
from neuralfields.potential_based import PotentialBased


class NeuralField(PotentialBased):
    """A potential-based recurrent neural network according to [Amari, 1977].

    See Also:
        [Amari, 1977] S.-I. Amari, "Dynamics of Pattern Formation in Lateral-Inhibition Type Neural Fields",
        Biological Cybernetics, 1977.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        input_embedding: Optional[nn.Module] = None,
        output_embedding: Optional[nn.Module] = None,
        activation_nonlin: Union[ActivationFunction, Sequence[ActivationFunction]] = torch.sigmoid,
        mirrored_conv_weights: bool = True,
        conv_kernel_size: Optional[int] = None,
        conv_padding_mode: str = "circular",
        conv_out_channels: int = 1,
        conv_pooling_norm: int = 1,
        tau_init: Union[float, int] = 10,
        tau_learnable: bool = True,
        kappa_init: Union[float, int] = 1e-5,
        kappa_learnable: bool = True,
        potentials_init: Optional[torch.Tensor] = None,
        init_param_kwargs: Optional[dict] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Args:
            input_size: Number of input dimensions.
            hidden_size: Number of neurons with potential in the (single) hidden layer.
            output_size: Number of output dimensions. By default, the number of outputs is equal to the number of
                hidden neurons.
            input_embedding: Optional (custom) [Module][torch.nn.Module] to extract features from the inputs.
                This module must transform the inputs such that the dimensionality matches the number of
                neurons of the neural field, i.e., `hidden_size`. By default, a [linear layer][torch.nn.Linear]
                without biases is used.
            output_embedding: Optional (custom) [Module][torch.nn.Module] to compute the outputs from the activations.
                This module must map the activations of shape (`hidden_size`,) to the outputs of shape (`output_size`,)
                By default, a [linear layer][torch.nn.Linear] without biases is used.
            activation_nonlin: Nonlinearity used to compute the activations from the potential levels.
            mirrored_conv_weights: If `True`, re-use weights for the second half of the kernel to create a
                symmetric convolution kernel.
            conv_kernel_size: Size of the kernel for the 1-dim convolution along the potential-based neurons.
            conv_padding_mode: Padding mode forwarded to [Conv1d][torch.nn.Conv1d], options are "circular",
                "reflect", or "zeros".
            conv_out_channels: Number of filter for the 1-dim convolution along the potential-based neurons.
            conv_pooling_norm: Norm type of the [torch.nn.LPPool1d][] pooling layer applied after the convolution.
                Unlike in typical scenarios, here the pooling is performed over the channel dimension. Thus, varying
                `conv_pooling_norm` only has an effect if `conv_out_channels > 1`.
            tau_init: Initial value for the shared time constant of the potentials.
            tau_learnable: Whether the time constant is a learnable parameter or fixed.
            kappa_init: Initial value for the cubic decay, pass 0 to disable the cubic decay.
            kappa_learnable: Whether the cubic decay is a learnable parameter or fixed.
            potentials_init: Initial for the potentials, i.e., the network's hidden state.
            init_param_kwargs: Additional keyword arguments for the policy parameter initialization.
            device: Device to move this module to (after initialization).
            dtype: Data type forwarded to the initializer of [Conv1d][torch.nn.Conv1d].
        """
        if hidden_size < 2:
            raise ValueError("The humber of hidden neurons hidden_size must be at least 2!")
        if conv_kernel_size is None:
            conv_kernel_size = hidden_size
        if conv_padding_mode not in ["circular", "reflect", "zeros"]:
            raise ValueError("The conv_padding_mode must be either 'circular', 'reflect', or 'zeros'!")
        if not callable(activation_nonlin):
            raise ValueError("The activation function activation_nonlin must be a callable!")
        init_param_kwargs = init_param_kwargs if init_param_kwargs is not None else dict()

        # Set the multiprocessing start method to spawn, since PyTorch is using the GPU for convolutions if it can.
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)

        # Create the common layers and parameters.
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            activation_nonlin=activation_nonlin,
            tau_init=tau_init,
            tau_learnable=tau_learnable,
            kappa_init=kappa_init,
            kappa_learnable=kappa_learnable,
            potentials_init=potentials_init,
            input_embedding=input_embedding,
            output_embedding=output_embedding,
            device=device,
        )

        # Create the custom convolution layer that models the interconnection of neurons, i.e., their potentials.
        self.mirrored_conv_weights = mirrored_conv_weights
        conv1d_class = MirroredConv1d if self.mirrored_conv_weights else nn.Conv1d
        self.conv_layer = conv1d_class(
            in_channels=1,  # treat potentials as a time series of values (convolutions is over the "time" axis)
            out_channels=conv_out_channels,
            kernel_size=conv_kernel_size,
            padding_mode=conv_padding_mode,
            padding="same",  # to preserve the length od the output sequence
            bias=False,
            stride=1,
            dilation=1,
            groups=1,
            device=device,
            dtype=dtype,
        )
        init_param_(self.conv_layer, **init_param_kwargs)

        # Create a pooling layer that reduced all output channels to one.
        self.conv_pooling_layer = torch.nn.LPPool1d(
            conv_pooling_norm, kernel_size=conv_out_channels, stride=conv_out_channels
        )

        # Create the layer that converts the activations of the previous time step into potentials.
        self.potentials_to_activations = IndependentNonlinearitiesLayer(
            self._hidden_size, activation_nonlin, bias=True, weight=True
        )

        # Create the custom output embedding layer that combines the activations.
        self.output_embedding = nn.Linear(self._hidden_size, self.output_size, bias=False)

        # Move the complete model to the given device.
        self.to(device=device)

    def potentials_dot(self, potentials: torch.Tensor, stimuli: torch.Tensor) -> torch.Tensor:
        r"""Compute the derivative of the neurons' potentials w.r.t. time.

         $/tau /dot{u} = s + h - u + /kappa (h - u)^3,
        /quad /text{with} s = s_{int} + s_{ext} = W*o + /int{w(u, v) f(u) dv}$
        with the potentials $u$, the combined stimuli $s$, the resting level $h$, and the cubic decay $\kappa$.

        Args:
            potentials: Potential values at the current point in time, of shape `(hidden_size,)`.
            stimuli: Sum of external and internal stimuli at the current point in time, of shape `(hidden_size,)`.

        Returns:
            Time derivative of the potentials $\frac{dp}{dt}$, of shape `(hidden_size,)`.
        """
        rhs = stimuli + self.resting_level - potentials + self.kappa * torch.pow(self.resting_level - potentials, 3)
        return rhs / self.tau

    # pylint: disable=duplicate-code
    def forward_one_step(
        self, inputs: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get the batch size, and prepare the inputs accordingly.
        batch_size = PotentialBased._infer_batch_size(inputs)
        inputs = inputs.view(batch_size, self.input_size).to(device=self.device)

        # If given use the hidden tensor, i.e., the potentials of the last step, else initialize them.
        potentials = self.init_hidden(batch_size, hidden)

        # Don't track the gradient through the hidden state but though the initial potentials.
        if hidden is not None:
            potentials = potentials.detach()

        # Compute the activations: scale the potentials, subtract a bias, and pass them through a nonlinearity.
        activations_prev = self.potentials_to_activations(potentials)

        # Combine the current inputs to the external simuli.
        self._stimuli_external = self.input_embedding(inputs)

        # Reshape and convolve the previous activations to the internal stimuli. There is only 1 input channel.
        self._stimuli_internal = self.conv_layer(activations_prev.view(batch_size, 1, self._hidden_size))

        if self._stimuli_internal.size(1) > 1:
            # In PyTorch, 1-dim pooling is done over the last, i.e., here the potentials' dimension. Instead, we want to
            # pool over the channel dimension, and then squeeze it since this has been reduced to 1 by the pooling.
            self._stimuli_internal = self.conv_pooling_layer(self._stimuli_internal.permute(0, 2, 1))
            self._stimuli_internal = self._stimuli_internal.squeeze(2)
        else:
            # No pooling necessary since there was only one output channel for the convolution.
            self._stimuli_internal = self._stimuli_internal.squeeze(1)

        # Eagerly check the shapes before adding the resting level since the broadcasting could mask errors from
        # the later convolution operation.
        if self._stimuli_external.shape != self._stimuli_internal.shape:
            raise RuntimeError(
                f"The shape of the internal and external stimuli do not match! They are {self._stimuli_internal.shape} "
                f"and {self._stimuli_external.shape}."
            )

        # Potential dynamics forward integration (dt = 1).
        potentials = potentials + self.potentials_dot(potentials, self._stimuli_external + self._stimuli_internal)

        # Clip the potentials for numerical stabilization.
        potentials = potentials.clamp(min=-self._potentials_max, max=self._potentials_max)

        # Compute the activations: scale the potentials, subtract a bias, and pass them through a nonlinearity.
        activations = self.potentials_to_activations(potentials)

        # Compute the outputs from the activations. If there is no output embedding, they are the same thing.
        outputs = self.output_embedding(activations)

        return outputs, potentials  # the current potentials are the hidden state
