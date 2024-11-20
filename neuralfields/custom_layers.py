import copy
import math
from typing import Any, Optional, Sequence, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _single

from neuralfields.custom_types import ActivationFunction


def _is_iterable(obj: Any) -> bool:
    """Check if the input is iterable by trying to create an iterator from the input.

    Args:
        obj: Any object.

    Returns:
        `True` if input is iterable, else `False`.
    """
    try:
        _ = iter(obj)
        return True
    except TypeError:
        return False


@torch.no_grad()
def apply_bell_shaped_weights_conv_(m: nn.Module, w: torch.Tensor, ks: int) -> None:
    """Helper function to set the weights of a convolution layer according to a squared exponential.

    Args:
        m: Module containing the weights to be set.
        w: Linearly spaced weights.
        ks: Size of the convolution kernel.
    """
    dim_ch_out, dim_ch_in = m.weight.data.size(0), m.weight.data.size(1)  # type: ignore[operator]
    amp = torch.rand(dim_ch_out * dim_ch_in)
    for i in range(dim_ch_out):
        for j in range(dim_ch_in):
            m.weight.data[i, j, :] = amp[i * dim_ch_in + j] * 2 * (torch.exp(-torch.pow(w, 2) / (ks / 2) ** 2) - 0.5)


# pylint: disable=too-many-branches
@torch.no_grad()
def init_param_(m: torch.nn.Module, **kwargs: Any) -> None:
    """Initialize the parameters of the PyTorch Module / layer / network / cell according to its type.

    Args:
        m: Module containing the weights to be set.
        kwargs: Optional keyword arguments, e.g. `bell=True` to initialize a convolution layer's weight with a
            centered "bell-shaped" parameter value distribution.
    """
    kwargs = kwargs if kwargs is not None else dict()

    if isinstance(m, nn.Conv1d):
        if kwargs.get("bell", False):
            # Initialize the kernel weights with a shifted of shape exp(-x^2 / sigma^2).
            # The biases are left unchanged.
            if m.weight.data.size(2) % 2 == 0:
                ks_half = m.weight.data.size(2) // 2
                ls_half = torch.linspace(ks_half, 0, ks_half)  # descending
                ls = torch.cat([ls_half, torch.flip(ls_half, (0,))])
            else:
                ks_half = math.ceil(m.weight.data.size(2) / 2)
                ls_half = torch.linspace(ks_half, 0, ks_half)  # descending
                ls = torch.cat([ls_half, torch.flip(ls_half[:-1], (0,))])
            apply_bell_shaped_weights_conv_(m, ls, ks_half)
        else:
            m.reset_parameters()

    elif isinstance(m, MirroredConv1d):
        if kwargs.get("bell", False):
            # Initialize the kernel weights with a shifted of shape exp(-x^2 / sigma^2).
            # The biases are left unchanged (does not exist by default).
            ks = m.weight.data.size(2)  # ks_mirr = ceil(ks_conv1d / 2)
            ls = torch.linspace(ks, 0, ks)  # descending
            apply_bell_shaped_weights_conv_(m, ls, ks)
        else:
            m.reset_parameters()

    elif isinstance(m, IndependentNonlinearitiesLayer):
        # Initialize the network's parameters according to a normal distribution.
        for tensor in (m.weight, m.bias):
            if tensor is not None:
                nn.init.normal_(tensor, std=1.0 / math.sqrt(tensor.nelement()))

    elif isinstance(m, nn.Linear):
        if kwargs.get("self_centric_init", False):
            m.weight.data.fill_(-0.5)  # inhibit others
            for i in range(m.weight.data.size(0)):
                m.weight.data[i, i] = 1.0  # excite self


class IndependentNonlinearitiesLayer(nn.Module):
    """Neural network layer to add a bias, multiply the result with a scaling factor, and then apply the given
    nonlinearity. If a list of nonlinearities is provided, every dimension will be processed separately.
    The scaling and the bias are learnable parameters.
    """

    weight: Union[nn.Parameter, torch.Tensor]
    bias: Union[nn.Parameter, torch.Tensor]

    def __init__(
        self,
        in_features: int,
        nonlin: Union[ActivationFunction, Sequence[ActivationFunction]],
        bias: bool,
        weight: bool = True,
    ):
        """
        Args:
            in_features: Number of dimensions of each input sample.
            nonlin: The nonlinear function to apply.
            bias: If `True`, a learnable bias is subtracted, else no bias is used.
            weight: If `True`, the input is multiplied with a learnable scaling factor.
        """
        if not callable(nonlin):
            if len(nonlin) != in_features:
                raise RuntimeError(
                    f"Either one, or {in_features} nonlinear functions have been expected, but "
                    f"{len(nonlin)} have been given!"
                )

        super().__init__()

        # Create and initialize the parameters, and the activation function.
        self.nonlin = copy.deepcopy(nonlin) if _is_iterable(nonlin) else nonlin
        if weight:
            self.weight = nn.Parameter(torch.empty(in_features, dtype=torch.get_default_dtype()))
        else:
            self.weight = torch.ones(in_features, dtype=torch.get_default_dtype())
        if bias:
            self.bias = nn.Parameter(torch.empty(in_features, dtype=torch.get_default_dtype()))
        else:
            self.bias = torch.zeros(in_features, dtype=torch.get_default_dtype())

        init_param_(self)

    def extra_repr(self) -> str:
        return f"in_features={self.weight.numel()}, weight={self.weight}, " f"bias={self.bias}"

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Apply a bias, scaling, and a nonliterary to each input separately.

        $y = f_{nlin}( w * (x + b) )$

        Args:
            inp: Arbitrary input tensor.

        Returns:
            Output tensor.
        """
        tmp = inp + self.bias
        tmp = self.weight * tmp

        # Every dimension runs through an individual nonlinearity.
        if _is_iterable(self.nonlin):
            return torch.tensor([fcn(tmp[idx]) for idx, fcn in enumerate(self.nonlin)])

        # All dimensions identically.
        return self.nonlin(tmp)  # type: ignore[operator]


class MirroredConv1d(_ConvNd):
    """A variant of the [Conv1d][torch.nn.Conv1d] module that re-uses parts of the convolution weights by mirroring
    the first half of the kernel (along the columns). This way we can save almost half of the parameters, under
    the assumption that we have a kernel that obeys this kind of symmetry. The biases are left unchanged.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[int, str] = "same",  # kernel_size // 2 if padding_mode != "circular" else kernel_size - 1
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = "zeros",
        device: Optional[Union[str, torch.device]] = None,
        dtype=None,
    ):
        # Same as in PyTorch 1.12.
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=_single(kernel_size),  # type: ignore[arg-type]
            stride=_single(stride),  # type: ignore[arg-type]
            padding=_single(padding) if not isinstance(padding, str) else padding,  # type: ignore[arg-type]
            dilation=_single(dilation),  # type: ignore[arg-type]
            transposed=False,
            output_padding=_single(0),
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        # Memorize PyTorch's weight shape (out_channels x in_channels x kernel_size) for later reconstruction.
        self.orig_weight_shape = self.weight.shape

        # Get number of kernel elements we later want to use for mirroring.
        self.half_kernel_size = math.ceil(self.weight.size(2) / 2)  # kernel_size = 4 --> 2, kernel_size = 5 --> 3

        # Initialize the weights values the same way PyTorch does.
        new_weight_init = torch.zeros(
            self.orig_weight_shape[0], self.orig_weight_shape[1], self.half_kernel_size, device=device
        )
        nn.init.kaiming_uniform_(new_weight_init, a=math.sqrt(5))

        # Overwrite the weight attribute (transposed is False by default for the Conv1d module, we don't use it here).
        self.weight = nn.Parameter(new_weight_init)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Computes the 1-dim convolution just like [Conv1d][torch.nn.Conv1d], however, the kernel has mirrored weights,
        i.e., it is symmetric around its middle element, or in case of an even kernel size around an imaginary middle
        element.

        Args:
            inp: 3-dim input tensor just like for [Conv1d][torch.nn.Conv1d].

        Returns:
            3-dim output tensor just like for [Conv1d][torch.nn.Conv1d].
        """
        # Reconstruct symmetric weights for convolution (original size).
        mirr_weight = torch.empty(self.orig_weight_shape, dtype=inp.dtype, device=self.weight.device)

        # Loop over input channels.
        for i in range(self.orig_weight_shape[1]):
            # Fill first half.
            mirr_weight[:, i, : self.half_kernel_size] = self.weight[:, i, :]

            # Fill second half (flip columns left-right).
            if self.orig_weight_shape[2] % 2 == 1:
                # Odd kernel size for convolution, don't flip the last column.
                mirr_weight[:, i, self.half_kernel_size :] = torch.flip(self.weight[:, i, :], (1,))[:, 1:]
            else:
                # Even kernel size for convolution, flip all columns.
                mirr_weight[:, i, self.half_kernel_size :] = torch.flip(self.weight[:, i, :], (1,))

        # Run through the same function as the original PyTorch implementation, but with mirrored kernel.
        return F.conv1d(
            input=inp,
            weight=mirr_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
