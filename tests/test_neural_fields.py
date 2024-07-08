from typing import Optional, Union

import pytest
import torch
from torch import nn

from tests.conftest import m_needs_cuda

from neuralfields import ActivationFunction, NeuralField, PotentialBased


@pytest.mark.parametrize("input_size", [1, 6], ids=["1dim_input", "6dim_input"])
@pytest.mark.parametrize("hidden_size", [4, 7], ids=["4dim_hidden", "7dim_hidden"])
@pytest.mark.parametrize("output_size", [None, 3], ids=["default_output", "3dim_output"])
@pytest.mark.parametrize("input_embedding", [None], ids=["default_input_embedding"])
@pytest.mark.parametrize("output_embedding", [None], ids=["default_output_embedding"])
@pytest.mark.parametrize("activation_nonlin", [torch.sigmoid], ids=["sigmoid"])
@pytest.mark.parametrize("mirrored_conv_weights", [True, False], ids=["mirrored", "not_mirrored"])
@pytest.mark.parametrize("conv_kernel_size", [None, 2], ids=["default_kernel_size", "kernel_size2"])
@pytest.mark.parametrize("conv_padding_mode", ["circular", "reflect", "zeros"], ids=["circular", "reflect", "zeros"])
@pytest.mark.parametrize("conv_out_channels", [1, 5], ids=["1dim_conv_out", "5dim_conv_out"])
@pytest.mark.parametrize("tau_init", [10], ids=["default_tau_init"])
@pytest.mark.parametrize("tau_learnable", [True, False], ids=["learnable_tau", "fixed_tau"])
@pytest.mark.parametrize("kappa_init", [0, 1e-2], ids=["kappa_init_0", "1e-2"])
@pytest.mark.parametrize("kappa_learnable", [True, False], ids=["learnable_kappa", "fixed_kappa"])
@pytest.mark.parametrize("potentials_init", [None], ids=["default_pot_init"])
@pytest.mark.parametrize("init_param_kwargs", [None, dict(bell=True)], ids=["default_init", "bell_init"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=m_needs_cuda)], ids=["cpu", "cuda"])
def test_neural_fields(
    input_size: int,
    hidden_size: int,
    output_size: Optional[int],
    input_embedding: Optional[nn.Module],
    output_embedding: Optional[nn.Module],
    activation_nonlin: ActivationFunction,
    mirrored_conv_weights: bool,
    conv_kernel_size: Optional[int],
    conv_padding_mode: str,
    conv_out_channels: int,
    tau_init: Union[float, int],
    tau_learnable: bool,
    kappa_init: Union[float, int],
    kappa_learnable: bool,
    potentials_init: Optional[torch.Tensor],
    init_param_kwargs: Optional[dict],
    device: Optional[Union[str, torch.device]],
    dtype: Union[str, torch.device] = None,
    seed: int = 0,
    batch_size: int = 20,
    len_input_seq: int = 5,
):
    torch.manual_seed(seed)

    nf = NeuralField(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        input_embedding=input_embedding,
        output_embedding=output_embedding,
        activation_nonlin=activation_nonlin,
        mirrored_conv_weights=mirrored_conv_weights,
        conv_out_channels=conv_out_channels,
        conv_kernel_size=conv_kernel_size,
        conv_padding_mode=conv_padding_mode,
        tau_init=tau_init,
        tau_learnable=tau_learnable,
        kappa_init=kappa_init,
        kappa_learnable=kappa_learnable,
        potentials_init=potentials_init,
        init_param_kwargs=init_param_kwargs,
        device=device,
        dtype=dtype,
    )
    assert isinstance(nf, PotentialBased)

    # Get and set the parameters.
    param_vec = nf.param_values
    assert isinstance(param_vec, torch.Tensor)
    new_param_vec = param_vec + torch.randn_like(param_vec, device=device)
    nf.param_values = new_param_vec
    assert torch.allclose(nf.param_values, new_param_vec)

    # Compute dp/dt.
    for _ in range(10):
        p_dot = nf.potentials_dot(
            potentials=torch.randn(hidden_size, device=device), stimuli=torch.randn(hidden_size, device=device)
        )
        assert isinstance(p_dot, torch.Tensor)
        assert p_dot.shape == (hidden_size,)
        assert isinstance(nf.stimuli_internal, torch.Tensor)
        assert nf.stimuli_internal.shape == (hidden_size,)
        assert isinstance(nf.stimuli_external, torch.Tensor)
        assert nf.stimuli_external.shape == (hidden_size,)

    # Compute the unbatched forward pass.
    hidden = None
    for _ in range(5):
        outputs, hidden_next = nf.forward_one_step(inputs=torch.randn(input_size, device=device), hidden=hidden)
        hidden = hidden_next.clone()
        assert isinstance(outputs, torch.Tensor)
        assert outputs.shape == (1, output_size or nf.hidden_size)
        assert isinstance(hidden_next, torch.Tensor)
        assert hidden_next.shape == (1, nf.hidden_size)

    # Compute the batched forward pass.
    hidden = None
    for _ in range(5):
        outputs, hidden_next = nf.forward_one_step(
            inputs=torch.randn(batch_size, input_size, device=device), hidden=hidden
        )
        hidden = hidden_next.clone()
        assert isinstance(outputs, torch.Tensor)
        assert outputs.shape == (batch_size, output_size or nf.hidden_size)
        assert isinstance(hidden_next, torch.Tensor)
        assert hidden_next.shape == (batch_size, nf.hidden_size)

    # Evaluate a time series of inputs.
    for hidden in (None, torch.randn(batch_size, hidden_size, device=device)):
        output_seq, hidden_seq = nf.forward(
            inputs=torch.randn(batch_size, len_input_seq, input_size, device=device), hidden=hidden
        )
        assert isinstance(output_seq, torch.Tensor)
        assert output_seq.shape == (batch_size, len_input_seq, output_size or nf.hidden_size)
        assert isinstance(hidden_seq, torch.Tensor)
        assert hidden_seq.shape == (batch_size, len_input_seq, nf.hidden_size)

    # Convert to Torch Script.
    # scripted_nf = torch.jit.script(nf)


def test_neural_fields_repr():
    nf = NeuralField(input_size=3, hidden_size=4)
    print(nf)


@pytest.mark.parametrize("batch_size", [None, 20], ids=["unbatched", "batched"])
@pytest.mark.parametrize("potentials_init", [None, torch.randn(4)], ids=["default", "random"])
def test_neural_fields_init_hidden(batch_size: Optional[int], potentials_init: Optional[torch.Tensor]):
    nf = NeuralField(input_size=3, hidden_size=4)
    hidden = nf.init_hidden(batch_size, potentials_init)
    assert isinstance(hidden, torch.Tensor)


def test_neural_fields_init_potentials():
    nf = NeuralField(input_size=6, hidden_size=7, potentials_init=5 * torch.ones(7))
    assert torch.allclose(nf._potentials_init, 5 * torch.ones(7))


def test_neural_fields_fail():
    nf = NeuralField(input_size=6, hidden_size=7)

    with pytest.raises(ValueError):
        NeuralField(input_size=6, hidden_size=1)

    with pytest.raises(ValueError):
        NeuralField(input_size=6, hidden_size=7, conv_padding_mode="wrong")

    with pytest.raises(ValueError):
        NeuralField(input_size=6, hidden_size=7, activation_nonlin="wrong")

    with pytest.raises(RuntimeError):
        nf.forward_one_step(inputs=torch.randn(1, 2, 3, 4))

    with pytest.raises(RuntimeError):
        nf.forward_one_step(inputs=torch.randn(6), hidden=torch.randn(1, 2, 3, 4))

    with pytest.raises(RuntimeError):
        nf.input_embedding = torch.nn.Linear(6, 7 + 1)
        nf.forward_one_step(inputs=torch.randn(6))
