from typing import Optional, Union

import pytest
import torch
from torch import nn

from tests.conftest import m_needs_cuda

from neuralfields import (
    ActivationFunction,
    PotentialBased,
    PotentialsDynamicsType,
    SimpleNeuralField,
    pd_capacity_21,
    pd_capacity_21_abs,
    pd_capacity_32,
    pd_capacity_32_abs,
    pd_cubic,
    pd_linear,
)


@pytest.mark.parametrize("input_size", [1, 6], ids=["1dim_input", "6dim_input"])
@pytest.mark.parametrize("output_size", [1, 3], ids=["1dim_output", "3dim_output"])
@pytest.mark.parametrize(
    "potentials_dyn_fcn",
    [pd_linear, pd_cubic, pd_capacity_21, pd_capacity_21_abs, pd_capacity_32, pd_capacity_32_abs],
    ids=["pd_linear", "pd_cubic", "pd_capacity_21", "pd_capacity_21_abs", "pd_capacity_32", "pd_capacity_32_abs"],
)
@pytest.mark.parametrize("input_embedding", [None], ids=["default_input_embedding"])
@pytest.mark.parametrize("output_embedding", [None], ids=["default_output_embedding"])
@pytest.mark.parametrize("activation_nonlin", [torch.sigmoid, torch.tanh], ids=["sigmoid", "tanh"])
@pytest.mark.parametrize("tau_init", [10], ids=["default_tau_init"])
@pytest.mark.parametrize("tau_learnable", [True, False], ids=["learnable_tau", "fixed_tau"])
@pytest.mark.parametrize("kappa_init", [1e-2], ids=["kappa_1e-2"])
@pytest.mark.parametrize("kappa_learnable", [True, False], ids=["learnable_kappa", "fixed_kappa"])
@pytest.mark.parametrize("capacity_learnable", [True, False], ids=["learnable_capacity", "fixed_capacity"])
@pytest.mark.parametrize("potentials_init", [None], ids=["default_pot_init"])
@pytest.mark.parametrize(
    "init_param_kwargs", [None, dict(self_centric_init=True)], ids=["default_init", "self_centric_init"]
)
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=m_needs_cuda)], ids=["cpu", "cuda"])
def test_simple_neural_fields(
    input_size: int,
    output_size: Optional[int],
    potentials_dyn_fcn: PotentialsDynamicsType,
    input_embedding: Optional[nn.Module],
    output_embedding: Optional[nn.Module],
    activation_nonlin: ActivationFunction,
    tau_init: Union[float, int],
    tau_learnable: bool,
    kappa_init: Union[float, int],
    kappa_learnable: bool,
    capacity_learnable: bool,
    potentials_init: Optional[torch.Tensor],
    init_param_kwargs: Optional[dict],
    device: Optional[Union[str, torch.device]],
    seed: int = 0,
    batch_size: int = 20,
    len_input_seq: int = 5,
):
    torch.manual_seed(seed)

    snf = SimpleNeuralField(
        input_size=input_size,
        output_size=output_size,
        potentials_dyn_fcn=potentials_dyn_fcn,
        input_embedding=input_embedding,
        output_embedding=output_embedding,
        activation_nonlin=activation_nonlin,
        tau_init=tau_init,
        tau_learnable=tau_learnable,
        kappa_init=kappa_init,
        kappa_learnable=kappa_learnable,
        capacity_learnable=capacity_learnable,
        potentials_init=potentials_init,
        init_param_kwargs=init_param_kwargs,
        device=device,
    )
    assert isinstance(snf, PotentialBased)
    hidden_size = output_size

    # Get and set the parameters.
    param_vec = snf.param_values
    assert isinstance(param_vec, torch.Tensor)
    new_param_vec = param_vec + torch.randn_like(param_vec, device=device)
    snf.param_values = new_param_vec
    assert torch.allclose(snf.param_values, new_param_vec)

    # Compute dp/dt.
    for _ in range(10):
        p_dot = snf.potentials_dot(
            potentials=torch.randn(hidden_size, device=device), stimuli=torch.randn(hidden_size, device=device)
        )
        assert isinstance(p_dot, torch.Tensor)
        assert p_dot.shape == (hidden_size,)
        assert isinstance(snf.stimuli_internal, torch.Tensor)
        assert snf.stimuli_internal.shape == (hidden_size,)
        assert isinstance(snf.stimuli_external, torch.Tensor)
        assert snf.stimuli_external.shape == (hidden_size,)

    # Compute the unbatched forward pass.
    hidden = None
    for _ in range(5):
        outputs, hidden_next = snf.forward_one_step(inputs=torch.randn(input_size, device=device), hidden=hidden)
        hidden = hidden_next.clone()
        assert isinstance(outputs, torch.Tensor)
        assert outputs.shape == (1, output_size or snf.hidden_size)
        assert isinstance(hidden_next, torch.Tensor)
        assert hidden_next.shape == (1, snf.hidden_size)

    # Compute the batched forward pass.
    hidden = None
    for _ in range(5):
        outputs, hidden_next = snf.forward_one_step(
            inputs=torch.randn(batch_size, input_size, device=device), hidden=hidden
        )
        hidden = hidden_next.clone()
        assert isinstance(outputs, torch.Tensor)
        assert outputs.shape == (batch_size, output_size or snf.hidden_size)
        assert isinstance(hidden_next, torch.Tensor)
        assert hidden_next.shape == (batch_size, snf.hidden_size)

    # Evaluate a time series of inputs.
    for hidden in (None, torch.randn(batch_size, hidden_size, device=device)):
        output_seq, hidden_seq = snf.forward(
            inputs=torch.randn(batch_size, len_input_seq, input_size, device=device), hidden=hidden
        )
        assert isinstance(output_seq, torch.Tensor)
        assert output_seq.shape == (batch_size, len_input_seq, output_size or snf.hidden_size)
        assert isinstance(hidden_seq, torch.Tensor)
        assert hidden_seq.shape == (batch_size, len_input_seq, snf.hidden_size)

    # Convert to Torch Script.
    # scripted_nf = torch.jit.script(nf)


def test_simple_neural_fields_repr():
    snf = SimpleNeuralField(input_size=3, output_size=2, potentials_dyn_fcn=pd_linear)
    print(snf)


@pytest.mark.parametrize("batch_size", [None, 20], ids=["unbatched", "batched"])
@pytest.mark.parametrize("potentials_init", [None, torch.randn(4)], ids=["default", "random"])
def test_simple_neural_fields_init_hidden(batch_size: Optional[int], potentials_init: Optional[torch.Tensor]):
    snf = SimpleNeuralField(input_size=3, output_size=4, potentials_dyn_fcn=pd_capacity_21)
    hidden = snf.init_hidden(batch_size, potentials_init)
    assert isinstance(hidden, torch.Tensor)


def test_simple_neural_fields_init_potentials():
    snf = SimpleNeuralField(
        input_size=6, output_size=7, potentials_dyn_fcn=pd_capacity_21, potentials_init=5 * torch.ones(7)
    )
    assert torch.allclose(snf._potentials_init, 5 * torch.ones(7))


def test_simple_neural_fields_fail():
    with pytest.raises(ValueError):
        SimpleNeuralField(input_size=6, output_size=3, potentials_dyn_fcn=pd_capacity_21, activation_nonlin=torch.sqrt)

    with pytest.raises(ValueError):
        SimpleNeuralField(input_size=6, output_size=3, potentials_dyn_fcn=pd_capacity_21, tau_init=0)

    with pytest.raises(ValueError):
        SimpleNeuralField(input_size=6, output_size=3, potentials_dyn_fcn=pd_capacity_21, kappa_init=0)

    with pytest.raises(ValueError):
        pd_linear(
            p=torch.randn(3), s=torch.randn(3), h=torch.randn(3), tau=torch.tensor(-1.0), kappa=None, capacity=None
        )

    with pytest.raises(ValueError):
        pd_cubic(
            p=torch.randn(3),
            s=torch.randn(3),
            h=torch.randn(3),
            tau=torch.tensor(1.0),
            kappa=torch.tensor(-1.0),
            capacity=None,
        )


@pytest.mark.parametrize(
    "potentials_dyn_fcn",
    [pd_linear, pd_capacity_21, pd_capacity_32_abs],
    ids=["pd_linear", "pd_capacity_21", "pd_capacity_32_abs"],
)
def test_simple_neural_fields_multiple_activation_fcns(potentials_dyn_fcn: ActivationFunction):
    snf = SimpleNeuralField(
        input_size=6,
        output_size=2,
        potentials_dyn_fcn=potentials_dyn_fcn,
        activation_nonlin=[torch.sigmoid, torch.tanh],
    )
    assert snf.potentials_to_activations.nonlin == [torch.sigmoid, torch.tanh]
