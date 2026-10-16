import pytest
import torch

from neuralfields import NeuralField, PotentialBased, SimpleNeuralField, pd_cubic


@pytest.fixture(params=[NeuralField, SimpleNeuralField])
def model(request: pytest.FixtureRequest) -> PotentialBased:
    torch.manual_seed(0)
    if request.param is NeuralField:
        return NeuralField(input_size=1, hidden_size=2, output_size=2, activation_nonlin=torch.tanh)
    return SimpleNeuralField(input_size=1, output_size=2, potentials_dyn_fcn=pd_cubic, activation_nonlin=torch.tanh)


def test_hidden_state_is_detached(model: PotentialBased):
    """Test that gradients do not propagate through prior hidden states."""
    inputs = torch.randn(1, 3, model.input_size, requires_grad=True)
    hidden = torch.zeros(1, model.hidden_size, requires_grad=True)
    outputs, _ = model(inputs, hidden)
    outputs[:, -1].sum().backward()

    assert inputs.grad is not None
    torch.testing.assert_close(inputs.grad[:, :-1], torch.zeros_like(inputs.grad[:, :-1]))
    assert inputs.grad[:, -1].abs().sum() > 0
    assert hidden.grad is None
    input_weight = next(model.input_embedding.parameters())
    assert input_weight.grad is not None
    assert input_weight.grad.abs().sum() > 0


def test_model_dtype_conversion(model: PotentialBased):
    """Test that converting the model updates parameter, buffer, and output dtypes."""
    model.double()
    assert model.init_hidden().dtype == torch.float64
    assert model.init_hidden(batch_size=2).dtype == torch.float64
    assert model.stimuli_external.dtype == torch.float64
    assert model.stimuli_internal.dtype == torch.float64
    outputs, _ = model(torch.zeros(2, 3, model.input_size, dtype=torch.float64))
    assert outputs.dtype == torch.float64


def test_custom_input_embedding(model: PotentialBased):
    """Test that a replacement input embedding preserves the expected output shape."""
    model.input_embedding = torch.nn.Sequential(torch.nn.Linear(model.input_size, model.hidden_size))
    outputs, _ = model(torch.zeros(1, 2, model.input_size))
    assert outputs.shape == (1, 2, model.output_size)


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")),
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
    ],
)
def test_accelerator_movement(model: PotentialBased, device: str):
    """Test that moving a model to an accelerator preserves forward and gradient behavior."""
    inputs = torch.randn(2, 3, model.input_size)
    hidden = torch.randn(2, model.hidden_size)
    with torch.no_grad():
        expected, _ = model(inputs, hidden)
    model.to(device)
    assert model.init_hidden().device.type == device
    assert model.init_hidden(batch_size=2).device.type == device
    assert model.stimuli_external.device.type == device
    assert model.stimuli_internal.device.type == device

    inputs = inputs.to(device).requires_grad_()
    hidden = hidden.to(device)
    outputs, _ = model(inputs, hidden)
    torch.testing.assert_close(outputs.cpu(), expected, rtol=1e-4, atol=1e-6)
    outputs[:, -1].sum().backward()
    assert inputs.grad is not None
    assert inputs.grad[:, -1].abs().sum() > 0


def test_neural_field_custom_output_and_dtype():
    """Test custom embeddings and float64 conversion in a neural field."""
    embedding = torch.nn.Sequential()
    model = NeuralField(input_size=2, hidden_size=2, input_embedding=embedding, output_embedding=embedding).to(
        dtype=torch.float64
    )
    assert model.input_embedding is embedding
    assert model.output_embedding is embedding
    assert all(parameter.dtype == torch.float64 for parameter in model.parameters())
    outputs, _ = model(torch.zeros(1, 2, 2, dtype=torch.float64))
    assert outputs.dtype == torch.float64


@pytest.mark.parametrize("tau_init", [1e-6, PotentialBased._tau_min])
def test_invalid_tau_initialization(tau_init: float):
    """Test that tau values at or below the minimum are rejected."""
    with pytest.raises(ValueError):
        NeuralField(input_size=1, hidden_size=2, tau_init=tau_init)
