"""Compare scripted and compiled neural fields with eager execution."""

import pytest
import torch

from neuralfields import NeuralField, PotentialBased, SimpleNeuralField, pd_linear


@pytest.mark.parametrize(
    "model_class, kwargs",
    [
        (NeuralField, {"input_size": 3, "hidden_size": 4}),
        (SimpleNeuralField, {"input_size": 3, "output_size": 4, "potentials_dyn_fcn": pd_linear}),
    ],
    ids=["neural_field", "simple_neural_field"],
)
@pytest.mark.filterwarnings("ignore:.*torch.jit.script.*deprecated:FutureWarning")
@pytest.mark.xfail(
    reason="Full models are not yet TorchScript-compatible (parameter types and Python callables).",
    raises=RuntimeError,
    strict=True,
)
def test_jit_script(model_class: type[PotentialBased], kwargs: dict):
    """Compare both outputs of a scripted model with its eager counterpart when supported."""
    model = model_class(**kwargs).eval()
    inputs = torch.randn(2, 3, model.input_size)
    hidden = torch.randn(2, model.hidden_size)

    with torch.no_grad():
        expected = model(inputs, hidden)
        scripted_model = torch.jit.script(model)
        actual = scripted_model(inputs, hidden)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "model_class, kwargs",
    [
        (NeuralField, {"input_size": 3, "hidden_size": 4}),
        (SimpleNeuralField, {"input_size": 3, "output_size": 4, "potentials_dyn_fcn": pd_linear}),
    ],
    ids=["neural_field", "simple_neural_field"],
)
def test_compile(model_class: type[PotentialBased], kwargs: dict):
    """Compare both outputs of a compiled model with its eager counterpart."""
    model = model_class(**kwargs).eval()
    inputs = torch.randn(2, 3, model.input_size)
    hidden = torch.randn(2, model.hidden_size)

    with torch.no_grad():
        expected = model(inputs, hidden)
        compiled_model = torch.compile(model)
        actual = compiled_model(inputs, hidden)

    torch.testing.assert_close(actual, expected)
