import pytest
import torch

from neuralfields import IndependentNonlinearitiesLayer, MirroredConv1d, init_param_


@pytest.mark.parametrize("in_features", [1, 3], ids=["1dim", "3dim"])
@pytest.mark.parametrize("same_nonlin", [True, False], ids=["same_nonlin", "different_nonlin"])
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
@pytest.mark.parametrize("weight", [True, False], ids=["weight", "no_weight"])
def test_independent_nonlin_layer(in_features: int, same_nonlin: bool, bias: bool, weight: bool):
    """Test independent nonlinearities, optional parameters, and invalid configurations."""
    if not same_nonlin and in_features > 1:
        nonlin = in_features * [torch.tanh]
    else:
        nonlin = torch.sigmoid
    layer = IndependentNonlinearitiesLayer(in_features, nonlin, bias, weight)
    assert isinstance(layer, torch.nn.Module)
    if not weight:
        torch.testing.assert_close(layer.weight, torch.ones(in_features))
    if not bias:
        torch.testing.assert_close(layer.bias, torch.zeros(in_features))

    i = torch.randn(in_features)
    o = layer(i)
    assert isinstance(o, torch.Tensor)
    assert i.shape == o.shape

    with pytest.raises(RuntimeError):
        IndependentNonlinearitiesLayer(in_features, (in_features + 1) * [torch.tanh], bias, weight)


def test_independent_nonlin_layer_repr():
    """Test that an independent nonlinearities layer can be represented."""
    print(IndependentNonlinearitiesLayer(in_features=2, nonlin=torch.relu, bias=False))


def test_independent_nonlinearities_batched_gradients():
    """Test batched nonlinear transformations and gradient propagation."""
    layer = IndependentNonlinearitiesLayer(2, [torch.sigmoid, torch.tanh], bias=True).double()
    inputs = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    transformed = layer.weight * (inputs + layer.bias)
    expected = torch.stack((transformed[..., 0].sigmoid(), transformed[..., 1].tanh()), dim=-1)

    output = layer(inputs)
    torch.testing.assert_close(output, expected)
    output.sum().backward()

    assert inputs.grad is not None
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None


def test_independent_nonlinearities_composed_callable():
    """Test that a composed callable is applied elementwise with gradients preserved."""
    nonlin = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Sigmoid())
    layer = IndependentNonlinearitiesLayer(3, nonlin, bias=False, weight=False)
    inputs = torch.tensor([[-2.0, 1.0, -3.0], [3.0, -4.0, 2.0]], requires_grad=True)

    outputs = layer(inputs)
    torch.testing.assert_close(outputs, nonlin(inputs))

    outputs.sum().backward()
    assert inputs.grad is not None
    assert inputs.grad[0, 1] > 0


@pytest.mark.parametrize("container", [list, torch.nn.ModuleList], ids=["list", "module_list"])
def test_independent_nonlinearities_registered_modules(container: type):
    """Test parameter registration, optimization, and state restoration for module containers."""
    # Create the custom layer.
    nonlin = container([torch.nn.PReLU(), torch.nn.Dropout()])
    layer = IndependentNonlinearitiesLayer(2, nonlin, bias=False, weight=False).eval()
    parameters = list(layer.parameters())
    assert len(parameters) == 1

    # Do one random forward pass and update step.
    inputs = torch.tensor([[-2.0, 3.0], [-4.0, 5.0]])
    outputs = layer(inputs)
    torch.testing.assert_close(outputs, torch.tensor([[-0.5, 3.0], [-1.0, 5.0]]))
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    outputs.sum().backward()
    optimizer.step()
    torch.testing.assert_close(parameters[0], torch.tensor([0.85]))

    # Restore the layer from its state dict and check that it produces the same outputs.
    restored = IndependentNonlinearitiesLayer(
        2, container([torch.nn.PReLU(), torch.nn.Dropout()]), bias=False, weight=False
    ).eval()
    restored.load_state_dict(layer.state_dict())
    torch.testing.assert_close(restored(inputs), layer(inputs))


@pytest.mark.parametrize("in_channels", [1, 3], ids=["in1", "in3"])
@pytest.mark.parametrize("out_channels", [1, 4], ids=["out1", "out4"])
@pytest.mark.parametrize("kernel_size", [6, 7], ids=["ks6", "ks7"])
@pytest.mark.parametrize("padding_mode", ["circular", "reflect", "zeros"], ids=["circular", "reflect", "zeros"])
@pytest.mark.parametrize("use_custom_bell_init", [True, False], ids=["bell_init", "default_init"])
def test_mirrored_conv1d_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    padding_mode: str,
    use_custom_bell_init: bool,
    in_length: int = 50,
    batch_size: int = 20,
):
    """Test mirrored convolution output shapes for batched and unbatched inputs."""
    conv_layer = MirroredConv1d(
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding="same",
        dilation=1,
        groups=1,
        bias=False,
        padding_mode=padding_mode,
    )
    init_param_(conv_layer, bell=use_custom_bell_init)

    for _ in range(10):
        # Unbatched forward pass.
        output = conv_layer(torch.randn(in_channels, in_length))
        assert isinstance(output, torch.Tensor)
        assert output.dim() == 2
        assert output.shape == (out_channels, in_length)

        # Batched forward pass.
        output = conv_layer(torch.randn(batch_size, in_channels, in_length))
        assert isinstance(output, torch.Tensor)
        assert output.dim() == 3
        assert output.shape == (batch_size, out_channels, in_length)
