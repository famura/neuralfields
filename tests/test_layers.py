import pytest
import torch

from neuralfields import IndependentNonlinearitiesLayer, MirroredConv1d, init_param_


@pytest.mark.parametrize("in_features", [1, 3], ids=["1dim", "3dim"])
@pytest.mark.parametrize("same_nonlin", [True, False], ids=["same_nonlin", "different_nonlin"])
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
@pytest.mark.parametrize("weight", [True, False], ids=["weight", "no_weight"])
def test_independent_nonlin_layer(in_features: int, same_nonlin: bool, bias: bool, weight: bool):
    if not same_nonlin and in_features > 1:
        nonlin = in_features * [torch.tanh]
    else:
        nonlin = torch.sigmoid
    layer = IndependentNonlinearitiesLayer(in_features, nonlin, bias, weight)
    assert isinstance(layer, torch.nn.Module)

    i = torch.randn(in_features)
    o = layer(i)
    assert isinstance(o, torch.Tensor)
    assert i.shape == o.shape

    with pytest.raises(RuntimeError):
        IndependentNonlinearitiesLayer(in_features, (in_features + 1) * [torch.tanh], bias, weight)


def test_independent_nonlin_layer_repr():
    print(IndependentNonlinearitiesLayer(in_features=2, nonlin=torch.relu, bias=False))


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

    # Convert to Torch Script.
    torch.jit.script(conv_layer)
