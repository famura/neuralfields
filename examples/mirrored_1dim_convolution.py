"""
Play around with PyTorch's 1-dim convolution class in the context of using it for the neural fields.

See Also:
    # https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728
    # https://github.com/jayleicn/TVQAplus/blob/master/model/cnn.py
"""
import torch
from matplotlib import pyplot as plt
from torch import nn

from neuralfields import MirroredConv1d, init_param_


if __name__ == "__main__":
    # Configure this script.
    hand_coded_filter = False  # if True, use a ramp from 0 to 1 instead of random weights
    use_depth_wise_conv = False  # just fooling around with that, just False in most cases
    use_custom_mirr_layer = True
    use_custom_bell_init = True
    torch.manual_seed(0)

    # More configuration.
    batch_size = 1
    num_neurons = 360  # each potential-based neuron is basically like time steps of a signal
    in_channels = 2  # number of input signals
    out_channels = 4  # number of filters, typically = 1
    kernel_size = 17  # larger number smooth out and reduce the length of the output signal, use odd numbers
    padding_mode = "circular"  # circular, reflect, zeros

    # Create arbitrary signal.
    signal = torch.zeros(batch_size, in_channels, num_neurons)
    signal[:, 0, :] = torch.cat(
        [torch.zeros(num_neurons // 3), torch.ones(num_neurons // 3), torch.zeros(num_neurons // 3)]
    )
    if hand_coded_filter:
        for i in range(1, in_channels):
            signal[:, i, :] = torch.rand_like(signal[:, 0, :]) / 3 + signal[:, 0, :] / (i + 1)

    if use_depth_wise_conv:
        conv_layer = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=1,
            padding="same",
            dilation=1,
            groups=1,
            bias=False,
            padding_mode=padding_mode,
        )
        ptwise_conv_layer = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            padding_mode="zeros",
        )
        print(f"conv_layer weights shape: {conv_layer.weight.shape}")
        print(f"ptwise_conv_layer weights shape: {ptwise_conv_layer.weight.shape}")

    else:
        # Standard convolution.
        conv_layer = nn.Conv1d(
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
        print(f"conv_layer weights shape: {conv_layer.weight.shape}")

        # Create a ramp filter.
        if hand_coded_filter:
            conv_layer.weight.data = (
                torch.linspace(0, 1, kernel_size).reshape(1, 1, -1).repeat(out_channels, in_channels, 1)
            )

        # Use mirrored weights.
        elif use_custom_mirr_layer:
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
            print(f"mirr conv_layer weights shape: {conv_layer.weight.shape}")

    print(f"input shape:  {signal.shape}")

    with torch.no_grad():
        if use_depth_wise_conv:
            result = ptwise_conv_layer(conv_layer(signal))
        else:
            result = conv_layer(signal)

    sum_over_channels = torch.sum(result, dim=1, keepdim=True)

    print(f"result shape: {result.shape}")
    print(f"sum_over_channels shape: {sum_over_channels.shape}")

    # Plot signals.
    fig_s, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))
    for b in range(batch_size):
        for j in range(in_channels):
            axs[0].plot(signal[b, j, :].squeeze(0).numpy())
        for k in range(out_channels):
            axs[1].plot(result[b, k, :].squeeze(0).numpy())
        axs[2].plot(sum_over_channels[b, 0, :].squeeze(0).numpy())
    axs[0].set_ylabel("input signal")
    axs[1].set_ylabel("individual convolutions")
    axs[2].set_ylabel("summed over channels")

    # Plot weights.
    fig_w = plt.figure(figsize=(8, 12))
    gs = fig_w.add_gridspec(nrows=out_channels, ncols=in_channels)
    for j in range(out_channels):
        for k in range(in_channels):
            ax = fig_w.add_subplot(gs[j, k])
            if use_depth_wise_conv:
                ax.plot(ptwise_conv_layer.weight[j, k, :].detach().cpu().numpy())
            else:
                ax.plot(conv_layer.weight[j, k, :].detach().cpu().numpy())
            ax.set_xlabel(f"in channel {k}")
            ax.set_ylabel(f"out channel {j}")

    plt.show()
