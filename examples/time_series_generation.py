import warnings

import matplotlib.pyplot as plt
import seaborn
import torch

from neuralfields import EXAMPLES_DIR, PotentialBased


if __name__ == "__main__":
    seaborn.set_theme()

    # Configure.
    torch.manual_seed(0)
    num_samples = 5
    use_test_inputs = True  # if False, the inputs will be zero for every sample (vary via the hidden state)
    default_hidden = True  # if True, the hidden state will be initialized with zeroes by the net (vary via the inputs)
    if not use_test_inputs and default_hidden:
        warnings.warn("All generated sequences will be the same. Please change the scripts configuration.")
    len_time_series = 800 if use_test_inputs else 10

    # Load the model previously trained with the time_series_learning.py example script.
    try:
        model = torch.load(EXAMPLES_DIR / "model.pt")
        assert isinstance(model, PotentialBased)
    except FileNotFoundError:
        raise FileNotFoundError(
            "There was no file called 'model.pt' found in neuralfields' example directory. Most likely, you need to "
            "run the 'time_series_learning.py' script first."
        )

    # Use the model to generate several time series. This could either be done by providing inputs that are different
    # along the first dimension of the tensor, or by varying the initial hidden state.
    if use_test_inputs:
        try:
            inputs = torch.load(EXAMPLES_DIR / "data_tst.pt").unsqueeze(0).repeat(num_samples, 1, 1)
        except FileNotFoundError:
            raise FileNotFoundError(
                "There was no file called 'data_tst.pt' found in neuralfields' example directory. Most likely, you "
                "need to run the 'time_series_learning.py' script first."
            )
        inputs = inputs[:, :len_time_series, :]
        inputs[1:] += torch.randn(num_samples - 1, len_time_series, model.input_size) / 50
    else:
        inputs = torch.zeros(num_samples, len_time_series, model.input_size)
    if default_hidden:
        hidden = None
    else:
        hidden = torch.randn(num_samples, model.hidden_size) * 10
    with torch.no_grad():
        generated, _ = model(inputs, hidden)

    # Plot.
    fig, axs = plt.subplots(1, 1, figsize=(16, 9))
    for idx_ts, gen_ts in enumerate(generated):
        plt.plot(gen_ts.numpy(), label=f"sample {idx_ts}")
    axs.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=inputs.size(0))
    axs.set_xlabel("months")
    axs.set_ylabel("spot count")
    plt.show()
