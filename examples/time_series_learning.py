import matplotlib.pyplot as plt
import numpy
import seaborn
import torch

from neuralfields import EXAMPLES_DIR, NeuralField, SimpleNeuralField, pd_capacity_21_abs


def load_and_split_data(dataset_name: str, normalize: bool = False):
    # Load and move the torch.
    data = torch.from_numpy(numpy.load(EXAMPLES_DIR / f"{dataset_name}.npy"))
    data = torch.atleast_2d(data).float().contiguous()
    if data.size(0) == 1:
        # torch.atleast_2d() adds dimensions to the front, but we want the first dim to be the length of the series.
        data = data.T

    # Make the first half the training set and the second half the test set.
    num_samples = len(data) // 2
    if normalize:
        scale = data[:num_samples].abs().max()
        if scale > 0:
            data /= scale

    data_trn = data[:num_samples]
    data_tst = data[num_samples : num_samples * 2]

    return data, data_trn, data_tst


def create_windows(data: torch.Tensor, len_window: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pair left-padded histories with the next observation to predict."""
    inputs = torch.nn.functional.pad(data[:-1], (0, 0, len_window - 1, 0))
    inputs = inputs.unfold(0, len_window, 1).permute(0, 2, 1).contiguous()
    return inputs, data[1:].unsqueeze(1)


def create_figure(dataset_name: str) -> plt.Figure:
    if dataset_name not in ("monthly_sunspots", "mackey_glass"):
        raise NotImplementedError(f"Unknown dataset {dataset_name}! Please specify the necessary parts in the script.")

    fig, axs = plt.subplots(2, 1, figsize=(16, 9))
    axs[0].plot(data_trn, label="data train")
    axs[1].plot(data_tst, label="data test")
    axs[1].set_xlabel("months" if dataset_name == "monthly_sunspots" else "time")
    axs[0].set_ylabel("spot count" if dataset_name == "monthly_sunspots" else "P")
    axs[1].set_ylabel("spot count" if dataset_name == "monthly_sunspots" else "P")

    return fig


def plot_results(fig: plt.Figure) -> None:
    axs = fig.get_axes()
    axs[0].plot(numpy.arange(1, len(predictions_trn) + 1), predictions_trn, label="predictions")
    axs[1].plot(numpy.arange(1, len(predictions_tst) + 1), predictions_tst, label="predictions")
    axs[0].legend(loc="upper right", ncol=2)
    axs[1].legend(loc="upper right", ncol=2)


def simple_training_loop(
    model: torch.nn.Module,
    packed_inputs: torch.Tensor,
    packed_targets: torch.Tensor,
    dataset_name: str,
    normalized_data: bool,
    num_epochs: int | None = None,
) -> None:
    # Use a simple heuristic for the optimization hyper-parameters.
    if dataset_name == "monthly_sunspots" and not normalized_data:  # data is in [0, 100] so we need more steps
        if num_epochs is None:
            num_epochs = 10001 if isinstance(model, SimpleNeuralField) else 4001
        lr = 1e-1
    else:
        if num_epochs is None:
            num_epochs = 2501 if isinstance(model, SimpleNeuralField) else 501
        lr = 1e-2

    model.train()
    device = next(model.parameters()).device
    packed_inputs = packed_inputs.to(device=device)
    packed_targets = packed_targets.to(device=device)

    loss_fcn = torch.nn.MSELoss()
    optim = torch.optim.Adam([{"params": model.parameters()}], lr=lr, eps=1e-8)

    for idx_e in range(num_epochs + 1):
        # Reset the gradients.
        optim.zero_grad(set_to_none=True)

        # Make the predictions.
        packed_predictions, _ = model(packed_inputs, hidden=None)
        loss = loss_fcn(packed_predictions[:, -1:, :], packed_targets)

        # Call optimizer.
        loss.backward()
        optim.step()

        if idx_e % 10 == 0:
            print(f"iter: {idx_e: >4} | loss: {loss.item()}")


if __name__ == "__main__":
    seaborn.set_theme()

    # Configure.
    torch.manual_seed(0)
    use_simplification = False  # switch between models
    normalize_data = True  # scales the data to be in [-1, 1]
    dataset_name = "monthly_sunspots"  # monthly_sunspots or mackey_glass

    # Get the data.
    data, data_trn, data_tst = load_and_split_data(dataset_name, normalize_data)
    dim_data = data.size(1)

    fig = create_figure(dataset_name)

    # Create the neural field.
    if use_simplification:
        model = SimpleNeuralField(
            input_size=dim_data,
            output_size=dim_data,
            potentials_dyn_fcn=pd_capacity_21_abs,
            activation_nonlin=torch.tanh,
        )
    else:
        model = NeuralField(input_size=dim_data, hidden_size=13, output_size=dim_data, conv_out_channels=1)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Create data set with rolling forecast scheme.
    # i t ...
    # i i t ...
    # i i ... t
    len_window = 10  # tested 10 and 20
    packed_inputs, packed_targets = create_windows(data_trn, len_window)

    # Run a simple optimization loop.
    simple_training_loop(model, packed_inputs, packed_targets, dataset_name, normalize_data)

    # Evaluate model.
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        predictions_trn, _ = model(packed_inputs.to(device), hidden=None)
        predictions_trn = predictions_trn[:, -1, :].cpu().numpy()

        inputs_tst, _ = create_windows(data_tst, len_window)
        predictions_tst, _ = model(inputs_tst.to(device), hidden=None)
        predictions_tst = predictions_tst[:, -1, :].cpu().numpy()

    # Safe the model and the associated data.
    torch.save(model, EXAMPLES_DIR / "model.pt")
    torch.save(data_trn, EXAMPLES_DIR / "data_trn.pt")
    torch.save(data_tst, EXAMPLES_DIR / "data_tst.pt")

    # Plot the results.
    plot_results(fig)
    plt.show()
