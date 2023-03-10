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

    if normalize:
        data /= max(abs(data.min()), abs(data.max()))

    # Make the first half the training set and the second half the test set.
    num_samples = len(data) // 2
    data_trn = data[:num_samples]
    data_tst = data[num_samples : num_samples * 2]

    return data, data_trn, data_tst


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
    axs[0].plot(predictions_trn, label="predictions")
    axs[1].plot(predictions_tst, label="predictions")
    axs[0].legend(loc="upper right", ncol=2)
    axs[1].legend(loc="upper right", ncol=2)


def simple_training_loop(
    model: torch.nn.Module,
    packed_inputs: torch.Tensor,
    packed_targets: torch.Tensor,
    dataset_name: str,
    normalized_data: bool,
):
    # Use a simple heuristic for the optimization hyper-parameters.
    if dataset_name == "monthly_sunspots" and not normalized_data:  # data is in [0, 100] so we need more steps
        num_epochs = 10001 if isinstance(model, SimpleNeuralField) else 4001
        lr = 1e-1
    else:
        num_epochs = 2501 if isinstance(model, SimpleNeuralField) else 501
        lr = 1e-2

    loss_fcn = torch.nn.MSELoss()
    optim = torch.optim.Adam([{"params": model.parameters()}], lr=lr, eps=1e-8)

    for idx_e in range(num_epochs + 1):
        # Reset the gradients.
        optim.zero_grad(set_to_none=True)

        # Make the predictions.
        packed_predictions, _ = model(packed_inputs, hidden=None)
        loss = loss_fcn(packed_predictions, packed_targets)

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

    # Create data set with rolling forcast scheme.
    # i t ...
    # i i t ...
    # i i ... t
    inputs = []
    targets = []
    len_window = 10  # tested 10 and 20
    for idx in range(1, data_trn.size(0)):
        # Slice the input.
        idx_begin = max(idx - len_window, 0)
        inp = data_trn[idx_begin:idx, :].view(-1, dim_data)

        # Pad with zeros. This is not special to the models in this repo, but rather to the dataset structure.
        pad = (0, 0, len_window - inp.size(0), 0)  # from the left pad such that the input length is always 20
        inp_padded = torch.nn.functional.pad(inp, pad, mode="constant", value=0)

        # Store the data.
        inputs.append(inp_padded)
        targets.append(data_trn[idx, :].view(-1, dim_data))

    # Collect all and bring it in the form for batch processing.
    packed_inputs = torch.stack(inputs, dim=0)
    packed_targets = torch.stack(inputs, dim=0)

    # Run a simple optimization loop.
    simple_training_loop(model, packed_inputs, packed_targets, dataset_name, normalize_data)

    # Evaluate model.
    with torch.no_grad():
        predictions_trn, _ = model(data_trn[:-1].unsqueeze(0), hidden=None)
        predictions_trn = predictions_trn.squeeze(0).detach().numpy()

        predictions_tst, _ = model(data_tst[:-1].unsqueeze(0), hidden=None)
        predictions_tst = predictions_tst.squeeze(0).detach().numpy()

    # Safe the model and the associated data.
    torch.save(model, EXAMPLES_DIR / "model.pt")
    torch.save(data_trn, EXAMPLES_DIR / "data_trn.pt")
    torch.save(data_tst, EXAMPLES_DIR / "data_tst.pt")

    # Plot the results.
    plot_results(fig)
    plt.show()
