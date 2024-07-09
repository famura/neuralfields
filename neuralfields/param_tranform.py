import torch


def inv_softplus(x: torch.Tensor) -> torch.Tensor:
    r"""Computes the inverse of the softplus function which is defined as:

    $$
        f(x) = \mathrm{ln}( 1 + \mathrm{e}^x ).
    $$

    Thus, its inverse is:
    $$
        f^{-1}(x) = \mathrm{ln}( \mathrm{e}^x - 1 ).
    $$

    Args:
        x: Input tensor to be transformed to a range of $(-\infty, \infty)$.

    Returns:
        Value of the inverse softplus at the input `x`.
    """
    return torch.log(torch.expm1(x))
