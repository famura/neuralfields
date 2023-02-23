from typing import Any, Callable

import torch


ActivationFunction = Callable[[torch.Tensor], torch.Tensor]

PotentialsDynamicsType = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any], torch.Tensor]
