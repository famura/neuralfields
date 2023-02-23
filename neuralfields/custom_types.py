from typing import Callable, Optional

import torch


ActivationFunction = Callable[[torch.Tensor], torch.Tensor]

PotentialsDynamicsType = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]],
    torch.Tensor,
]
