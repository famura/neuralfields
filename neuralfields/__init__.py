from pathlib import Path

from neuralfields.custom_layers import IndependentNonlinearitiesLayer, MirroredConv1d, init_param_
from neuralfields.custom_types import ActivationFunction, PotentialsDynamicsType
from neuralfields.neural_fields import NeuralField
from neuralfields.potential_based import PotentialBased
from neuralfields.simple_neural_fields import (
    SimpleNeuralField,
    pd_capacity_21,
    pd_capacity_21_abs,
    pd_capacity_32,
    pd_capacity_32_abs,
    pd_cubic,
    pd_linear,
)


# Define variables for important folders.
SRC_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = SRC_DIR.parent / "examples"

# Set the public API.
__all__ = [
    "EXAMPLES_DIR",
    "SRC_DIR",
    "ActivationFunction",
    "IndependentNonlinearitiesLayer",
    "MirroredConv1d",
    "NeuralField",
    "PotentialBased",
    "PotentialsDynamicsType",
    "SimpleNeuralField",
    "init_param_",
    "pd_capacity_21",
    "pd_capacity_21_abs",
    "pd_capacity_32",
    "pd_capacity_32_abs",
    "pd_cubic",
    "pd_linear",
]
