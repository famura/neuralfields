"""This file is found by pytest and contains fixtures that can be used for all tests."""
import pytest
import torch


# Check if CUDA support is available.
m_needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not supported in this setup.")
