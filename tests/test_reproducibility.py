"""Tests for reproducibility utilities."""

import numpy as np
import torch

from src.utils.reproducibility import set_seed


def test_set_seed() -> None:
    """Test reproducible random numbers."""
    set_seed(42)

    numpy_a = np.random.rand()
    torch_a = torch.rand(1)

    set_seed(42)

    numpy_b = np.random.rand()
    torch_b = torch.rand(1)

    assert numpy_a == numpy_b
    assert torch.equal(
        torch_a,
        torch_b,
    )
