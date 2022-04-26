import pytest
import numpy as np
import torch


@pytest.fixture(autouse=True)
def random_seed():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_grad_enabled(False)
