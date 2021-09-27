import torch
import torch.nn as nn


class ScaleClipper(nn.Module):
    def __init__(self, eps: float = 1e-5):
        """
        Constraints the scale to be positive.

        :param eps: The epsilon minimum value threshold.
        :raises ValueError: If the epsilon value is out of domain.
        """
        if eps <= 0.0:
            raise ValueError("The epsilon value must be positive")
        super(ScaleClipper, self).__init__()
        self.register_buffer('eps', torch.tensor(eps))

    def __call__(self, module: nn.Module):
        """
        Call the constraint.

        :param module: The module.
        """
        with torch.no_grad():
            module.scale.clamp_(self.eps)
