from torch import nn
from torch.nn import functional as F


class Swish(nn.SiLU):
    """
    Swish activation function introduced in 'https://arxiv.org/abs/1710.05941'
    Mathematically identical to SiLU. See note in nn.SiLU for references.
    """


class GEGLU(nn.Module):
    """
    Gated Exponential Linear Unit (GEGLU) activation function introduced in 'https://arxiv.org/abs/1606.08415'
    """

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)
