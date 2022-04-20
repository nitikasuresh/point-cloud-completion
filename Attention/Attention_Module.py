import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Cluster_Attention import MultiHeadAttention
from positionwiseFeedForward import PositionwiseFeedForward

class Attention_Module(nn.Module):
    """
    Apply Multi Head Attention block followed by a Point-wise Feed Forward block.
    Residual sum and normalization are applied at each step.

    Parameters
    ----------
    d_model:
        Dimension of feature.
    q:
        Dimension of query matrix.
    v:
        Dimension of value matrix.
    h:
        Number of heads.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    dropout:
        Dropout probability after each MHA or PFF block.
        Default is ``0.3``.
    """
    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 attention_size: int = None,
                 dropout: float = 0.3,
                 M: int = 16):
        """Initialize the Encoder block"""
        super().__init__()

        MHA = MultiHeadAttention

        self._selfAttention = MHA(d_model, q, v, h, attention_size=attention_size)
        self._feedForward = PositionwiseFeedForward(d_model)

        self._layerNorm1 = nn.LayerNorm(d_model)
        self._layerNorm2 = nn.LayerNorm(d_model)

        self._dopout = nn.Dropout(p=dropout)
        # maximum length, use this to do padding
        self._M = M

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate the input through the Encoder block.

        Apply the Multi Head Attention block, add residual and normalize.
        Apply the Point-wise Feed Forward block, add residual and normalize.

        Parameters
        ----------
        x:
            Input tensor with shape (batch_size, K, d_model).

        Returns
        -------
            Output tensor with shape (batch_size, K, d_model).
        """

        # since DBSCAN outputs variable length
        # needs zero padding here
        # (batch_size, K, d_model) -> (batch_size, M, d_model) ->
        # attention mechanism ->
        # truncate back to (batch_size, K, d_model)

        K = x.shape[1]

        if K < self._M:

            fill = self.M - K
            pp = (0, 0, 0, fill)

            # add padding to the end
            x = F.pad(x, pp, 'constant')

        # Self attention
        residual = x
        x = self._selfAttention(query=x, key=x, value=x)
        x = self._dopout(x)
        x = self._layerNorm1(x + residual)

        # Feed forward
        residual = x
        x = self._feedForward(x)
        x = self._dopout(x)
        x = self._layerNorm2(x + residual)

        return x[:, :K, :]

    @property
    def attention_map(self) -> torch.Tensor:
        """Attention map after a forward propagation,
        variable `score` in the original paper.
        """
        return self._selfAttention.attention_map