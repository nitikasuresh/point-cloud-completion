from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Given input (batch_size, M, d_model), that will be used
    to compute query, keys and values, we output a self attention
    tensor of shape (batch_size, M, d_model).

    Inputs
    ----------
    M:
        Number of clusters.
    d_model:
        Dimension of features.

    Parameters
    ----------
    d_model:
        Dimension of features.
    q:
        Dimension of query matrix.
    v:
        Dimension of value matrix.
        note: q usually equals v
    h:
        Number of heads.
    attention_size:
        Number of neighboring elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    """

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 attention_size: int = None):
        """Initialize the Multi Head Block."""
        super().__init__()

        self._h = h
        self._attention_size = attention_size

        # Query, keys and value matrices
        self._W_q = nn.Linear(d_model, q * self._h)
        self._W_k = nn.Linear(d_model, q * self._h)
        self._W_v = nn.Linear(d_model, v * self._h)

        # Output linear function
        self._W_o = nn.Linear(self._h * v, d_model)

        # Score placeholder
        self._scores = None

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        """Propagate forward the input through the MHB.

        We compute for each head the queries, keys and values matrices,
        followed by the Scaled Dot-Product. The result is concatenated
        and returned with shape (batch_size, M, d_model).

        Parameters
        ----------
        query:
            Input tensor with shape (batch_size, M, d_model) used to compute queries.
        key:
            Input tensor with shape (batch_size, M, d_model) used to compute keys.
        value:
            Input tensor with shape (batch_size, M, d_model) used to compute values.

        Returns
        -------
            Self attention tensor with shape (batch_size, M, d_model).
        """
        M = query.shape[1]

        # Compute Q, K and V, concatenate heads on batch dimension
        queries = torch.cat(self._W_q(query).chunk(self._h, dim=-1), dim=0)
        keys = torch.cat(self._W_k(key).chunk(self._h, dim=-1), dim=0)
        values = torch.cat(self._W_v(value).chunk(self._h, dim=-1), dim=0)

        # Scaled Dot Product
        self._scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(M)

        # Apply sotfmax
        self._scores = F.softmax(self._scores, dim=-1)

        attention = torch.bmm(self._scores, values)

        # Concatenat the heads
        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)

        # Apply linear transformation W^O
        self_attention = self._W_o(attention_heads)

        return self_attention

    @property
    def attention_map(self) -> torch.Tensor:
        """Attention map after a forward propagation,
        variable `score` in the original paper.
        """
        if self._scores is None:
            raise RuntimeError(
                "Evaluate the model once to generate attention map")
        return self._scores

