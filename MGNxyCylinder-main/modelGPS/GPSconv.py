import inspect
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Sequential

from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch

from modelGPS.convBlock import EdgeConv


class GPSConv(torch.nn.Module):

    def __init__(
        self,
        channels: int,
        edge_channels: int,
        heads: int = 1,
        act: str = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Optional[str] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        attn_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.channels = channels
        self.conv = EdgeConv(channels, channels, edge_feat_dim=edge_channels)
        self.heads = heads

        attn_kwargs = attn_kwargs or {}

        self.attn = PerformerAttention(
            channels=channels,
            heads=heads,
            **attn_kwargs,
        )

        self.mlp = Sequential(
            Linear(channels, channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            Linear(channels * 2, channels),
        )

        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)

        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = 'batch' in signature.parameters

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.conv is not None:
            self.conv.reset_parameters()
        self.attn._reset_parameters()
        reset(self.mlp)
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.norm3 is not None:
            self.norm3.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tensor:

        hs = []
        if self.conv is not None:  # Local MPNN.
            h, edge_attr = self.conv(x, edge_index, edge_attr=edge_attr)
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h = self.norm1(h, batch=batch)
                else:
                    h = self.norm1(h)
            h = F.relu(h)
            hs.append(h)

        # Global attention transformer-style model.
        h, mask = to_dense_batch(x, batch)

        h = self.attn(h, mask=mask)

        h = h[mask]
        h = F.dropout(h, p=0.5, training=self.training)
        h = h + x  # Residual connection.
        if self.norm2 is not None:
            if self.norm_with_batch:
                h = self.norm2(h, batch=batch)
            else:
                h = self.norm2(h)
        # h = F.relu(h)
        hs.append(h)

        out = sum(hs)  # Combine local and global outputs.

        out = out + self.mlp(out)
        if self.norm3 is not None:
            if self.norm_with_batch:
                out = self.norm3(out, batch=batch)
            else:
                out = self.norm3(out)
        out = F.relu(out)

        return out, edge_attr

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.channels}, "
            f"conv={self.conv}, heads={self.heads})"
        )
