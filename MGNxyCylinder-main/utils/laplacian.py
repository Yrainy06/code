from typing import Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.typing import OptTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes


def laplacianNorm(
    edge_index: Tensor,
    edge_weight: OptTensor = None,
    dtype: Optional[torch.dtype] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:

    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=dtype,
                                 device=edge_index.device)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce="sum")

    # Compute A_norm = -D^{-1/2} A D^{-1/2}.
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight
