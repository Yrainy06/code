import torch
import torch.nn as nn
from utils.utils import decompose_graph, copy_geometric_data
from torch_geometric.data import Data

import torch.nn.functional as F
from torch.nn import LayerNorm
from torch_geometric.nn import BatchNorm

from modelGPS.basicEdgeConv import basicEdge


class EdgeLayers(nn.Module):

    def __init__(self, message_passing_num, hidden_size=128, edge_size=128):
        super(EdgeLayers, self).__init__()

        self.first_convs = torch.nn.ModuleList()
        self.first_norms = torch.nn.ModuleList()

        # Create layers with edge features
        for _ in range(message_passing_num):
            self.first_convs.append(
                basicEdge(hidden_size, hidden_size, edge_feat_dim=edge_size)
            )
            self.first_norms.append(LayerNorm(hidden_size))

    def forward(self, x, edge_index, edge_attr):

        # Initial Layers and norms
        # Apply normalization before activation
        for conv, norm in zip(self.first_convs, self.first_norms):
            old_edge_attr = edge_attr

            x, edge_attr = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = F.relu(x)

            edge_attr = old_edge_attr + edge_attr

        return x, edge_attr
