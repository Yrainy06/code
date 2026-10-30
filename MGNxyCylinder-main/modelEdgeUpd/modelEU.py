import torch
import torch.nn as nn
from utils.utils import decompose_graph, copy_geometric_data
from torch_geometric.data import Data

import torch.nn.functional as F
from torch.nn import LayerNorm
from torch_geometric.nn import BatchNorm

from modelEdgeUpd.convBlock import EdgeConv


# Define module-level constants
EDGE_FEATURE_SIZE = 128


def build_mlp(in_size, hidden_size, out_size, lay_norm=True):

    module = nn.Sequential(
        nn.Linear(in_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, out_size),
    )
    if lay_norm:
        return nn.Sequential(module, nn.LayerNorm(normalized_shape=out_size))

    return module


class Encoder(nn.Module):

    def __init__(self, edge_input_size=128, node_input_size=128, hidden_size=128):
        super(Encoder, self).__init__()

        self.eb_encoder = build_mlp(edge_input_size, EDGE_FEATURE_SIZE, EDGE_FEATURE_SIZE)
        self.nb_encoder = build_mlp(node_input_size, hidden_size, hidden_size)

    def forward(self, graph):

        node_attr, _, edge_attr, _ = decompose_graph(graph)
        node_ = self.nb_encoder(node_attr)
        edge_ = self.eb_encoder(edge_attr)

        return Data(x=node_, edge_attr=edge_, edge_index=graph.edge_index)


class Decoder(nn.Module):

    def __init__(self, hidden_size=128, output_size=2):
        super(Decoder, self).__init__()
        self.decode_module = build_mlp(hidden_size, hidden_size, output_size, lay_norm=False)

    def forward(self, graph):
        return self.decode_module(graph.x)


class EncoderProcesserDecoder(nn.Module):

    def __init__(self, message_passing_num, node_input_size, edge_input_size, hidden_size=128):
        super(EncoderProcesserDecoder, self).__init__()

        self.encoder = Encoder(
            edge_input_size=edge_input_size,
            node_input_size=node_input_size,
            hidden_size=hidden_size,
        )

        self.first_convs = torch.nn.ModuleList()
        self.first_norms = torch.nn.ModuleList()

        # Create layers with edge features
        for _ in range(message_passing_num):
            self.first_convs.append(
                EdgeConv(hidden_size, hidden_size, edge_feat_dim=EDGE_FEATURE_SIZE)
            )
            self.first_norms.append(LayerNorm(hidden_size))

        self.decoder = Decoder(hidden_size=hidden_size, output_size=2)

    def forward(self, graph):

        graph = self.encoder(graph)

        x, edge_index, edge_attr, _ = decompose_graph(graph)

        # Initial Layers and norms
        # Apply normalization before activation
        for conv, norm in zip(self.first_convs, self.first_norms):
            old_edge_attr = edge_attr

            x, edge_attr = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = F.relu(x)

            edge_attr = old_edge_attr + edge_attr

        graph = Data(x=x, edge_attr=edge_attr, edge_index=edge_index)

        decoded = self.decoder(graph)

        return decoded
