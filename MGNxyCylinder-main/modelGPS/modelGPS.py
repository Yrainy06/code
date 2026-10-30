import torch
import torch.nn as nn
from utils.utils import decompose_graph, copy_geometric_data
from torch_geometric.data import Data

import torch.nn.functional as F
from torch_geometric.nn import LayerNorm
from torch_geometric.nn import BatchNorm

from modelGPS.GPSconv import GPSConv
from modelGPS.EdgeLayers import EdgeLayers


# Define module-level constants
EDGE_FEATURE_SIZE = 128

NUM_EDGE_LS = 1


def build_mlp(in_size, hidden_size, out_size, lay_norm=True):

    module = nn.Sequential(
        nn.Linear(in_size, hidden_size),
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

        return Data(
            x=node_, edge_attr=edge_, edge_index=graph.edge_index, batch=graph.batch
        )


class Decoder(nn.Module):

    def __init__(self, hidden_size=128, output_size=2):
        super(Decoder, self).__init__()
        self.decode_module = build_mlp(hidden_size, hidden_size, output_size, lay_norm=False)

    def forward(self, graph):
        return self.decode_module(graph.x)


class EncoderProcesserDecoder(nn.Module):

    def __init__(self, message_passing_num, node_input_size, edge_input_size, hidden_size=128):
        super(EncoderProcesserDecoder, self).__init__()

        # Nums of Layers
        # Nums of (Local+Global)
        numT_LG = NUM_EDGE_LS + 2
        numRounds = message_passing_num // numT_LG

        self.encoder = Encoder(
            edge_input_size=edge_input_size,
            node_input_size=node_input_size,
            hidden_size=hidden_size,
        )

        self.edge_convs = torch.nn.ModuleList()
        self.first_convs = torch.nn.ModuleList()

        # Create layers with edge features
        for _ in range(numRounds):
            self.edge_convs.append(
                EdgeLayers(NUM_EDGE_LS, hidden_size, EDGE_FEATURE_SIZE)
            )

        for _ in range(numRounds):
            self.first_convs.append(
                GPSConv(hidden_size, EDGE_FEATURE_SIZE, heads=4, norm="batch_norm")
            )

        self.decoder = Decoder(hidden_size=hidden_size, output_size=2)

    def forward(self, graph):

        graph = self.encoder(graph)

        x, edge_index, edge_attr, _ = decompose_graph(graph)

        for edgeConv, conv in zip(self.edge_convs, self.first_convs):

            # basic edge conv layers
            x, edge_attr = edgeConv(x, edge_index, edge_attr=edge_attr)

            # Local+Global layer
            old_edge_attr = edge_attr

            x, edge_attr = conv(x, edge_index, edge_attr=edge_attr, batch=graph.batch)

            edge_attr = old_edge_attr + edge_attr

        graph = Data(x=x, edge_attr=edge_attr, edge_index=edge_index)

        decoded = self.decoder(graph)

        return decoded
