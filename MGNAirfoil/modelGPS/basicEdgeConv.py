import torch
from torch.nn import Sequential as Seq, Linear, LayerNorm, ReLU
from torch_geometric.nn import MessagePassing


class basicEdge(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_feat_dim):
        super().__init__(aggr="add")

        self.mlp = Seq(
            Linear(2 * in_channels + edge_feat_dim, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
            LayerNorm(out_channels),
        )
        self.lin_root = Linear(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index, edge_attr):
        # edge_index: [2, E]
        # edge_attr:  [E, edge_feat_dim]

        # Get source and target node features
        x_j = x[edge_index[0]]
        x_i = x[edge_index[1]]

        # Compute new edge features separately
        tmp = torch.cat([x_i, x_j, edge_attr], dim=1)
        new_edge_attr = self.mlp(tmp)  # [E, out_channels]

        # Propagate messages using new_edge_attr
        out = self.propagate(edge_index, x=x, new_edge_attr=new_edge_attr)

        out = out + self.lin_root(x)
        return out, new_edge_attr

    def message(self, new_edge_attr):
        # Use new_edge_attr directly as messages
        return new_edge_attr
