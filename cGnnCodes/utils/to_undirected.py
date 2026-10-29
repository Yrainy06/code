import torch
from torch_geometric.data import Data

def make_edges_undirected(data: Data) -> Data:

    # 1. Clone original (directed, unique) edges
    data.edge_ind_unique = data.edge_index.clone()
    
    # 2. Compute reversed edges by swapping source and target
    row, col = data.edge_ind_unique
    reversed_edges = torch.stack([col, row], dim=0)
    
    # 3. Concatenate original + reversed (no sorting)
    data.edge_index = torch.cat([data.edge_ind_unique, reversed_edges], dim=1)
    
    return data
