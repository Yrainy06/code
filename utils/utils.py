import torch
from torch_geometric.data import Data
import enum

class NodeType(enum.IntEnum):
    NORMAL = 0
    WALL = 1
    OUTFLOW = 2
    INFLOW = 3



def decompose_graph(graph):
    x, edge_index, edge_attr, global_attr = None, None, None, None
    for key in graph.keys():
        if key=="x":
            x = graph.x
        elif key=="edge_index":
            edge_index = graph.edge_index
        elif key=="edge_attr":
            edge_attr = graph.edge_attr
        elif key=="global_attr":
            global_attr = graph.global_attr
        else:
            pass
    return (x, edge_index, edge_attr, global_attr)


def copy_geometric_data(graph):
    """return a copy of torch_geometric.data.data.Data
    This function should be carefully used based on
    which keys in a given graph.
    """
    node_attr, edge_index, edge_attr, global_attr = decompose_graph(graph)
    
    ret = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)
    ret.global_attr = global_attr
    
    return ret
