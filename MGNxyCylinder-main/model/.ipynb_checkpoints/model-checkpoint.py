import torch.nn as nn
import torch
from .blocks import EdgeBlock, NodeBlock
from utils.utils import decompose_graph, copy_geometric_data
from torch_geometric.data import Data


def build_mlp(in_size, hidden_size, out_size, lay_norm=True, device='cuda'):

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

    return module.to(device)


class Encoder(nn.Module):

    def __init__(self,
                edge_input_size=128,
                node_input_size=128,
                hidden_size=128,
                device='cuda'):
        super(Encoder, self).__init__()

        self.eb_encoder = build_mlp(edge_input_size, hidden_size, hidden_size, device=device)
        self.nb_encoder = build_mlp(node_input_size, hidden_size, hidden_size)
        # self.nb_encoder = None  # 延迟初始化
        # self.hidden_size = hidden_size   
        # self.device = device

    def forward(self, graph):

        node_attr, _, edge_attr, _ = decompose_graph(graph)

        # device = node_attr.device  # 获取输入数据的设备

        # # 动态初始化 nb_encoder，确保输入维度匹配
        # if self.nb_encoder is None:
        #     self.nb_encoder = build_mlp(node_attr.shape[1], self.hidden_size, self.hidden_size).to(device) 

        node_ = self.nb_encoder(node_attr)
        edge_ = self.eb_encoder(edge_attr)

        
        return Data(x=node_, edge_attr=edge_, edge_index=graph.edge_index)


class GnBlock(nn.Module):

    def __init__(self, hidden_size=128):

        super(GnBlock, self).__init__()


        eb_input_dim = 3 * hidden_size
        nb_input_dim = 2 * hidden_size
        nb_custom_func = build_mlp(nb_input_dim, hidden_size, hidden_size)
        eb_custom_func = build_mlp(eb_input_dim, hidden_size, hidden_size)
        
        self.eb_module = EdgeBlock(custom_func=eb_custom_func)
        self.nb_module = NodeBlock(custom_func=nb_custom_func)

    def forward(self, graph):
    
        graph_last = copy_geometric_data(graph)
        graph = self.eb_module(graph)
        graph = self.nb_module(graph)
        edge_attr = graph_last.edge_attr + graph.edge_attr
        x = graph_last.x + graph.x
        return Data(x=x, edge_attr=edge_attr, edge_index=graph.edge_index)


class Decoder(nn.Module):

    def __init__(self, hidden_size=128, output_size=2):
        super(Decoder, self).__init__()
        self.decode_module = build_mlp(hidden_size, hidden_size, output_size, lay_norm=False)

    def forward(self, graph):
        return self.decode_module(graph.x)


class EncoderProcesserDecoder(nn.Module):

    def __init__(self, message_passing_num, node_input_size, edge_input_size, hidden_size=128):

        super(EncoderProcesserDecoder, self).__init__()

        self.encoder = Encoder(edge_input_size=edge_input_size, node_input_size=node_input_size, hidden_size=hidden_size)

        processer_list = []
        for _ in range(message_passing_num):
            processer_list.append(GnBlock(hidden_size=hidden_size))
        self.processer_list = nn.ModuleList(processer_list)

        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        
        self.decoder = Decoder(hidden_size=hidden_size, output_size=2)

    def forward(self, graph):

        graph= self.encoder(graph)
        
        for model in self.processer_list:
            graph = model(graph)

        # lstm_out, _ = self.lstm(graph.x)
        # graph.x = lstm_out
        # Apply one-step LSTM for temporal processing
        node_features = graph.x.unsqueeze(1)  # Add a sequence dimension (seq_len=1)
        lstm_out, _ = self.lstm(node_features)
        graph.x = lstm_out.squeeze(1)  # Remove the sequence dimension
        
        decoded = self.decoder(graph)    

        return decoded
    