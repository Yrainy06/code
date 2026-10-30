import torch.nn as nn
import torch
from torch_geometric.data import Data
from utils import normalization
import os

from .model import EncoderProcesserDecoder
# from modelEdgeUpd.modelEU import EncoderProcesserDecoder
# from modelGPS.modelGPS import EncoderProcesserDecoder


class Simulator(nn.Module):

    def __init__(self, message_passing_num, node_input_size, edge_input_size, device, use_gru=False, model_dir='checkpoint/simulator.pth') -> None:
        super(Simulator, self).__init__()

        self.node_input_size =  node_input_size
        self.edge_input_size = edge_input_size
        self.model_dir = model_dir
        self.use_gru = use_gru
        self.model = EncoderProcesserDecoder(message_passing_num=message_passing_num, node_input_size=node_input_size, edge_input_size=edge_input_size).to(device)
        self._output_normalizer = normalization.Normalizer(size=2, name='output_normalizer', device=device)
        self._node_normalizer = normalization.Normalizer(size=node_input_size, name='node_normalizer', device=device)
        # self._edge_normalizer = normalization.Normalizer(size=edge_input_size, name='edge_normalizer', device=device)

        # LSTM for temporal modeling
        # self.lstm_input_size = 128 + 2  # GNN hidden + velocity
        self.lstm_input_size = 128

        self.lstm_hidden_size = 256
        # LSTM or GRU temporal module
        if use_gru:
            self.rnn_type = 'GRU'
            self.rnn = nn.GRU(input_size=self.lstm_input_size,
                              hidden_size=self.lstm_hidden_size, 
                              num_layers=1,
                              batch_first=True)
        else:
            self.rnn_type = 'LSTM'
            self.rnn = nn.LSTM(input_size=self.lstm_input_size,
                               hidden_size=self.lstm_hidden_size,
                               num_layers=1,
                               batch_first=True)
        # self.lstm_predictor = nn.LSTM(input_size=self.lstm_input_size, hidden_size=self.lstm_hidden_size, num_layers=1, batch_first=True)

        self.output_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, self.lstm_hidden_size),
            nn.ReLU(),
            nn.Linear(self.lstm_hidden_size, 2)  # output acceleration or velocity
        )
        # from .model import build_mlp
        # self.output_head = build_mlp(self.lstm_hidden_size, self.lstm_hidden_size, 2, lay_norm=False)

        # self.lstm_predictor = self.lstm_predictor.to(device)
        self.rnn = self.rnn.to(device)
        self.output_head = self.output_head.to(device)


        print('Simulator model initialized')

    def update_node_attr(self, frames, types:torch.Tensor):
        node_feature = []

        node_feature.append(frames) #velocity
        node_type = torch.squeeze(types.long())
        one_hot = torch.nn.functional.one_hot(node_type, 9)
        node_feature.append(one_hot)
        node_feats = torch.cat(node_feature, dim=1)
        attr = self._node_normalizer(node_feats, self.training)

        return attr

    def velocity_to_accelation(self, noised_frames, next_velocity):

        acc_next = next_velocity - noised_frames
        return acc_next


    def forward(self, graph:Data, velocity_sequence_noise, hidden=None):
        
        node_type = graph.x[:, 0:1]
        frames = graph.x[:, 1:3]
        if self.training:
            target = graph.y
            noised_frames = frames + velocity_sequence_noise
        else:
            noised_frames = frames

        node_attr = self.update_node_attr(noised_frames, node_type)
        graph.x = node_attr

        # Step 1: spatial encoder-processor-decoder
        graph_encoded = self.model.encoder(graph)
        for layer in self.model.processer_list:
            graph_encoded = layer(graph_encoded)
        gnn_out = graph_encoded.x  # shape: [num_nodes, 128]

        rnn_input = gnn_out.unsqueeze(1)  # [num_nodes, 1, 128]
        rnn_out, hidden = self.rnn(rnn_input, hidden)  # GRU: hidden=(h,), LSTM: hidden=(h,c)
        rnn_out = rnn_out.squeeze(1)  # [num_nodes, hidden_size]

        # Step 3: LSTM temporal modeling
        # predicted = self.model.decoder.decode_module(rnn_out)  # [num_nodes, 2]
        predicted = self.output_head(rnn_out)  # [num_nodes, 2]

        # Step 4: MLP output head
        # predicted = self.output_head(lstm_out)  # [num_nodes, 2]

        if self.training:
            target_acc = self.velocity_to_accelation(noised_frames, target)
            target_acc_normalized = self._output_normalizer(target_acc, self.training)
            return predicted, target_acc_normalized, hidden
        else:
            velocity_update = self._output_normalizer.inverse(predicted)
            predicted_velocity = frames + velocity_update
            return predicted_velocity, hidden


    def load_checkpoint(self, ckpdir=None):
        
        if ckpdir is None:
            ckpdir = self.model_dir
        dicts = torch.load(ckpdir)
        self.load_state_dict(dicts['model'])

        keys = list(dicts.keys())
        keys.remove('model')

        for k in keys:
            v = dicts[k]
            for para, value in v.items():
                object = eval('self.'+k)
                setattr(object, para, value)

        print("Simulator model loaded checkpoint %s"%ckpdir)

    def save_checkpoint(self, savedir=None):
        if savedir is None:
            savedir=self.model_dir

        # os.makedirs(os.path.dirname(self.model_dir), exist_ok=True)
        # ✅ 用 savedir 的目录，而不是 self.model_dir
        os.makedirs(os.path.dirname(self.model_dir), exist_ok=True)

        
        model = self.state_dict()
        _output_normalizer = self._output_normalizer.get_variable()
        _node_normalizer  = self._node_normalizer.get_variable()
        # _edge_normalizer = self._edge_normalizer.get_variable()

        to_save = {'model':model, '_output_normalizer':_output_normalizer, '_node_normalizer':_node_normalizer}

        torch.save(to_save, savedir)
        print('Simulator model saved at %s'%savedir)