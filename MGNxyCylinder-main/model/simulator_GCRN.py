import torch.nn as nn
import torch
from torch_geometric.data import Data
from utils import normalization
import os

from .model import EncoderProcesserDecoder
from .gconv_lstm import GConvLSTM  # ⬅️ 引入 GConvLSTM
from .gconv_lstm import MinLSTMCell
# from .gconv_lstm import MinGRUCell
# from .gconv_lstm import GConvMinLSTM   # ✅ 引入 minLSTM 版本

class Simulator(nn.Module):

    def __init__(self, message_passing_num, node_input_size, edge_input_size, device,
                 use_gru=False, use_gconv_lstm=False, use_minlstm=False, use_mingru=False, use_minlstm_gat=False, model_dir='checkpoint/simulator.pth') -> None:
        super(Simulator, self).__init__()

        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size
        self.model_dir = model_dir
        self.use_gru = use_gru
        self.use_gconv_lstm = use_gconv_lstm
        self.use_minlstm = use_minlstm
        self.use_mingru = use_mingru
        self.use_minlstm_gat = use_minlstm_gat
        self.device = device

        self.model = EncoderProcesserDecoder(
            message_passing_num=message_passing_num,
            node_input_size=node_input_size,
            edge_input_size=edge_input_size
        ).to(device)

        self._output_normalizer = normalization.Normalizer(size=2, name='output_normalizer', device=device)
        self._node_normalizer = normalization.Normalizer(size=node_input_size, name='node_normalizer', device=device)

        self.lstm_input_size = 128
        self.lstm_hidden_size = 256

        # # 统一 projector 映射器（将 GNN 输出从 128 → 256）
        # self.x_projector = nn.Linear(self.lstm_input_size, self.lstm_hidden_size)

        if self.use_gconv_lstm:
            self.rnn_type = "GConvLSTM"
            self.rnn = GConvLSTM(
                in_channels=self.lstm_input_size,
                out_channels=self.lstm_hidden_size,
                K=3,
                normalization="sym"
            )
        # elif self.use_minlstm_gat:
        #     self.rnn_type = 'MinLSTM_GAT'
        #     self.rnn = GConvMinLSTM(
        #         in_channels=self.lstm_input_size,
        #         out_channels=self.lstm_hidden_size,
        #     )

        elif use_minlstm:  # ✅ 新增 minLSTM 分支
            self.rnn_type = 'MinLSTM'
            self.rnn = MinLSTMCell(
                in_channels=self.lstm_input_size,
                out_channels=self.lstm_hidden_size,
            )
        # elif use_mingru:  # ✅ 新增 minGRU 分支
        #     self.rnn_type = 'MinGRU'
        #     self.rnn = MinGRUCell(
        #         in_channels=self.lstm_input_size,
        #         out_channels=self.lstm_hidden_size,
        #     )
        # elif use_minlstm:  # ✅ 新增多层 minLSTM
        #     self.rnn_type = 'MinLSTM'
        #     self.rnn_layers = nn.ModuleList([
        #         MinLSTMCell(
        #             in_channels=self.lstm_input_size if i == 0 else self.lstm_hidden_size,
        #             out_channels=self.lstm_hidden_size,
        #         )
        #         for i in range(2)  # ✅ 2 层
        #     ])
        elif use_gru:
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

        self.output_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, self.lstm_hidden_size),
            nn.ReLU(),
            nn.Linear(self.lstm_hidden_size, 2)
        )

        if not self.use_gconv_lstm:
            self.rnn = self.rnn.to(device)
        # if self.use_gconv_lstm:
        #     pass  # GConvLSTM 自己在内部 to(device)
        # elif self.use_minlstm:
        #     self.rnn_layers = self.rnn_layers.to(device)
        # else:
        #     self.rnn = self.rnn.to(device)

        self.output_head = self.output_head.to(device)

        print('Simulator model initialized')

    def update_node_attr(self, frames, types: torch.Tensor):
        node_feature = [frames]
        node_type = torch.squeeze(types.long())
        one_hot = torch.nn.functional.one_hot(node_type, 9)
        node_feature.append(one_hot)
        node_feats = torch.cat(node_feature, dim=1)
        attr = self._node_normalizer(node_feats, self.training)
        return attr

    def velocity_to_accelation(self, noised_frames, next_velocity):
        return next_velocity - noised_frames

    def forward(self, graph: Data, velocity_sequence_noise, hidden=None):
        node_type = graph.x[:, 0:1]
        frames = graph.x[:, 1:3]

        if self.training:
            target = graph.y
            noised_frames = frames + velocity_sequence_noise
        else:
            noised_frames = frames

        node_attr = self.update_node_attr(noised_frames, node_type)
        graph.x = node_attr

        graph_encoded = self.model.encoder(graph)
        for layer in self.model.processer_list:
            graph_encoded = layer(graph_encoded)
        gnn_out = graph_encoded.x  # [num_nodes, 128]

        # === 时间建模部分 ===
        if self.use_gconv_lstm:
            H = hidden[0] if hidden is not None else None
            C = hidden[1] if hidden is not None else None
            # rnn_out, C = self.rnn(
            #     X=gnn_out,
            #     edge_index=graph.edge_index,
            #     edge_weight=None,
            #     H=H,
            #     C=C,
            #     lambda_max=None  # 可选
            # )
            rnn_out, C = self.rnn(
                X=gnn_out,
                edge_index=graph.edge_index,  
                H=H,
                C=C
            )

            hidden = (rnn_out, C)
        elif self.use_minlstm:  # ✅ 新增 minLSTM 处理
            H_prev = hidden[0] if hidden is not None else None
            C_prev = hidden[1] if hidden is not None else None

            # 单步计算（适用于当前 timestep）
            H_new, C_new = self.rnn(gnn_out, H_prev, C_prev)
            rnn_out = H_new
            hidden = (H_new, C_new)
        # elif self.use_minlstm:
        # # hidden 结构为 [(H0, C0), (H1, C1)]
        #     if hidden is None:
        #         hidden = [None for _ in range(len(self.rnn_layers))]

        #     out = gnn_out
        #     new_hidden = []
        #     for i, rnn_cell in enumerate(self.rnn_layers):
        #         H_prev = hidden[i][0] if hidden[i] is not None else None
        #         C_prev = hidden[i][1] if hidden[i] is not None else None

        #         out, C_out = rnn_cell(out, H_prev, C_prev)
        #         new_hidden.append((out, C_out))

        #     rnn_out = out
        #     hidden = new_hidden
        elif self.use_mingru:
            rnn_out = self.rnn(gnn_out, hidden)  # X: [N, 128]
            hidden = rnn_out
        # elif self.use_minlstm_gat:
        #     H_prev = hidden[0] if hidden is not None else None
        #     C_prev = hidden[1] if hidden is not None else None
        #     H_new, C_new = self.rnn(gnn_out, graph.edge_index, H_prev, C_prev)
        #     rnn_out = H_new
        #     hidden = (H_new, C_new)
        elif self.use_minlstm_gat:
            H_prev = hidden[0] if hidden is not None else None
            H_new = self.rnn(gnn_out, graph.edge_index, H_prev)
            rnn_out = H_new
            hidden = (H_new, None)  # ✅ 保持接口一致

        else:
            rnn_input = gnn_out.unsqueeze(1)  # [N, 1, 128]
            rnn_out, hidden = self.rnn(rnn_input, hidden)
            rnn_out = rnn_out.squeeze(1)

        # predicted = self.output_head(rnn_out)
        predicted = self.output_head(rnn_out.to(self.device))


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
                object = eval('self.' + k)
                setattr(object, para, value)

        print("Simulator model loaded checkpoint %s" % ckpdir)

    def save_checkpoint(self, savedir=None):
        if savedir is None:
            savedir = self.model_dir

        os.makedirs(os.path.dirname(self.model_dir), exist_ok=True)
        # os.makedirs(os.path.dirname(savedir), exist_ok=True)
        
        model = self.state_dict()
        _output_normalizer = self._output_normalizer.get_variable()
        _node_normalizer = self._node_normalizer.get_variable()

        to_save = {
            'model': model,
            '_output_normalizer': _output_normalizer,
            '_node_normalizer': _node_normalizer
        }

        torch.save(to_save, savedir)
        print('Simulator model saved at %s' % savedir)
