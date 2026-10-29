import torch.nn as nn
import torch
from torch_geometric.data import Data
from utils import normalization
import os

from .model import EncoderProcesserDecoder
from .gconv_lstm import GConvLSTM  # ⬅️ 引入 GConvLSTM
# from .gconv_lstm import MinLSTMCell
from .gconv_lstm import MinGRUCell
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

        self._output_normalizer = normalization.Normalizer(size=4, name='output_normalizer', device=device)
        self._node_normalizer = normalization.Normalizer(size=5, name='node_normalizer', device=device)

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
        elif use_mingru:  # ✅ 新增 minGRU 分支
            self.rnn_type = 'MinGRU'
            self.rnn = MinGRUCell(
                in_channels=self.lstm_input_size,
                out_channels=self.lstm_hidden_size,
            )
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
            nn.Linear(self.lstm_hidden_size, 4)
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

    def update_node_attr(self, graph, frames: torch.Tensor, types: torch.Tensor):
        # --- Solution features ---
        mach = graph.x[:, 5].unsqueeze(-1)

        graph.valMach = mach.clone().detach()

        sol_feats = torch.cat([frames, mach], dim=-1)

        graph.x = self._node_normalizer(sol_feats, self.training)

        # --- Geometric features ---
        pos = graph.pos

        node_types = types.squeeze().long()
        one_hot = torch.nn.functional.one_hot(node_types, 4).float()

        geo_feats = torch.cat([pos, one_hot], dim=-1)
        graph.x_geometric = geo_feats

        return graph
    
    def add_edge_unit_normal(self, graph: Data, eps: float = 1e-8) -> Data:
        """
        Given graph.edge_attr of shape [E, F] where
        - columns 0..F-2  = relative displacements r_ij
        - column  F-1     = distance d_ij,
        compute u_ij = r_ij / d_ij and
        - append u_ij to edge_attr, making it shape [E, F + (F-1)]
        - store u_ij separately in graph.edge_unit_normal
        """
        edge_attr = graph.edge_attr              # [E, F]
        r = edge_attr[:, :-1]                    # [E, F-1]
        d = edge_attr[:, -1:].clamp(min=eps)     # [E, 1], avoid div0

        # compute unit normals
        edge_unit_normal = r / d                 # [E, F-1]

        # append to edge_attr
        graph.edge_attr = torch.cat([edge_attr, edge_unit_normal], dim=1)

        # also save separately
        graph.edge_unit_normal = edge_unit_normal

        return graph

    def velocity_to_accelation(self, noised_frames, next_velocity):
        return next_velocity - noised_frames

    def forward(self, graph: Data, sequence_noise, hidden=None):
        node_type = graph.x[:, 0]
        frames = graph.x[:, 1:5]

        graph = self.add_edge_unit_normal(graph)

        if self.training:
            target = graph.y                          # [N,4]
            noised_frames = frames + sequence_noise   # teacher forcing 噪声
            graph = self.update_node_attr(graph, noised_frames, node_type)
        else:
            noised_frames = frames
            graph = self.update_node_attr(graph, frames, node_type)

        # ---------------- 空间编码：拿到节点隐向量 gnn_out ----------------
        # 如果你的 EPD 有 encoder/processer_list：
        if hasattr(self.model, "encoder") and hasattr(self.model, "processer_list"):
            g_enc = self.model.encoder(graph)
            for layer in self.model.processer_list:
                g_enc = layer(g_enc)
            gnn_out = g_enc.x                        # [N, gnn_out_dim]
        else:
            # 否则认为 forward 返回 Tensor 或 Data
            out = self.model(graph)
            if hasattr(out, "x"):
                gnn_out = out.x
            else:
                gnn_out = out 

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
            # target_acc_normalized = self._output_normalizer(target_acc, self.training)
            return predicted, target_acc, hidden
        else:
            # velocity_update = self._output_normalizer.inverse(predicted)
            predicted_velocity = frames + predicted
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