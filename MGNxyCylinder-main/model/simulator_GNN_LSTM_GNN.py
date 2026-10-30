import torch.nn as nn 
import torch
from torch_geometric.data import Data
from utils import normalization
import os

from .model import Encoder, GnBlock, Decoder  # ✅ 直接复用原来的子模块
from .gconv_lstm import MinLSTMCell


class Simulator(nn.Module):

    def __init__(self, message_passing_num, node_input_size, edge_input_size, device,
                 model_dir='checkpoint/simulator.pth') -> None:
        super(Simulator, self).__init__()

        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size
        self.model_dir = model_dir
        self.device = device

        # === 原始 Normalizer 保持不变 ===
        self._output_normalizer = normalization.Normalizer(size=2, name='output_normalizer', device=device)
        self._node_normalizer = normalization.Normalizer(size=node_input_size, name='node_normalizer', device=device)

        # === 1️⃣ Encoder ===
        self.encoder = Encoder(
            edge_input_size=edge_input_size,
            node_input_size=node_input_size,
            hidden_size=128
        )

        # === 2️⃣ Processor 前后分段（12层 GNN 拆成两段） ===
        self.gnn_front = nn.ModuleList([GnBlock(hidden_size=128) for _ in range(11)])
        self.gnn_back  = nn.ModuleList([GnBlock(hidden_size=128) for _ in range(message_passing_num - 11)])

        # === 3️⃣ 中间 LSTM（只处理节点特征） ===
        self.mid_lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )
        # self.mid_lstm = MinLSTMCell(
        #     in_channels=128,
        #     out_channels=256
        # )

        # === ✅ 添加 LSTM 输出映射层 ===
        self.post_lstm_proj = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )


        # === 4️⃣ Decoder ===
        self.decoder = Decoder(hidden_size=128, output_size=2)

        # === 5️⃣ 输出 MLP 仍保留（可以选择直接用 decoder）===
        # self.output_head = nn.Sequential(
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 2)
        # ).to(device)

        print('Simulator model initialized (前6层GNN + LSTM + 后6层GNN)')
        self.to(device)

    def update_node_attr(self, frames, types: torch.Tensor):
        node_feature = [frames]
        node_type = torch.squeeze(types.long())
        one_hot = torch.nn.functional.one_hot(node_type, 9)
        node_feature.append(one_hot)
        node_feats = torch.cat(node_feature, dim=1)
        attr = self._node_normalizer(node_feats, self.training)
        return attr.to(self.device)

    def velocity_to_accelation(self, noised_frames, next_velocity):
        return next_velocity - noised_frames

    def forward(self, graph: Data, velocity_sequence_noise=None, hidden=None):
        graph = graph.to(self.device)
        graph.edge_index = graph.edge_index.to(torch.long)
        # === 预处理：节点类型、噪声注入 ===
        node_type = graph.x[:, 0:1]
        frames = graph.x[:, 1:3]

        if self.training:
            target = graph.y
            noised_frames = frames + velocity_sequence_noise
        else:
            noised_frames = frames

        # === 节点特征归一化 ===
        node_attr = self.update_node_attr(noised_frames, node_type)
        graph.x = node_attr

        # === 1️⃣ 编码阶段 ===
        graph = self.encoder(graph)

        # === 2️⃣ 前6层 GNN ===
        for layer in self.gnn_front:
            graph = layer(graph)

        # === 3️⃣ LSTM 时间建模（只更新节点特征） ===
        node_feat = graph.x                      # [N, hidden]
        rnn_input = node_feat.unsqueeze(1)       # [N, 1, H]
        rnn_out, hidden = self.mid_lstm(rnn_input, hidden)
        # graph.x = rnn_out.squeeze(1)             # ✅ 更新节点特征
        # === ✅ LSTM 输出映射（Linear → LayerNorm → ReLU） ===
        graph.x = self.post_lstm_proj(rnn_out.squeeze(1))
        # 注意：edge_attr 不变，继续传递
        # node_feat = graph.x                      # [N, H]
        # H_prev = hidden if hidden is not None else None

        # out = self.mid_lstm(node_feat, H_prev)   # 可能返回 tuple

        # if isinstance(out, (tuple, list)):
        #     H_new = out[0]       # ✅ 取第一个作为节点特征
        # else:
        #     H_new = out          # ✅ 直接就是 Tensor

        # graph.x = H_new          # ✅ graph.x 永远是 Tensor
        # hidden = H_new           # ✅ 下个时间步的 hidden


        # === 4️⃣ 后6层 GNN ===
        for layer in self.gnn_back:
            graph = layer(graph)

        # === 5️⃣ 解码阶段 ===
        predicted_acc = self.decoder(graph)      # 直接用 decoder 输出加速度

        # === 6️⃣ 输出和训练/推理逻辑与原版保持一致 ===
        if self.training:
            target_acc = self.velocity_to_accelation(noised_frames, target)
            target_acc_normalized = self._output_normalizer(target_acc, self.training)
            return predicted_acc, target_acc_normalized, hidden
        else:
            velocity_update = self._output_normalizer.inverse(predicted_acc)
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
                obj = eval('self.' + k)
                setattr(obj, para, value)

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
