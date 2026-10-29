import torch.nn as nn
import torch
from torch_geometric.data import Data
from utils import normalization
import os

from .model import EncoderProcesserDecoder


class Simulator(nn.Module):

    def __init__(self, message_passing_num, node_input_size, edge_input_size, device, use_gru=False, model_dir='checkpoint/simulator.pth') -> None:
        super(Simulator, self).__init__()

        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size
        self.model_dir = model_dir
        self.model = EncoderProcesserDecoder(message_passing_num=message_passing_num, node_input_size=node_input_size, edge_input_size=edge_input_size).to(device)
        self._output_normalizer = normalization.Normalizer(size=4, name='output_normalizer', device=device)
        self._node_normalizer = normalization.Normalizer(size=5, name='node_normalizer', device=device)
        # self._edge_normalizer = normalization.Normalizer(size=edge_input_size, name='edge_normalizer', device=device)

        self.use_gru = use_gru
        self.concat_frames_to_rnn = False   # 如需把 frames 拼到 RNN 输入就设 True
        self.rnn_input_size  = 128
        self.rnn_hidden_size = 256
        if use_gru:
            self.rnn_type = "GRU"
            self.rnn = nn.GRU(input_size=self.rnn_input_size, hidden_size=self.rnn_hidden_size,
                              num_layers=1, batch_first=True)
        else:
            self.rnn_type = "LSTM"
            self.rnn = nn.LSTM(input_size=self.rnn_input_size, hidden_size=self.rnn_hidden_size,
                               num_layers=1, batch_first=True)

        # --- 输出头 ---
        self.output_head = nn.Sequential(
            nn.Linear(self.rnn_hidden_size, self.rnn_hidden_size),
            nn.ReLU(),
            nn.Linear(self.rnn_hidden_size, 4),  # 4 维增量
        )

        self.rnn.to(device)
        self.output_head.to(device)

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

        acc_next = next_velocity - noised_frames
        return acc_next


    def forward(self, graph:Data, sequence_noise, hidden=None):
        
        node_type = graph.x[:, 0]

        # "node_type, rho, v1, v2, pres, Mach"
        frames = graph.x[:, 1:5]

        # compute and add unit normals
        # col = edge_index[1] = i  target (central) nodes
        # pointing from central to neighbor
        graph = self.add_edge_unit_normal(graph)

        # if self.training:
            
        #     target = graph.y

        #     noised_frames = frames + sequence_noise
        #     graph = self.update_node_attr(graph, noised_frames, node_type)

        #     predicted = self.model(graph)

        #     target_acceration = self.velocity_to_accelation(noised_frames, target)
        #     # target_normalized = self._output_normalizer(target_acceration, self.training)

        #     return predicted, target_acceration

        # else:
            
        #     graph = self.update_node_attr(graph, frames, node_type)

        #     predicted = self.model(graph)

        #     # velocity_update = self._output_normalizer.inverse(predicted)
        #     predicted_velocity = frames + predicted

        #     return predicted_velocity

        if self.training:
            target = graph.y                          # [N,4]
            noised_frames = frames + sequence_noise   # teacher forcing 噪声
            graph = self.update_node_attr(graph, noised_frames, node_type)
        else:
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
                gnn_out = out                        # Tensor: [N, gnn_out_dim]

        # d = gnn_out.size(-1)
        # if not hasattr(self, "_printed_gnn_dim"):
        #     print(f"[DEBUG] gnn_out_dim = {d}")
        #     self._printed_gnn_dim = True


        # ---------------- 时间建模：RNN ----------------
        if getattr(self, "concat_frames_to_rnn", False):
            rnn_feat = torch.cat([gnn_out, noised_frames if self.training else frames], dim=-1)
        else:
            rnn_feat = gnn_out

        rnn_in = rnn_feat.unsqueeze(1)               # [N,1,in_dim]
        rnn_out, hidden = self.rnn(rnn_in, hidden)   # LSTM: hidden=(h,c)；GRU: h
        rnn_out = rnn_out.squeeze(1)                 # [N, hidden]

        # ---------------- 输出头：标准化空间的增量/加速度 ----------------
        pred_norm = self.output_head(rnn_out)        # [N,4]

        # if self.training:
        #     target_acc = self.velocity_to_accelation(
        #         noised_frames, target
        #     )                                        # [N,4]
        #     target_norm = self._output_normalizer(target_acc, self.training)
        #     return pred_norm, target_norm, hidden
        # else:
        #     update_phys = self._output_normalizer.inverse(pred_norm)  # [N,4]
        #     predicted_next = frames + update_phys
        #     return predicted_next, hidden

        if self.training:
            target_acc = self.velocity_to_accelation(
                noised_frames, target
            )
            return pred_norm, target_acc, hidden
        else:
            predicted_next = frames + pred_norm
            return predicted_next, hidden


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

        os.makedirs(os.path.dirname(self.model_dir), exist_ok=True)

        model = self.state_dict()
        _output_normalizer = self._output_normalizer.get_variable()
        _node_normalizer  = self._node_normalizer.get_variable()
        # _edge_normalizer = self._edge_normalizer.get_variable()

        to_save = {'model':model, '_output_normalizer':_output_normalizer, '_node_normalizer':_node_normalizer}

        torch.save(to_save, savedir)
        print('Simulator model saved at %s'%savedir)
