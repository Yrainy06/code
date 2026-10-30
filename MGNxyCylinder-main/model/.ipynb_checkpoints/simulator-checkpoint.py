import torch.nn as nn
import torch
from torch_geometric.data import Data
from utils import normalization
import os

from .model import EncoderProcesserDecoder
# from modelEdgeUpd.modelEU import EncoderProcesserDecoder
# from modelGPS.modelGPS import EncoderProcesserDecoder

# 多步
class LSTMPredictor(nn.Module):
    def __init__(self, window_size=1, input_size=2, hidden_size=128, output_size=2, spatial_size=128):
        super().__init__()
        self.window_size = window_size
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.fc_in = nn.Linear(input_size + spatial_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, velocity_window, spatial_features):
        batch_size = velocity_window.size(0)
        spatial_expand = spatial_features.unsqueeze(1).expand(-1, self.window_size, -1)
        x = torch.cat([velocity_window, spatial_expand], dim=-1)
        x = self.fc_in(x)
        out, _ = self.lstm(x)
        pred_velocity = self.fc_out(out[:, -1, :])
        return pred_velocity
    
    def predict_autoregressive(self, init_window, spatial_features, predict_steps):
        """
        自回归预测：给定初始滑动窗口 init_window（形状 [batch_size, window_size, 2]），连续预测 predict_steps 个时刻的速度状态。
        每次预测后更新窗口，且每次不传递隐藏状态。
        返回: Tensor, shape = [batch_size, predict_steps, 2]
        """
        batch_size = init_window.size(0)
        predictions = []
        current_window = init_window.clone()  # 保持 shape: [batch_size, window_size, 2]
        hidden =None

        for _ in range(predict_steps):
            self.lstm.flatten_parameters()
            # self.gru.flatten_parameters()
            spatial_expand = spatial_features.unsqueeze(1).expand(-1, current_window.size(1), -1)
            x = torch.cat([current_window, spatial_expand], dim=-1)
            x = self.fc_in(x.reshape(batch_size * current_window.size(1), -1)).reshape(batch_size, current_window.size(1), -1)
            out, hidden = self.lstm(x, hidden)
            # out, hidden = self.gru(current_window, hidden)
            pred_velocity = self.fc_out(out[:, -1, :])

            predictions.append(pred_velocity.unsqueeze(1))
            current_window = torch.cat([current_window[:, 1:, :], pred_velocity.unsqueeze(1)], dim=1)

        return torch.cat(predictions, dim=1)

class SimulatorWithLSTM(nn.Module):
    def __init__(self, message_passing_num, node_input_size, edge_input_size, device,
                 lstm_window=1, lstm_hidden_size=128, model_dir='checkpoint/simulator.pth'):
        super(SimulatorWithLSTM, self).__init__()
        # 初始化原有 GNN 模型部分
        self.gnn_model = EncoderProcesserDecoder(message_passing_num=message_passing_num,
                                                 node_input_size=node_input_size,
                                                 edge_input_size=edge_input_size,
                                                 hidden_size=128).to(device)
        self._node_normalizer = normalization.Normalizer(size=node_input_size, name='node_normalizer', device=device)
        self._output_normalizer = normalization.Normalizer(size=2, name='output_normalizer', device=device)
        # 新增 LSTM 模块，用于时间序列预测速度状态
        self.lstm_window = lstm_window
        self.lstm_predictor = LSTMPredictor(window_size=lstm_window, input_size=2, hidden_size=lstm_hidden_size, output_size=2).to(device)
        self.device = device
        self.model_dir = model_dir
        print('Simulator with LSTM initialized')

    def update_node_attr(self, frames, types:torch.Tensor):
        node_feature = []

        node_feature.append(frames) #velocity
        node_type = torch.squeeze(types.long())
        # one_hot = torch.nn.functional.one_hot(node_type, 9)
        one_hot = torch.nn.functional.one_hot(node_type, num_classes=9).float()
        node_feature.append(one_hot)
        node_feats = torch.cat(node_feature, dim=1)
        attr = self._node_normalizer(node_feats, self.training)

        return attr

    # 多步
    def forward(self, graph: Data, velocity_window: torch.Tensor = None, predict_steps: int = 1):
        original_target = graph.y if hasattr(graph, 'y') else None
        if original_target is None:
            raise AttributeError("Input graph must have y (ground truth)")

        spatial_features = self.gnn_model(graph)  # shape: [N, hidden]

        if self.training:
            # 多步预测
            pred_velocity_seq = self.lstm_predictor.predict_autoregressive(
                init_window=velocity_window,
                spatial_features=spatial_features,
                predict_steps=predict_steps
            )
            return pred_velocity_seq  # shape: [N, predict_steps, 2]
        else:
            # 推理时反归一化预测值
            pred_seq = self.lstm_predictor.predict_autoregressive(
                init_window=velocity_window,
                spatial_features=spatial_features,
                predict_steps=predict_steps
            )
            pred_seq = self._output_normalizer.inverse(pred_seq)  # 恢复真实速度
            return pred_seq  # shape: [num_nodes, predict_steps, 2]


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

        print(f"Simulator model loaded checkpoint {ckpdir}")

    def save_checkpoint(self, savedir=None):
        if savedir is None:
            savedir = self.model_dir

        os.makedirs(os.path.dirname(savedir), exist_ok=True)

        model = self.state_dict()
        _output_normalizer = self._output_normalizer.get_variable()
        _node_normalizer = self._node_normalizer.get_variable()

        to_save = {'model': model, '_output_normalizer': _output_normalizer, '_node_normalizer': _node_normalizer}

        torch.save(to_save, savedir)
        print(f'Simulator model saved at {savedir}')
   