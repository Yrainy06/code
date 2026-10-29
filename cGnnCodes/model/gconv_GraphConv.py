# GraphConv
import torch
from torch import nn
from torch.nn import Parameter
from torch_geometric.nn import GraphConv      # ✅ 改成 GraphConv
from torch_geometric.nn.inits import glorot, zeros
from typing import Tuple


class GConvLSTM(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int = 1,
        normalization: str = "sym",
        bias: bool = True,
    ):
        super(GConvLSTM, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalization = normalization
        self.bias = bias
        self._create_parameters_and_layers()
        self._set_parameters()

    def _create_input_gate_parameters_and_layers(self):
        self.lin_h_i = nn.Linear(self.out_channels, self.out_channels, bias=False)
        self.w_c_i = Parameter(torch.Tensor(1, self.out_channels))
        self.b_i = Parameter(torch.Tensor(1, self.out_channels))

    def _create_forget_gate_parameters_and_layers(self):
        self.lin_h_f = nn.Linear(self.out_channels, self.out_channels, bias=False)
        self.w_c_f = Parameter(torch.Tensor(1, self.out_channels))
        self.b_f = Parameter(torch.Tensor(1, self.out_channels))

    def _create_cell_state_parameters_and_layers(self):
        # 原来是 TransformerConv，这里改为 GraphConv 实现：
        # x'_i = W1 x_i + W2 sum_{j∈N(i)} e_{j,i} x_j  (默认 e_{j,i}=1)
        self.graph_conv_h_c = GraphConv(
            self.out_channels, self.out_channels, aggr="add", bias=False
        )
        self.b_c = Parameter(torch.Tensor(1, self.out_channels))

    def _create_output_gate_parameters_and_layers(self):
        self.lin_h_o = nn.Linear(self.out_channels, self.out_channels, bias=False)
        self.w_c_o = Parameter(torch.Tensor(1, self.out_channels))
        self.b_o = Parameter(torch.Tensor(1, self.out_channels))

    def _create_parameters_and_layers(self):
        # ✅ 共享的 X 线性层（保持不变）
        self.shared_lin_x = nn.Linear(self.in_channels, self.out_channels, bias=False)

        self._create_input_gate_parameters_and_layers()
        self._create_forget_gate_parameters_and_layers()
        self._create_cell_state_parameters_and_layers()
        self._create_output_gate_parameters_and_layers()

    def _set_parameters(self):
        glorot(self.w_c_i)
        glorot(self.w_c_f)
        glorot(self.w_c_o)
        zeros(self.b_i)
        zeros(self.b_f)
        zeros(self.b_c)
        zeros(self.b_o)

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels, device=X.device)
        return H

    def _set_cell_state(self, X, C):
        if C is None:
            C = torch.zeros(X.shape[0], self.out_channels, device=X.device)
        return C

    def _calculate_input_gate(self, X, H, C, shared_x):
        I = shared_x + self.lin_h_i(H) + self.w_c_i * C + self.b_i
        return torch.sigmoid(I)

    def _calculate_forget_gate(self, X, H, C, shared_x):
        F = shared_x + self.lin_h_f(H) + self.w_c_f * C + self.b_f
        return torch.sigmoid(F)

    def _calculate_cell_state(self, X, H, C, I, F, shared_x, edge_index):
        # 使用 GraphConv 对 H 进行图卷积：W1 H_i + W2 ∑_j e_{j,i} H_j
        H_gnn = self.graph_conv_h_c(H, edge_index)   # 默认 e_{j,i}=1
        T = shared_x + H_gnn + self.b_c
        T = torch.tanh(T)
        C = F * C + I * T
        return C

    def _calculate_output_gate(self, X, H, C, shared_x):
        O = shared_x + self.lin_h_o(H) + self.w_c_o * C + self.b_o
        return torch.sigmoid(O)

    def _calculate_hidden_state(self, O, C):
        return O * torch.tanh(C)

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        H: torch.FloatTensor = None,
        C: torch.FloatTensor = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Shared-X Linear + GraphConv LSTM forward pass

        Args:
            X: Input features [num_nodes, in_channels]
            edge_index: [2, num_edges]
            H: Hidden state [num_nodes, out_channels]
            C: Cell state [num_nodes, out_channels]

        Returns:
            H: Updated hidden state
            C: Updated cell state
        """
        X = X.to(next(self.parameters()).device)
        edge_index = edge_index.to(next(self.parameters()).device)
        H = self._set_hidden_state(X, H)
        C = self._set_cell_state(X, C)

        shared_x = self.shared_lin_x(X)

        I = self._calculate_input_gate(X, H, C, shared_x)
        F = self._calculate_forget_gate(X, H, C, shared_x)
        C = self._calculate_cell_state(X, H, C, I, F, shared_x, edge_index)
        O = self._calculate_output_gate(X, H, C, shared_x)
        H = self._calculate_hidden_state(O, C)

        return H, C
