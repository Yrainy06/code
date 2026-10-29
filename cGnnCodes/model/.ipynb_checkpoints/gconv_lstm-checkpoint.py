# x线性层共享
import torch
from torch import nn
from torch.nn import Parameter
from torch_geometric.nn import ChebConv
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
        # self.conv_h_i = ChebConv(
        #     in_channels=self.out_channels,
        #     out_channels=self.out_channels,
        #     K=self.K,
        #     normalization=self.normalization,
        #     bias=self.bias,
        # )
        self.w_c_i = Parameter(torch.Tensor(1, self.out_channels))
        self.b_i = Parameter(torch.Tensor(1, self.out_channels))

    def _create_forget_gate_parameters_and_layers(self):
        self.lin_h_f = nn.Linear(self.out_channels, self.out_channels, bias=False)
        self.w_c_f = Parameter(torch.Tensor(1, self.out_channels))
        self.b_f = Parameter(torch.Tensor(1, self.out_channels))

    def _create_cell_state_parameters_and_layers(self):
        self.lin_h_c = nn.Linear(self.out_channels, self.out_channels, bias=False)
        # ✅ 替换 lin_h_c 为图卷积 conv_h_c
        # self.conv_h_c = ChebConv(
        #     in_channels=self.out_channels,
        #     out_channels=self.out_channels,
        #     K=self.K,
        #     normalization=self.normalization,
        #     bias=self.bias,
        # )
        self.b_c = Parameter(torch.Tensor(1, self.out_channels))

    def _create_output_gate_parameters_and_layers(self):
        self.lin_h_o = nn.Linear(self.out_channels, self.out_channels, bias=False)
        self.w_c_o = Parameter(torch.Tensor(1, self.out_channels))
        self.b_o = Parameter(torch.Tensor(1, self.out_channels))

    def _create_parameters_and_layers(self):
        # ✅ 新增：共享的 X 线性层
        self.shared_lin_x = nn.Linear(self.in_channels, self.out_channels, bias=False)

        # 原始结构保持不变
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
        # edge_index = edge_index.to(H.device)
        I = shared_x + self.lin_h_i(H) + self.w_c_i * C + self.b_i
        # I = shared_x + self.conv_h_i(H, edge_index) + self.w_c_i * C + self.b_i
        return torch.sigmoid(I)

    def _calculate_forget_gate(self, X, H, C, shared_x):
        F = shared_x + self.lin_h_f(H) + self.w_c_f * C + self.b_f
        return torch.sigmoid(F)

    def _calculate_cell_state(self, X, H, C, I, F, shared_x):
        T = shared_x + self.lin_h_c(H) + self.b_c
        # edge_index = edge_index.to(H.device)  # ✅ 确保 edge_index 在 GPU 上
        # T = shared_x + self.conv_h_c(H, edge_index) + self.b_c
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
        Shared-X Linear LSTM forward pass

        Args:
            X: Input features [num_nodes, in_channels]
            H: Hidden state [num_nodes, out_channels]
            C: Cell state [num_nodes, out_channels]

        Returns:
            H: Updated hidden state
            C: Updated cell state
        """
        X = X.to(next(self.parameters()).device)
        H = self._set_hidden_state(X, H)
        C = self._set_cell_state(X, C)

        shared_x = self.shared_lin_x(X)

        I = self._calculate_input_gate(X, H, C, shared_x)
        F = self._calculate_forget_gate(X, H, C, shared_x)
        C = self._calculate_cell_state(X, H, C, I, F, shared_x)
        # C = self._calculate_cell_state(X, H, C, I, F, shared_x, edge_index)
        O = self._calculate_output_gate(X, H, C, shared_x)
        H = self._calculate_hidden_state(O, C)

        return H, C

# minLSTM
import torch
from torch import nn
from torch.nn import Parameter
from torch_geometric.nn.inits import zeros
from typing import Tuple


class MinLSTMCell(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(MinLSTMCell, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Input gate
        self.lin_x_i = nn.Linear(in_channels, out_channels, bias=True)
        # Forget gate
        self.lin_x_f = nn.Linear(in_channels, out_channels, bias=True)
        # Cell candidate
        self.lin_x_c = nn.Linear(in_channels, out_channels, bias=True)

        # Initialize bias
        self._reset_parameters()

    def _reset_parameters(self):
        zeros(self.lin_x_i.bias)
        zeros(self.lin_x_f.bias)
        zeros(self.lin_x_c.bias)

    def _init_hidden_state(self, X, H):
        if H is None:
            return torch.zeros(X.shape[0], self.out_channels, device=X.device)
        return H

    def forward(
        self,
        X: torch.FloatTensor,
        H: torch.FloatTensor = None,
        C: torch.FloatTensor = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Single-step minLSTM cell.

        Args:
            X: [batch_size, in_channels]
            H: [batch_size, out_channels]  # kept for interface compatibility, unused.
            C: [batch_size, out_channels]

        Returns:
            H: Updated hidden state
            C: same as H (for compatibility)
        """

        X = X.to(next(self.parameters()).device)
        H = self._init_hidden_state(X, H)  # H is not used, kept for compatibility
        C = self._init_hidden_state(X, C)

        # Gate computations (no dependency on H)
        i_t = torch.sigmoid(self.lin_x_i(X))
        f_t = torch.sigmoid(self.lin_x_f(X))
        c_tilde = self.lin_x_c(X)

        # Normalize gates to ensure sum = 1 (optional but recommended)
        gate_sum = i_t + f_t + 1e-6  # prevent division by zero
        i_t = i_t / gate_sum
        f_t = f_t / gate_sum

        # Cell state update (no output gate, no tanh)
        C_new = f_t * C + i_t * c_tilde

        # Hidden state equals to cell state
        H_new = C_new

        return H_new, C_new