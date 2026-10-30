from typing import Tuple

import torch
from torch.nn import Parameter
from torch_geometric.nn import ChebConv
from torch_geometric.nn.inits import glorot, zeros


# class GConvLSTM(torch.nn.Module):
#     r"""An implementation of the Chebyshev Graph Convolutional Long Short Term Memory
#     Cell. For details see this paper: `"Structured Sequence Modeling with Graph
#     Convolutional Recurrent Networks." <https://arxiv.org/abs/1612.07659>`_

#     Args:
#         in_channels (int): Number of input features.
#         out_channels (int): Number of output features.
#         K (int): Chebyshev filter size :math:`K`.
#         normalization (str, optional): The normalization scheme for the graph
#             Laplacian (default: :obj:`"sym"`):

#             1. :obj:`None`: No normalization
#             :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

#             2. :obj:`"sym"`: Symmetric normalization
#             :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
#             \mathbf{D}^{-1/2}`

#             3. :obj:`"rw"`: Random-walk normalization
#             :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

#             You need to pass :obj:`lambda_max` to the :meth:`forward` method of
#             this operator in case the normalization is non-symmetric.
#             :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
#             :obj:`[num_graphs]` in a mini-batch scenario and a
#             scalar/zero-dimensional tensor when operating on single graphs.
#             You can pre-compute :obj:`lambda_max` via the
#             :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
#         bias (bool, optional): If set to :obj:`False`, the layer will not learn
#             an additive bias. (default: :obj:`True`)
#     """

#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         K: int,
#         normalization: str = "sym",
#         bias: bool = True,
#     ):
#         super(GConvLSTM, self).__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.K = K
#         self.normalization = normalization
#         self.bias = bias
#         self._create_parameters_and_layers()
#         self._set_parameters()

#     def _create_input_gate_parameters_and_layers(self):

#         self.conv_x_i = ChebConv(
#             in_channels=self.in_channels,
#             out_channels=self.out_channels,
#             K=self.K,
#             normalization=self.normalization,
#             bias=self.bias,
#         )

#         self.conv_h_i = ChebConv(
#             in_channels=self.out_channels,
#             out_channels=self.out_channels,
#             K=self.K,
#             normalization=self.normalization,
#             bias=self.bias,
#         )

#         self.w_c_i = Parameter(torch.Tensor(1, self.out_channels))
#         self.b_i = Parameter(torch.Tensor(1, self.out_channels))

#     def _create_forget_gate_parameters_and_layers(self):

#         self.conv_x_f = ChebConv(
#             in_channels=self.in_channels,
#             out_channels=self.out_channels,
#             K=self.K,
#             normalization=self.normalization,
#             bias=self.bias,
#         )

#         self.conv_h_f = ChebConv(
#             in_channels=self.out_channels,
#             out_channels=self.out_channels,
#             K=self.K,
#             normalization=self.normalization,
#             bias=self.bias,
#         )

#         self.w_c_f = Parameter(torch.Tensor(1, self.out_channels))
#         self.b_f = Parameter(torch.Tensor(1, self.out_channels))

#     def _create_cell_state_parameters_and_layers(self):

#         self.conv_x_c = ChebConv(
#             in_channels=self.in_channels,
#             out_channels=self.out_channels,
#             K=self.K,
#             normalization=self.normalization,
#             bias=self.bias,
#         )

#         self.conv_h_c = ChebConv(
#             in_channels=self.out_channels,
#             out_channels=self.out_channels,
#             K=self.K,
#             normalization=self.normalization,
#             bias=self.bias,
#         )

#         self.b_c = Parameter(torch.Tensor(1, self.out_channels))

#     def _create_output_gate_parameters_and_layers(self):

#         self.conv_x_o = ChebConv(
#             in_channels=self.in_channels,
#             out_channels=self.out_channels,
#             K=self.K,
#             normalization=self.normalization,
#             bias=self.bias,
#         )

#         self.conv_h_o = ChebConv(
#             in_channels=self.out_channels,
#             out_channels=self.out_channels,
#             K=self.K,
#             normalization=self.normalization,
#             bias=self.bias,
#         )

#         self.w_c_o = Parameter(torch.Tensor(1, self.out_channels))
#         self.b_o = Parameter(torch.Tensor(1, self.out_channels))

#     def _create_parameters_and_layers(self):
#         self._create_input_gate_parameters_and_layers()
#         self._create_forget_gate_parameters_and_layers()
#         self._create_cell_state_parameters_and_layers()
#         self._create_output_gate_parameters_and_layers()

#     def _set_parameters(self):
#         glorot(self.w_c_i)
#         glorot(self.w_c_f)
#         glorot(self.w_c_o)
#         zeros(self.b_i)
#         zeros(self.b_f)
#         zeros(self.b_c)
#         zeros(self.b_o)

#     def _set_hidden_state(self, X, H):
#         if H is None:
#             H = torch.zeros(X.shape[0], self.out_channels)
#         return H

#     def _set_cell_state(self, X, C):
#         if C is None:
#             C = torch.zeros(X.shape[0], self.out_channels)
#         return C

#     def _calculate_input_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
#         I = self.conv_x_i(X, edge_index, edge_weight, lambda_max=lambda_max)
#         I = I + self.conv_h_i(H, edge_index, edge_weight, lambda_max=lambda_max)
#         I = I + (self.w_c_i * C)
#         I = I + self.b_i
#         I = torch.sigmoid(I)
#         return I

#     def _calculate_forget_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
#         F = self.conv_x_f(X, edge_index, edge_weight, lambda_max=lambda_max)
#         F = F + self.conv_h_f(H, edge_index, edge_weight, lambda_max=lambda_max)
#         F = F + (self.w_c_f * C)
#         F = F + self.b_f
#         F = torch.sigmoid(F)
#         return F

#     def _calculate_cell_state(self, X, edge_index, edge_weight, H, C, I, F, lambda_max):
#         T = self.conv_x_c(X, edge_index, edge_weight, lambda_max=lambda_max)
#         T = T + self.conv_h_c(H, edge_index, edge_weight, lambda_max=lambda_max)
#         T = T + self.b_c
#         T = torch.tanh(T)
#         C = F * C + I * T
#         return C

#     def _calculate_output_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
#         O = self.conv_x_o(X, edge_index, edge_weight, lambda_max=lambda_max)
#         O = O + self.conv_h_o(H, edge_index, edge_weight, lambda_max=lambda_max)
#         O = O + (self.w_c_o * C)
#         O = O + self.b_o
#         O = torch.sigmoid(O)
#         return O

#     def _calculate_hidden_state(self, O, C):
#         H = O * torch.tanh(C)
#         return H

#     def forward(
#         self,
#         X: torch.FloatTensor,
#         edge_index: torch.LongTensor,
#         edge_weight: torch.FloatTensor = None,
#         H: torch.FloatTensor = None,
#         C: torch.FloatTensor = None,
#         lambda_max: torch.Tensor = None,
#     ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
#         """
#         Making a forward pass. If edge weights are not present the forward pass
#         defaults to an unweighted graph. If the hidden state and cell state
#         matrices are not present when the forward pass is called these are
#         initialized with zeros.

#         Arg types:
#             * **X** *(PyTorch Float Tensor)* - Node features.
#             * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
#             * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
#             * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
#             * **C** *(PyTorch Float Tensor, optional)* - Cell state matrix for all nodes.
#             * **lambda_max** *(PyTorch Tensor, optional but mandatory if normalization is not sym)* - Largest eigenvalue of Laplacian.

#         Return types:
#             * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
#             * **C** *(PyTorch Float Tensor)* - Cell state matrix for all nodes.
#         """
#         X = X.to(next(self.parameters()).device)  # 保证和模型本身参数在同一设备
#         edge_index = edge_index.to(X.device)
#         if edge_weight is not None:
#             edge_weight = edge_weight.to(X.device)
#         if lambda_max is not None:
#             lambda_max = lambda_max.to(X.device)

#         H = self._set_hidden_state(X, H)
#         C = self._set_cell_state(X, C)
#         I = self._calculate_input_gate(X, edge_index, edge_weight, H, C, lambda_max)
#         F = self._calculate_forget_gate(X, edge_index, edge_weight, H, C, lambda_max)
#         C = self._calculate_cell_state(X, edge_index, edge_weight, H, C, I, F, lambda_max)
#         O = self._calculate_output_gate(X, edge_index, edge_weight, H, C, lambda_max)
#         H = self._calculate_hidden_state(O, C)
#         return H, C


# 线性LSTM验证
# import torch
# from torch import nn
# from torch.nn import Parameter
# from torch_geometric.nn.inits import glorot, zeros
# from typing import Tuple


# class GConvLSTM(torch.nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         K: int = 1,  # 保留参数以兼容外部调用，不使用
#         normalization: str = "sym",  # 同上
#         bias: bool = True,
#     ):
#         super(GConvLSTM, self).__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.K = K
#         self.normalization = normalization
#         self.bias = bias
#         self._create_parameters_and_layers()
#         self._set_parameters()

#     def _create_input_gate_parameters_and_layers(self):
#         self.lin_x_i = nn.Linear(self.in_channels, self.out_channels, bias=False)
#         self.lin_h_i = nn.Linear(self.out_channels, self.out_channels, bias=False)
#         self.w_c_i = Parameter(torch.Tensor(1, self.out_channels))
#         self.b_i = Parameter(torch.Tensor(1, self.out_channels))

#     def _create_forget_gate_parameters_and_layers(self):
#         self.lin_x_f = nn.Linear(self.in_channels, self.out_channels, bias=False)
#         self.lin_h_f = nn.Linear(self.out_channels, self.out_channels, bias=False)
#         self.w_c_f = Parameter(torch.Tensor(1, self.out_channels))
#         self.b_f = Parameter(torch.Tensor(1, self.out_channels))

#     def _create_cell_state_parameters_and_layers(self):
#         self.lin_x_c = nn.Linear(self.in_channels, self.out_channels, bias=False)
#         self.lin_h_c = nn.Linear(self.out_channels, self.out_channels, bias=False)
#         self.b_c = Parameter(torch.Tensor(1, self.out_channels))

#     def _create_output_gate_parameters_and_layers(self):
#         self.lin_x_o = nn.Linear(self.in_channels, self.out_channels, bias=False)
#         self.lin_h_o = nn.Linear(self.out_channels, self.out_channels, bias=False)
#         self.w_c_o = Parameter(torch.Tensor(1, self.out_channels))
#         self.b_o = Parameter(torch.Tensor(1, self.out_channels))

#     def _create_parameters_and_layers(self):
#         self._create_input_gate_parameters_and_layers()
#         self._create_forget_gate_parameters_and_layers()
#         self._create_cell_state_parameters_and_layers()
#         self._create_output_gate_parameters_and_layers()

#     def _set_parameters(self):
#         glorot(self.w_c_i)
#         glorot(self.w_c_f)
#         glorot(self.w_c_o)
#         zeros(self.b_i)
#         zeros(self.b_f)
#         zeros(self.b_c)
#         zeros(self.b_o)

#     def _set_hidden_state(self, X, H):
#         if H is None:
#             H = torch.zeros(X.shape[0], self.out_channels, device=X.device)
#         return H

#     def _set_cell_state(self, X, C):
#         if C is None:
#             C = torch.zeros(X.shape[0], self.out_channels, device=X.device)
#         return C

#     def _calculate_input_gate(self, X, H, C):
#         # I = self.lin_x_i(X) + self.lin_h_i(H) + self.w_c_i * C + self.b_i
#         I = X + self.lin_h_i(H) + self.w_c_i * C + self.b_i
#         return torch.sigmoid(I)

#     def _calculate_forget_gate(self, X, H, C):
#         # F = self.lin_x_f(X) + self.lin_h_f(H) + self.w_c_f * C + self.b_f
#         F = X + self.lin_h_f(H) + self.w_c_f * C + self.b_f
#         return torch.sigmoid(F)

#     def _calculate_cell_state(self, X, H, C, I, F):
#         T = self.lin_x_c(X) + self.lin_h_c(H) + self.b_c
#         T = torch.tanh(T)
#         C = F * C + I * T
#         return C

#     def _calculate_output_gate(self, X, H, C):
#         # O = self.lin_x_o(X) + self.lin_h_o(H) + self.w_c_o * C + self.b_o
#         O = X + self.lin_h_o(H) + self.w_c_o * C + self.b_o
#         return torch.sigmoid(O)

#     def _calculate_hidden_state(self, O, C):
#         return O * torch.tanh(C)

#     def forward(
#         self,
#         X: torch.FloatTensor,
#         H: torch.FloatTensor = None,
#         C: torch.FloatTensor = None,
#     ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
#         """
#         Standard LSTM forward pass using linear layers (no graph structure).

#         Args:
#             X: Input features [num_nodes, in_channels]
#             H: Hidden state [num_nodes, out_channels]
#             C: Cell state [num_nodes, out_channels]

#         Returns:
#             H: Updated hidden state
#             C: Updated cell state
#         """
#         X = X.to(next(self.parameters()).device)
#         H = self._set_hidden_state(X, H)
#         C = self._set_cell_state(X, C)
#         I = self._calculate_input_gate(X, H, C)
#         F = self._calculate_forget_gate(X, H, C)
#         C = self._calculate_cell_state(X, H, C, I, F)
#         O = self._calculate_output_gate(X, H, C)
#         H = self._calculate_hidden_state(O, C)
#         return H, C



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


# # GATv2Conv
# import torch
# from torch import nn
# from torch.nn import Parameter
# # from torch_geometric.nn import ChebConv
# from torch_geometric.nn import GATv2Conv
# from torch_geometric.nn.inits import glorot, zeros
# from typing import Tuple

# class GConvLSTM(torch.nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         K: int = 1,
#         normalization: str = "sym",
#         bias: bool = True,
#     ):
#         super(GConvLSTM, self).__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.K = K
#         self.normalization = normalization
#         self.bias = bias
#         self._create_parameters_and_layers()
#         self._set_parameters()

#     def _create_input_gate_parameters_and_layers(self):
#         self.lin_h_i = nn.Linear(self.out_channels, self.out_channels, bias=False)
#         self.w_c_i = Parameter(torch.Tensor(1, self.out_channels))
#         self.b_i = Parameter(torch.Tensor(1, self.out_channels))

#     def _create_forget_gate_parameters_and_layers(self):
#         self.lin_h_f = nn.Linear(self.out_channels, self.out_channels, bias=False)
#         self.w_c_f = Parameter(torch.Tensor(1, self.out_channels))
#         self.b_f = Parameter(torch.Tensor(1, self.out_channels))

#     def _create_cell_state_parameters_and_layers(self):
#         # self.lin_h_c = nn.Linear(self.out_channels, self.out_channels, bias=False)
#         # 将 Linear 替换为 GATv2Conv
#         self.gat_h_c = GATv2Conv(self.out_channels, self.out_channels, heads=1, concat=False)
#         self.b_c = Parameter(torch.Tensor(1, self.out_channels))

#     def _create_output_gate_parameters_and_layers(self):
#         self.lin_h_o = nn.Linear(self.out_channels, self.out_channels, bias=False)
#         self.w_c_o = Parameter(torch.Tensor(1, self.out_channels))
#         self.b_o = Parameter(torch.Tensor(1, self.out_channels))

#     def _create_parameters_and_layers(self):
#         # 新增：共享的 X 线性层
#         self.shared_lin_x = nn.Linear(self.in_channels, self.out_channels, bias=False)

#         # 原始结构保持不变
#         self._create_input_gate_parameters_and_layers()
#         self._create_forget_gate_parameters_and_layers()
#         self._create_cell_state_parameters_and_layers()
#         self._create_output_gate_parameters_and_layers()

#     def _set_parameters(self):
#         glorot(self.w_c_i)
#         glorot(self.w_c_f)
#         glorot(self.w_c_o)
#         zeros(self.b_i)
#         zeros(self.b_f)
#         zeros(self.b_c)
#         zeros(self.b_o)

#     def _set_hidden_state(self, X, H):
#         if H is None:
#             H = torch.zeros(X.shape[0], self.out_channels, device=X.device)
#         return H

#     def _set_cell_state(self, X, C):
#         if C is None:
#             C = torch.zeros(X.shape[0], self.out_channels, device=X.device)
#         return C

#     def _calculate_input_gate(self, X, H, C, shared_x):
#         I = shared_x + self.lin_h_i(H) + self.w_c_i * C + self.b_i
#         return torch.sigmoid(I)

#     def _calculate_forget_gate(self, X, H, C, shared_x):
#         F = shared_x + self.lin_h_f(H) + self.w_c_f * C + self.b_f
#         return torch.sigmoid(F)

#     def _calculate_cell_state(self, X, H, C, I, F, shared_x, edge_index):
#         # T = shared_x + self.lin_h_c(H) + self.b_c
#         # 用 GATv2Conv 处理 H
#         T = shared_x + self.gat_h_c(H, edge_index) + self.b_c
#         T = torch.tanh(T)
#         C = F * C + I * T
#         return C

#     def _calculate_output_gate(self, X, H, C, shared_x):
#         O = shared_x + self.lin_h_o(H) + self.w_c_o * C + self.b_o
#         return torch.sigmoid(O)

#     def _calculate_hidden_state(self, O, C):
#         return O * torch.tanh(C)

#     def forward(
#         self,
#         X: torch.FloatTensor,
#         edge_index: torch.LongTensor,
#         H: torch.FloatTensor = None,
#         C: torch.FloatTensor = None,
#     ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
#         """
#         Shared-X Linear LSTM forward pass

#         Args:
#             X: Input features [num_nodes, in_channels]
#             H: Hidden state [num_nodes, out_channels]
#             C: Cell state [num_nodes, out_channels]

#         Returns:
#             H: Updated hidden state
#             C: Updated cell state
#         """
#         X = X.to(next(self.parameters()).device)
#         edge_index = edge_index.to(next(self.parameters()).device)
#         H = self._set_hidden_state(X, H)
#         C = self._set_cell_state(X, C)

#         shared_x = self.shared_lin_x(X)

#         I = self._calculate_input_gate(X, H, C, shared_x)
#         F = self._calculate_forget_gate(X, H, C, shared_x)
#         # C = self._calculate_cell_state(X, H, C, I, F, shared_x)
#         # 传入 edge_index 给 cell_state
#         C = self._calculate_cell_state(X, H, C, I, F, shared_x, edge_index)
#         O = self._calculate_output_gate(X, H, C, shared_x)
#         H = self._calculate_hidden_state(O, C)

#         return H, C



# TransformerConv
# import torch
# from torch import nn
# from torch.nn import Parameter
# # from torch_geometric.nn import ChebConv
# from torch_geometric.nn import TransformerConv
# from torch_geometric.nn.inits import glorot, zeros
# from typing import Tuple

# class GConvLSTM(torch.nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         K: int = 1,
#         normalization: str = "sym",
#         bias: bool = True,
#     ):
#         super(GConvLSTM, self).__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.K = K
#         self.normalization = normalization
#         self.bias = bias
#         self._create_parameters_and_layers()
#         self._set_parameters()

#     def _create_input_gate_parameters_and_layers(self):
#         self.lin_h_i = nn.Linear(self.out_channels, self.out_channels, bias=False)
#         # self.trans_conv_h_i = TransformerConv(self.out_channels, self.out_channels, heads=1, concat=False)
#         self.w_c_i = Parameter(torch.Tensor(1, self.out_channels))
#         self.b_i = Parameter(torch.Tensor(1, self.out_channels))

#     def _create_forget_gate_parameters_and_layers(self):
#         self.lin_h_f = nn.Linear(self.out_channels, self.out_channels, bias=False)
#         self.w_c_f = Parameter(torch.Tensor(1, self.out_channels))
#         self.b_f = Parameter(torch.Tensor(1, self.out_channels))

#     def _create_cell_state_parameters_and_layers(self):
#         # self.lin_h_c = nn.Linear(self.out_channels, self.out_channels, bias=False)
#         # 将 Linear 替换为 TransformerConv
#         self.trans_conv_h_c = TransformerConv(self.out_channels, self.out_channels, heads=1, concat=False)
#         self.b_c = Parameter(torch.Tensor(1, self.out_channels))

#     def _create_output_gate_parameters_and_layers(self):
#         self.lin_h_o = nn.Linear(self.out_channels, self.out_channels, bias=False)
#         self.w_c_o = Parameter(torch.Tensor(1, self.out_channels))
#         self.b_o = Parameter(torch.Tensor(1, self.out_channels))

#     def _create_parameters_and_layers(self):
#         # ✅ 新增：共享的 X 线性层
#         self.shared_lin_x = nn.Linear(self.in_channels, self.out_channels, bias=False)

#         # 原始结构保持不变
#         self._create_input_gate_parameters_and_layers()
#         self._create_forget_gate_parameters_and_layers()
#         self._create_cell_state_parameters_and_layers()
#         self._create_output_gate_parameters_and_layers()

#     def _set_parameters(self):
#         glorot(self.w_c_i)
#         glorot(self.w_c_f)
#         glorot(self.w_c_o)
#         zeros(self.b_i)
#         zeros(self.b_f)
#         zeros(self.b_c)
#         zeros(self.b_o)

#     def _set_hidden_state(self, X, H):
#         if H is None:
#             H = torch.zeros(X.shape[0], self.out_channels, device=X.device)
#         return H

#     def _set_cell_state(self, X, C):
#         if C is None:
#             C = torch.zeros(X.shape[0], self.out_channels, device=X.device)
#         return C

#     def _calculate_input_gate(self, X, H, C, shared_x):
#         I = shared_x + self.lin_h_i(H) + self.w_c_i * C + self.b_i
#         return torch.sigmoid(I)

#     def _calculate_forget_gate(self, X, H, C, shared_x):
#         F = shared_x + self.lin_h_f(H) + self.w_c_f * C + self.b_f
#         return torch.sigmoid(F)

#     def _calculate_cell_state(self, X, H, C, I, F, shared_x, edge_index):
#         # T = shared_x + self.lin_h_c(H) + self.b_c
#         # 用TransformerConv 处理 H
#         T = shared_x + self.trans_conv_h_c(H, edge_index) + self.b_c
#         T = torch.tanh(T)
#         C = F * C + I * T
#         return C

#     def _calculate_output_gate(self, X, H, C, shared_x):
#         O = shared_x + self.lin_h_o(H) + self.w_c_o * C + self.b_o
#         return torch.sigmoid(O)

#     def _calculate_hidden_state(self, O, C):
#         return O * torch.tanh(C)

#     def forward(
#         self,
#         X: torch.FloatTensor,
#         edge_index: torch.LongTensor,
#         H: torch.FloatTensor = None,
#         C: torch.FloatTensor = None,
#     ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
#         """
#         Shared-X Linear LSTM forward pass

#         Args:
#             X: Input features [num_nodes, in_channels]
#             H: Hidden state [num_nodes, out_channels]
#             C: Cell state [num_nodes, out_channels]

#         Returns:
#             H: Updated hidden state
#             C: Updated cell state
#         """
#         X = X.to(next(self.parameters()).device)
#         edge_index = edge_index.to(next(self.parameters()).device)
#         H = self._set_hidden_state(X, H)
#         C = self._set_cell_state(X, C)

#         shared_x = self.shared_lin_x(X)

#         I = self._calculate_input_gate(X, H, C, shared_x)
#         F = self._calculate_forget_gate(X, H, C, shared_x)
#         # C = self._calculate_cell_state(X, H, C, I, F, shared_x)
#         # 传入 edge_index 给 cell_state
#         C = self._calculate_cell_state(X, H, C, I, F, shared_x, edge_index)
#         O = self._calculate_output_gate(X, H, C, shared_x)
#         H = self._calculate_hidden_state(O, C)

#         return H, C


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

# import torch
# from torch import nn
# from torch.nn import Parameter
# from torch_geometric.nn.inits import zeros
# from typing import Tuple


# class MinLSTMCell(torch.nn.Module):
#     def __init__(self, in_channels: int, out_channels: int):
#         super(MinLSTMCell, self).__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         # Input gate
#         self.lin_x_i = nn.Linear(in_channels+ out_channels, out_channels, bias=True)
#         # Forget gate
#         self.lin_x_f = nn.Linear(in_channels+ out_channels, out_channels, bias=True)
#         # Cell candidate
#         self.lin_x_c = nn.Linear(in_channels+ out_channels, out_channels, bias=True)

#         # Initialize bias
#         self._reset_parameters()

#     def _reset_parameters(self):
#         zeros(self.lin_x_i.bias)
#         zeros(self.lin_x_f.bias)
#         zeros(self.lin_x_c.bias)

#     def _init_hidden_state(self, X, H):
#         if H is None:
#             return torch.zeros(X.shape[0], self.out_channels, device=X.device)
#         return H

#     def forward(
#         self,
#         X: torch.FloatTensor,
#         H: torch.FloatTensor = None,
#         C: torch.FloatTensor = None,
#     ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
#         """
#         Single-step minLSTM cell.

#         Args:
#             X: [batch_size, in_channels]
#             H: [batch_size, out_channels]  # kept for interface compatibility, unused.
#             C: [batch_size, out_channels]

#         Returns:
#             H: Updated hidden state
#             C: same as H (for compatibility)
#         """

#         X = X.to(next(self.parameters()).device)
#         H = self._init_hidden_state(X, H)  # H is not used, kept for compatibility
#         C = self._init_hidden_state(X, C)

#         gate_input = torch.cat([X, H], dim=-1)

#         # Gate computations (no dependency on H)
#         i_t = torch.sigmoid(self.lin_x_i(gate_input))
#         f_t = torch.sigmoid(self.lin_x_f(gate_input))
#         c_tilde = self.lin_x_c(gate_input)

#         # Normalize gates to ensure sum = 1 (optional but recommended)
#         gate_sum = i_t + f_t + 1e-6  # prevent division by zero
#         i_t = i_t / gate_sum
#         f_t = f_t / gate_sum

#         # Cell state update (no output gate, no tanh)
#         H_new = f_t * C + i_t * c_tilde

#         # Hidden state equals to cell state
#         C_new = H_new

#         return H_new, C_new


# minGRU
# import torch
# from torch import nn
# from torch.nn import Parameter
# from torch_geometric.nn.inits import zeros
# from typing import Tuple


# class MinGRUCell(torch.nn.Module):
#     def __init__(self, in_channels: int, out_channels: int):
#         super(MinGRUCell, self).__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         # Update gate z_t
#         self.lin_x_z = nn.Linear(in_channels, out_channels, bias=True)
#         # Candidate hidden state
#         self.lin_x_h = nn.Linear(in_channels, out_channels, bias=True)

#         self._reset_parameters()

#     def _reset_parameters(self):
#         zeros(self.lin_x_z.bias)
#         zeros(self.lin_x_h.bias)

#     def _init_hidden_state(self, X, H):
#         if H is None:
#             return torch.zeros(X.shape[0], self.out_channels, device=X.device)
#         return H

#     def forward(
#         self,
#         X: torch.FloatTensor,
#         H: torch.FloatTensor = None,
#     ) -> torch.FloatTensor:
        
#         """
#         Single-step minGRU cell.

#         Args:
#             X: [batch_size, in_channels]
#             H: [batch_size, out_channels]

#         Returns:
#             H_new: Updated hidden state
#         """
#         X = X.to(next(self.parameters()).device)
#         H = self._init_hidden_state(X, H)

#         z_t = torch.sigmoid(self.lin_x_z(X))
#         h_tilde = self.lin_x_h(X)

#         H_new = (1 - z_t) * H + z_t * h_tilde
#         return H_new

# GATv2
# # 文件位置: gconv_lstm.py 或单独新文件
# from torch_geometric.nn import GATv2Conv
# import torch
# from torch import nn
# from typing import Tuple

# class MinLSTMCellGAT(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int):
#         super(MinLSTMCellGAT, self).__init__()
#         self.out_channels = out_channels

#         self.gat_i = GATv2Conv(in_channels, out_channels, heads=1, concat=False)
#         self.gat_f = GATv2Conv(in_channels, out_channels, heads=1, concat=False)
#         self.gat_c = GATv2Conv(in_channels, out_channels, heads=1, concat=False)

#     def _init_state(self, x, state):
#         if state is None:
#             return torch.zeros(x.size(0), self.out_channels, device=x.device)
#         return state

#     def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
#                 h: torch.Tensor = None, c: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:

#         h = self._init_state(x, h)
#         c = self._init_state(x, c)

#         i_t = torch.sigmoid(self.gat_i(x, edge_index))
#         f_t = torch.sigmoid(self.gat_f(x, edge_index))
#         c_tilde = self.gat_c(x, edge_index)

#         # normalize gates
#         gate_sum = f_t + i_t + 1e-6
#         i_t = i_t / gate_sum
#         f_t = f_t / gate_sum

#         c_new = f_t * c + i_t * c_tilde
#         h_new = c_new

#         return h_new, c_new
# import torch
# from torch import nn
# from torch_geometric.nn import GATv2Conv

# class GConvMinLSTM(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, bias=True):
#         super(GConvMinLSTM, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         # forget / input gate 各自独立线性映射
#         self.linear_f = nn.Linear(in_channels, out_channels, bias=bias)
#         self.linear_i = nn.Linear(in_channels, out_channels, bias=bias)

#         # 候选隐藏状态用 GATv2Conv 保留图结构信息
#         self.gat_h_tilde = GATv2Conv(in_channels, out_channels, heads=1, concat=False)

#     def _init_hidden(self, X, H):
#         """初始化隐藏状态 H"""
#         if H is None:
#             H = torch.zeros(X.shape[0], self.out_channels, device=X.device)
#         return H

#     def forward(
#         self,
#         X: torch.FloatTensor,
#         edge_index: torch.LongTensor,
#         H: torch.FloatTensor = None,
#     ):
#         """
#         GConvMinLSTM forward pass（无共享线性层）

#         Args:
#             X: 输入特征 [num_nodes, in_channels]
#             edge_index: 图边索引
#             H: 上一隐藏状态 [num_nodes, out_channels] (可选)

#         Returns:
#             H_new: 更新后的隐藏状态
#         """
#         device = next(self.parameters()).device
#         X = X.to(device)
#         edge_index = edge_index.to(device)

#         # 初始化 H
#         H = self._init_hidden(X, H)

#         # 1. forget/input gate 只依赖 X
#         f_t = torch.sigmoid(self.linear_f(X))
#         i_t = torch.sigmoid(self.linear_i(X))

#         # 可选：归一化使 f_t + i_t = 1，避免长期累积漂移
#         sum_ft_it = f_t + i_t + 1e-8
#         f_t = f_t / sum_ft_it
#         i_t = i_t / sum_ft_it

#         # 2. 候选隐藏状态 h_tilde：用 GATv2Conv 融合邻居信息
#         h_tilde = self.gat_h_tilde(X, edge_index)

#         # 3. minLSTM 更新公式
#         H_new = f_t * H + i_t * h_tilde

#         return H_new
# 超显存
# import torch
# from torch import nn
# from torch_geometric.nn import GATv2Conv

# class GConvMinLSTM(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, bias=True):
#         super(GConvMinLSTM, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         # 1️⃣ forget gate 用 GATv2Conv
#         self.gat_f = GATv2Conv(in_channels, out_channels, heads=1, concat=False, bias=bias)

#         # 2️⃣ input gate 用 GATv2Conv
#         self.gat_i = GATv2Conv(in_channels, out_channels, heads=1, concat=False, bias=bias)

#         # 3️⃣ 候选隐藏状态 h_tilde 依旧是 GATv2Conv
#         self.gat_h_tilde = GATv2Conv(in_channels, out_channels, heads=1, concat=False, bias=bias)

#     def _init_hidden(self, X, H):
#         """初始化隐藏状态 H"""
#         if H is None:
#             H = torch.zeros(X.shape[0], self.out_channels, device=X.device)
#         return H

#     def forward(
#         self,
#         X: torch.FloatTensor,
#         edge_index: torch.LongTensor,
#         H: torch.FloatTensor = None,
#     ):
#         """
#         GConvMinLSTM forward pass（所有门控都用 GATv2Conv）

#         Args:
#             X: 输入特征 [num_nodes, in_channels]
#             edge_index: 图边索引
#             H: 上一隐藏状态 [num_nodes, out_channels] (可选)

#         Returns:
#             H_new: 更新后的隐藏状态
#         """
#         device = next(self.parameters()).device
#         X = X.to(device)
#         edge_index = edge_index.to(device)

#         # 初始化 H
#         H = self._init_hidden(X, H)

#         # 1️⃣ forget gate & input gate 都通过 GATv2Conv 计算
#         f_t_raw = self.gat_f(X, edge_index)   # [num_nodes, out_channels]
#         i_t_raw = self.gat_i(X, edge_index)   # [num_nodes, out_channels]

#         # sigmoid 激活，保证在 (0,1)
#         f_t = torch.sigmoid(f_t_raw)
#         i_t = torch.sigmoid(i_t_raw)

#         # 可选：归一化让 f_t + i_t = 1，避免累计漂移
#         sum_ft_it = f_t + i_t + 1e-8
#         f_t = f_t / sum_ft_it
#         i_t = i_t / sum_ft_it

#         # 2️⃣ 候选隐藏状态 h_tilde：用 GATv2Conv 融合邻居信息
#         h_tilde = self.gat_h_tilde(X, edge_index)

#         # 3️⃣ minLSTM 更新公式
#         H_new = f_t * H + i_t * h_tilde

#         return H_new

# import torch
# from torch import nn
# from torch_geometric.nn import GATv2Conv

# class GConvMinLSTM(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int):
#         super(GConvMinLSTM, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         # 🔹 门控只用 Linear（节省显存）
#         # 这里输入是 X + H，因此 in_channels*2
#         self.linear_f = nn.Linear(in_channels + out_channels, out_channels, bias=False)
#         self.linear_i = nn.Linear(in_channels + out_channels, out_channels, bias=False)

#         # 🔹 仅候选隐藏状态用 GATv2Conv 保留图结构信息
#         self.gat_h_tilde = GATv2Conv(in_channels, out_channels, heads=1, concat=False)

#     def _init_hidden(self, X, H):
#         """初始化隐藏状态 H"""
#         if H is None:
#             H = torch.zeros(X.shape[0], self.out_channels, device=X.device)
#         return H

#     def forward(
#         self,
#         X: torch.FloatTensor,
#         edge_index: torch.LongTensor,
#         H: torch.FloatTensor = None,
#     ):
#         """
#         GConvMinLSTM forward pass（门控用 Linear 依赖 X+H，仅 h_tilde 用 GAT）

#         Args:
#             X: 输入特征 [num_nodes, in_channels]
#             edge_index: 图边索引
#             H: 上一隐藏状态 [num_nodes, out_channels] (可选)

#         Returns:
#             H_new: 更新后的隐藏状态
#         """
#         device = next(self.parameters()).device
#         X = X.to(device)
#         edge_index = edge_index.to(device)

#         # 初始化 H
#         H = self._init_hidden(X, H)

#         # 🔹 拼接 X + H 作为门控输入
#         gate_input = torch.cat([X, H], dim=-1)  # [num_nodes, in_channels+out_channels]

#         # 1️⃣ forget/input gate（Linear 节省显存）
#         f_t = torch.sigmoid(self.linear_f(gate_input))
#         i_t = torch.sigmoid(self.linear_i(gate_input))

#         # ✅ 可选：归一化让 f_t + i_t = 1，避免累计漂移
#         sum_ft_it = f_t + i_t + 1e-8
#         f_t = f_t / sum_ft_it
#         i_t = i_t / sum_ft_it

#         # 2️⃣ 候选隐藏状态 h_tilde：用 GATv2Conv 融合邻居信息
#         h_tilde = self.gat_h_tilde(X, edge_index)

#         # 3️⃣ minLSTM 更新公式
#         H_new = f_t * H + i_t * h_tilde

#         return H_new







# logspaceminLSTM
# import torch
# from torch import nn
# from torch.nn import Parameter
# from torch_geometric.nn.inits import zeros
# from typing import Tuple
# import torch.nn.functional as F


# class MinLSTMCell(torch.nn.Module):
#     def __init__(self, in_channels: int, out_channels: int):
#         super(MinLSTMCell, self).__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         # Input gate
#         self.lin_x_i = nn.Linear(in_channels, out_channels, bias=True)
#         # Forget gate
#         self.lin_x_f = nn.Linear(in_channels, out_channels, bias=True)
#         # Cell candidate
#         self.lin_x_c = nn.Linear(in_channels, out_channels, bias=True)

#         # Initialize bias
#         self._reset_parameters()

#     def _reset_parameters(self):
#         zeros(self.lin_x_i.bias)
#         zeros(self.lin_x_f.bias)
#         zeros(self.lin_x_c.bias)

#     def _init_hidden_state(self, X, H):
#         if H is None:
#             return torch.zeros(X.shape[0], self.out_channels, device=X.device)
#         return H

#     def _safe_log_g(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         保证 h̃_t 为正，使得 log(h̃_t) 有定义。
#         g(x) = x + 0.5 if x ≥ 0, sigmoid(x) otherwise
#         log(g(x)) = log(x + 0.5) or -softplus(-x)
#         """
#         # return torch.where(x >= 0, (x + 0.5).log(), -F.softplus(-x))
#         x_safe = torch.where(x >= 0, x + 0.5, torch.sigmoid(x))
#         return torch.log(x_safe.clamp(min=1e-8))


#     def forward(
#         self,
#         X: torch.FloatTensor,
#         H: torch.FloatTensor = None,
#         C: torch.FloatTensor = None,
#     ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
#         """
#         Single-step log-space minLSTM cell.

#         Args:
#             X: [batch_size, in_channels]
#             H: [batch_size, out_channels]  # unused in log-space minLSTM
#             C: [batch_size, out_channels]

#         Returns:
#             H: Updated hidden state (equal to cell state)
#             C: Updated cell state
#         """
#         X = X.to(next(self.parameters()).device)
#         C = self._init_hidden_state(X, C)

#         # Step 1: compute log f', log i' using softplus-normalized log-ratio
#         # k_i = self.lin_x_i(X)  # pre-activation for input gate
#         # k_f = self.lin_x_f(X)  # pre-activation for forget gate

#         # log_i = -F.softplus(-k_i)  # log sigmoid(k_i)
#         # log_f = -F.softplus(-k_f)  # log sigmoid(k_f)
#         # Step 1: compute log f, log i via log-sigmoid
#         log_i = -F.softplus(-self.lin_x_i(X))  # log(sigmoid(k_i))
#         log_f = -F.softplus(-self.lin_x_f(X))  # log(sigmoid(k_f))

#         # log i', f' = softmax(log_i, log_f)
#         # diff = log_i - log_f
#         # log_i_prime = -F.softplus(-diff)
#         # log_f_prime = -F.softplus(diff)
#         log_sum = torch.logsumexp(torch.stack([log_i, log_f], dim=0), dim=0)
#         log_i_prime = log_i - log_sum
#         log_f_prime = log_f - log_sum

#         # Step 2: compute log h̃_t
#         pre_h = self.lin_x_c(X)
#         log_h_tilde = self._safe_log_g(pre_h)

#         # Step 3: log-space update: log C_t = logsumexp(log f' + log C + log i' + log h̃)
#         log_C = C.clamp(min=1e-8).log()  # avoid log(0)
#         log_term1 = log_f_prime + log_C
#         log_term2 = log_i_prime + log_h_tilde

#         log_C_new = torch.logsumexp(torch.stack([log_term1, log_term2], dim=0), dim=0)
#         C_new = log_C_new.exp()

#         return C_new, C_new  # H = C
