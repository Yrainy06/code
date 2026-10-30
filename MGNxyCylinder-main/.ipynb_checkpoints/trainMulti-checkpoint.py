from dataset.fpcMulti import FPC
from model.simulator import Simulator
import torch
from utils.noise import get_velocity_noise
import time
import os
import sys
import copy
from utils.utils import NodeType
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T


dataset_dir = "dataset/data_downsampled"
batch_size = 10
noise_std = 1e-3

print_batch = 50
save_batch = 50


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
simulator = Simulator(message_passing_num=10, node_input_size=11, edge_input_size=3, device=device)

# Load the pre-trained model
# simulator.load_checkpoint(ckpdir="checkpoint/preSim.pth")

optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-5)
print('Optimizer initialized')



# def generate_graph_sequence(graph, num_steps, velocity_sequence_noise=None):
#     """
#     从初始的 graph 生成多个时间步的 graph_sequence。

#     Args:
#         graph: 初始时间步的图数据
#         num_steps: 时间步数
#         velocity_sequence_noise: 噪声用于每个时间步的更新

#     Returns:
#         graph_sequence: 包含 num_steps 步的图数据序列
#     """
#     graph_sequence = []

#     # 在每个时间步生成图数据
#     current_velocity = graph.x[:, 1:3]  # 获取当前速度（作为输入）
    
#     for step in range(num_steps):
#         # 如果需要加入噪声
#         if velocity_sequence_noise is not None:
#             noised_frames = current_velocity + velocity_sequence_noise
#         else:
#             noised_frames = current_velocity
        
#         # 你可以在这里更新图的数据（例如节点特征）
#         graph.x[:, 1:3] = noised_frames

#         # 将修改后的 graph 添加到图序列中
#         graph_sequence.append(graph.clone())

#         # 用模型预测的速度更新 current_velocity
#         if step < num_steps - 1:
#             velocity_update = get_velocity_noise(graph, noise_std=noise_std, device=device)  # 这里可以用模型预测的加速
#             current_velocity = current_velocity + velocity_update

#     # 将 graph_sequence 转化为 tensor 或者保持为列表形式
#     # return torch.stack(graph_sequence, dim=1)  # (batch_size, seq_len, graph)
#     return graph_sequence



def train(model: Simulator, dataloader, optimizer, use_teacher_forcing=True):

    for batch_index, graph in enumerate(dataloader):

        graph = transformer(graph)
        graph = graph.cuda()

        # # 生成包含多时间步的 graph_sequence
        # num_steps = 5  # 或者使用 `graph.future_velocities.shape[1] // 2`
        # graph_sequence = generate_graph_sequence(graph, num_steps)
        # node_type = graph.x[:, 0]  # 获取节点类型
        # current_velocity = graph.x[:, 1:3]  # 获取速度信息

        node_type = graph.x[:, 0] #"node_type, cur_v, pressure, time"
        current_velocity = graph.x[:, 1:3]  # Current velocity at time t

        # Mask for NORMAL and OUTFLOW nodes
        mask = torch.logical_or(
            node_type == NodeType.NORMAL, node_type == NodeType.OUTFLOW
        )
        # Mask for boundary nodes (INLET, WALL, CYLINDER)
        boundary_mask = torch.logical_not(mask)

        total_loss = 0.0
        num_steps = graph.future_velocities.shape[1] // 2

        velocity_sequence = []  # 存储时间序列

        # predicted_acc = None
        # target_acc = None

        # 初始化LSTM的隐藏状态
        hidden_state = None

        for step in range(num_steps):

            start_idx = step * 2
            end_idx = (step + 1) * 2
            target_velocity = graph.future_velocities[:, start_idx:end_idx]

            # graph_t = graph_sequence[step]  # 当前时间步的 graph
            # target_velocity = graph_t.future_velocities  # 对应的目标速度（也可以是加速度）

            # Prepare the graph data
            graph.x[:, 1:3] = current_velocity
            graph.y = target_velocity
            # frames = graph.x[:, 1:3]

            # Determine whether to add noise
            if use_teacher_forcing or step == 0:
                # Add noise to the current velocity
                velocity_sequence_noise = get_velocity_noise(
                    graph, noise_std=noise_std, device=device
                )
            else:
                # No noise added
                velocity_sequence_noise = torch.zeros_like(current_velocity).to(device)

            # # Clone the graph before passing it to the model
            # graph_clone = graph_t.clone()
            # # Run the model
            # predicted_acc, target_acc = model(graph_clone, velocity_sequence_noise)

            # 记录时间序列
            # velocity_sequence.append(current_velocity.unsqueeze(1))  # 将速度数据添加到时间序列中
            graph_clone = graph.clone()
            predicted_acc, target_acc, hidden_state = model(graph_clone, velocity_sequence_noise, hidden_state)
            # # 使用LSTM处理时间序列
            # if len(velocity_sequence) > 1:
            #     velocity_sequence_tensor = torch.cat(velocity_sequence, dim=1)  # 拼接成(batch_size, sequence_len, feature_size)
            #     predicted_acc, target_acc = model(graph.clone(), velocity_sequence_tensor)
            # else:
            #     # 如果时间序列只有一个时间步，则直接计算
            #     graph_clone = graph.clone()
            #     predicted_acc, target_acc = model(graph_clone, velocity_sequence_noise)

            # Compute loss
            errors = ((predicted_acc - target_acc) ** 2)[mask]
            loss = torch.mean(errors)
            total_loss += loss

            # Update current_velocity
            if use_teacher_forcing:
                # Use ground truth
                current_velocity = target_velocity
            else:
                # Use predicted acceleration to update velocity
                velocity_update = model._output_normalizer.inverse(predicted_acc)

                current_velocity = current_velocity + velocity_update

                # Enforce boundary conditions
                current_velocity[boundary_mask] = target_velocity[boundary_mask]

                # Prevent gradient propagation through time
                current_velocity = current_velocity.detach()

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if batch_index % print_batch == 0:
            print("batch %d [loss %.2e]" % (batch_index, total_loss.item()))

        if batch_index % save_batch == 0:
            model.save_checkpoint()


if __name__ == '__main__':

    dataset_fpc = FPC(
        max_epochs=3,
        dataset_dir=dataset_dir,
        split="train",
        tra_len=120,
        time_iterval=0.03,
        num_steps=5,
    )
    train_loader = DataLoader(dataset=dataset_fpc, batch_size=batch_size, num_workers=10)
    transformer = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])

    start_time = time.time()  # Record the start time
    train(simulator, train_loader, optimizer, use_teacher_forcing=False)
    end_time = time.time()  # Record the end time

    total_time = end_time - start_time
    print("Total training time: {:.2f} seconds".format(total_time))

    sys.stdout.flush()
    time.sleep(30)
    os.system("/usr/bin/shutdown")
