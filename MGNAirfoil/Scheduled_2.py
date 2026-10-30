from dataset.fpcMulti import FPC
import torch
from utils.noise import get_noise, get_noise_rel
import time
import os
import sys
import copy
from utils.utils import NodeType
from utils.to_undirected import make_edges_undirected
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from utils.lossCompute import dataLoss
import argparse
from datetime import datetime

from model.simulator_LSTM import Simulator


dataset_dir = "dataset/data_downsampled"
batch_size = 3

noise_std = torch.tensor([1.0, 1.0, 1e-3])
# noise_std_stage2 = torch.tensor([2e-3, 2e-3, 1e-3, 2e-3])

print_batch = 20
save_batch = 20

epochs = 10

traLength = 120
numSteps = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
simulator = Simulator(message_passing_num=15, node_input_size=12, edge_input_size=3, device=device, use_gru=False)

transformer = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])


# Limit normalizer max_accumulations
simulator._output_normalizer._max_accumulations = 300000
simulator._node_normalizer._max_accumulations = 300000


def train(model: Simulator, dataloader, optimizer, noise_std, tf_ratio: float):
    model.train()

    for batch_index, graph in enumerate(dataloader):

        # copy directed (unique) edge_index and to_undirected
        # graph = make_edges_undirected(graph)

        graph = transformer(graph)
        graph = graph.cuda()

        # "node_type, rho, v1, v2, pres, Mach"
        node_type = graph.x[:, 0]
        current_velocity = graph.x[:, 1:4]  # Current  at time t

        # Mask for NORMAL nodes
        mask = torch.logical_or(
            node_type == NodeType.NORMAL, node_type == NodeType.OUTFLOW
        )
        # Mask for boundary nodes (INLET, WALL, CYLINDER)
        boundary_mask = torch.logical_not(mask)

        total_loss = 0.0
        num_steps = graph.future_prims.shape[1] // 2
        hidden = None

        for step in range(num_steps):

            # Scheduled Sampling：这一时间步是否使用 GT
            if step == 0:
                # 第一个时间步一般强制用 GT
                use_tf_step = True
            else:
                use_tf_step = (torch.rand(1).item() < tf_ratio)

            start_idx = step * 2
            end_idx = (step + 1) * 2
            target_velocity = graph.future_velocities[:, start_idx:end_idx]
            target_density = graph.future_density[:, step]
            targetsT = torch.cat([target_velocity, target_density.unsqueeze(1)], dim=1)

            # Enforce boundary conditions
            # current_prims[boundary_mask] = targetsT[boundary_mask]

            # Prepare the graph data
            graph.x[:, 1:4] = current_prims
            graph.y = targetsT

            # Determine whether to add noise
            if use_tf_step or step == 0:
                # Add noise to the current
                sequence_noise = get_noise(graph, boundary_mask, noise_std=noise_std, device=device)
            else:
                # No noise added
                sequence_noise = torch.zeros_like(current_prims).to(device)

            # Clone the graph before passing it to the model
            graph_clone = graph.clone()
            # Run the model
            predicted_acc, target_acc, hidden = model(graph_clone, sequence_noise, hidden)

            # Compute loss
            errors = ((predicted_acc - target_acc) ** 2)[mask]
            loss = torch.mean(errors)
            total_loss += loss

            # # Update current_prims
            # if use_teacher_forcing:
            #     # Use ground truth
            #     current_prims = targetsT
            # else:
            #     # Use predicted  to update
            #     # current_prims = model._output_normalizer.inverse(predicted_acc)
            #     # current_prims = predicted_acc
            #     current_prims = current_prims + predicted_acc

            #     # Enforce boundary conditions
            #     current_prims[boundary_mask] = targetsT[boundary_mask]

            #     # Prevent gradient propagation through time
            #     current_prims = current_prims.detach()

            # 更新 current_prims：teacher forcing vs 自回归
            if use_tf_step:
                # teacher forcing：用 ground truth
                current_prims = targetsT
            else:
                # autoregressive：current_prims = current_prims + predicted_acc
                current_prims = current_prims + predicted_acc

                # Enforce boundary conditions
                current_prims[boundary_mask] = targetsT[boundary_mask]

                # Prevent gradient propagation through time
                current_prims = current_prims.detach()

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if batch_index % print_batch == 0:
            print(
                "batch %d [loss %.2e]  tf_ratio=%.3f  use_tf_batch=%s"
                % (batch_index, total_loss.item(), tf_ratio, use_tf_step)
            )

        if batch_index % save_batch == 0:
            model.save_checkpoint(savedir=checkpoint_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_root", type=str, default="checkpoint_tf_2")
    args = parser.parse_args()

    os.makedirs(args.save_root, exist_ok=True)
    checkpoint_path = os.path.join(args.save_root, "final_model_2.pth")
    print(f"Checkpoint path: {checkpoint_path}")


    start_time = time.time()  # Record the start time

    print(f"\n--- Linear teacher forcing decay for {epochs} epochs (1.0 -> 0.0) ------")
    optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-4)
    print('Optimizer initialized')

    # 指数衰减：从 eps_start 衰减到 eps_final（不严格等于 0，更稳定）
    eps_start = 1.0   # 第 0 个 epoch 的 TF 概率
    eps_final = 1e-3  # 最后一个 epoch 仍保留一点点 TF（可改小/大）

    if epochs > 1:
        # 计算每个 epoch 乘的衰减因子 gamma，使得：
        # eps_start * gamma^(epochs-1) = eps_final
        gamma = (eps_final / eps_start) ** (1.0 / (epochs - 1))
    else:
        gamma = 1.0

    for epoch in range(epochs):
        # 指数衰减：tf_ratio = eps_start * gamma^epoch
        tf_ratio = eps_start * (gamma ** epoch)
        # 数值安全截断
        tf_ratio = float(max(0.0, min(1.0, tf_ratio)))

        print(f"\nEpoch {epoch + 1}/{epochs}  tf_ratio = {tf_ratio:.4f}  (gamma = {gamma:.4f})")

        # 每个 epoch 新建一次 dataset/DataLoader，等价于重跑一遍 train split
        dataset = FPC(
            max_epochs=1,              # 每次 __iter__ 跑 1 个 epoch
            dataset_dir=dataset_dir,
            split="train",
            tra_len=traLength,
            time_iterval=0.04,
            num_steps=numSteps,
        )
        loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=5)

        train(simulator, loader, optimizer, noise_std, tf_ratio)

    end_time = time.time()  # Record the end time

    total_time = end_time - start_time
    print("Total training time: {:.2f} seconds".format(total_time))

    # sys.stdout.flush()
    # time.sleep(30)
    # os.system("/usr/bin/shutdown")
