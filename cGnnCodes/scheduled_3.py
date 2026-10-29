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
import math 
import argparse
from datetime import datetime


from model.simulator_LSTM import Simulator


dataset_dir = "dataset/data_downsampled"
batch_size = 2

noise_std = torch.tensor([2e-2, 2e-2, 1e-2, 2e-2])
# noise_std_stage2 = torch.tensor([2e-3, 2e-3, 1e-3, 2e-3])

print_batch = 20
save_batch = 20

epochs = 12

# inverse sigmoid 的 k 参数（可调，k 越大衰减越慢）
tf_k = 1.0   # k >= 1，根据“期望收敛速度”自己调

traLength = 80
numSteps = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
simulator = Simulator(message_passing_num=12, node_input_size=6, edge_input_size=5, device=device, use_gru=False)

transformer = T.Compose([T.NormalizeScale(), T.Cartesian(norm=False), T.Distance(norm=False)])


# Limit normalizer max_accumulations
simulator._output_normalizer._max_accumulations = 300000
simulator._node_normalizer._max_accumulations = 300000


def train(model: Simulator, dataloader, optimizer, noise_std, tf_ratio: float):
    model.train()

    for batch_index, graph in enumerate(dataloader):

        # copy directed (unique) edge_index and to_undirected
        graph = make_edges_undirected(graph)

        graph = transformer(graph)
        graph = graph.cuda()

        # "node_type, rho, v1, v2, pres, Mach"
        node_type = graph.x[:, 0]
        current_prims = graph.x[:, 1:5]  # Current  at time t

        # Mask for NORMAL nodes
        mask = (node_type == NodeType.NORMAL)
        # Mask for boundary nodes
        boundary_mask = torch.logical_not(mask)

        total_loss = 0.0
        num_steps = graph.future_prims.shape[1] // 4
        hidden = None

        for step in range(num_steps):

            # Scheduled Sampling：这一时间步是否使用 GT
            if step == 0:
                # 第一个时间步一般强制用 GT（起点用真值）
                use_tf_step = True
            else:
                use_tf_step = (torch.rand(1).item() < tf_ratio)

            start_idx = step * 4
            end_idx = (step + 1) * 4
            targetsT = graph.future_prims[:, start_idx:end_idx]

            # Enforce boundary conditions
            current_prims[boundary_mask] = targetsT[boundary_mask]

            # Prepare the graph data
            graph.x[:, 1:5] = current_prims
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
            errors = dataLoss(graph, predicted_acc, target_acc, mask)
            loss = errors
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
    parser.add_argument("--save_root", type=str, default="checkpoint_tf")
    args = parser.parse_args()

    os.makedirs(args.save_root, exist_ok=True)
    checkpoint_path = os.path.join(args.save_root, "final_model.pth")
    print(f"Checkpoint path: {checkpoint_path}")

    start_time = time.time()  # Record the start time

    print(f"\n--- Linear teacher forcing decay for {epochs} epochs (1.0 -> 0.0) ------")
    optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-4)
    print('Optimizer initialized')

    # 先算 eps_0 用于归一化，使第 0 个 epoch 的 tf_ratio = 1
    # 公式：eps_i = k / (k + exp(i / k))
    eps0 = tf_k / (tf_k + math.exp(0.0 / tf_k))  # = k / (k + 1)

    for epoch in range(epochs):
        # 论文中的 i 可以看作 epoch index（0,1,...,epochs-1）
        i = float(epoch)
        eps_i = tf_k / (tf_k + math.exp(i / tf_k))
        # 归一化到 [0,1]，保证第 0 个 epoch 为 1
        tf_ratio = eps_i / eps0
        tf_ratio = float(max(0.0, min(1.0, tf_ratio)))  # 稍作截断，防止数值误差

        print(f"\nEpoch {epoch + 1}/{epochs}  tf_ratio = {tf_ratio:.4f}")

        # 每个 epoch 新建一次 dataset/DataLoader，适配 FPC 这种 IterableDataset
        dataset = FPC(
            max_epochs=1,              # 每次 __iter__ 跑 1 个 epoch
            dataset_dir=dataset_dir,
            split="train",
            tra_len=traLength,
            time_iterval=0.1,
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
