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

# from model.simulator_gconvlstm import Simulator
from model.simulator_LSTM import Simulator

import argparse


dataset_dir = "dataset/data_downsampled"
batch_size = 2

noise_std_stage1 = torch.tensor([2e-2, 2e-2, 1e-2, 2e-2])
noise_std_stage2 = torch.tensor([2e-3, 2e-3, 1e-3, 2e-3])

print_batch = 20
save_batch = 20

epochs_stage1 = 6
epochs_stage2 = 6

traLength = 80
numSteps = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# simulator = Simulator(message_passing_num=12, node_input_size=6, edge_input_size=5, device=device, use_gconv_lstm=True)
simulator = Simulator(message_passing_num=12, node_input_size=6, edge_input_size=5, device=device)

transformer = T.Compose([T.NormalizeScale(), T.Cartesian(norm=False), T.Distance(norm=False)])


# Limit normalizer max_accumulations
simulator._output_normalizer._max_accumulations = 300000
simulator._node_normalizer._max_accumulations = 300000


def train(model: Simulator, dataloader, optimizer, noise_std, use_teacher_forcing=True):
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

            start_idx = step * 4
            end_idx = (step + 1) * 4
            targetsT = graph.future_prims[:, start_idx:end_idx]

            # Enforce boundary conditions
            current_prims[boundary_mask] = targetsT[boundary_mask]

            # Prepare the graph data
            graph.x[:, 1:5] = current_prims
            graph.y = targetsT

            # Determine whether to add noise
            if use_teacher_forcing or step == 0:
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

            # Update current_prims
            if use_teacher_forcing:
                # Use ground truth
                current_prims = targetsT
            else:
                # Use predicted  to update
                # current_prims = model._output_normalizer.inverse(predicted_acc)
                # current_prims = predicted_acc
                current_prims = current_prims + predicted_acc

                # Enforce boundary conditions
                current_prims[boundary_mask] = targetsT[boundary_mask]

                # Prevent gradient propagation through time
                current_prims = current_prims.detach()
                # current_prims = current_prims

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if batch_index % print_batch == 0:
            print("batch %d [loss %.2e]" % (batch_index, total_loss.item()))

        if batch_index % save_batch == 0:
            model.save_checkpoint(savedir=checkpoint_path)

            # model.save_checkpoint()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_root", type=str, default="checkpoint_GNN_6_6")
    args = parser.parse_args()

    # 构建最终保存路径：<save_root>/final_model.pth
    os.makedirs(args.save_root, exist_ok=True)
    checkpoint_path = os.path.join(args.save_root, "simulator.pth")
    print(f"Checkpoint path: {checkpoint_path}")

    start_time = time.time()  # Record the start time

    # ── Stage 1: teacher forcing ───────────────────────────────────────────────
    print(f"\n--- Stage 1: teacher forcing for {epochs_stage1} epochs ------")
    optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-4)
    print('Optimizer initialized')

    dataset_stage1 = FPC(
        max_epochs=epochs_stage1,
        dataset_dir=dataset_dir,
        split="train",
        tra_len=traLength,
        time_iterval=0.1,
        num_steps=numSteps,
    )
    loader1 = DataLoader(dataset=dataset_stage1, batch_size=batch_size, num_workers=5)

    train(simulator, loader1, optimizer, noise_std_stage1, use_teacher_forcing=True)


    # reduce learning rate in-place for stage 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = 1e-5

    # ── Stage 2: multistep ─────────────────────────────────────────────────────
    print(f"\n--- Stage 2: multistep training for {epochs_stage2} epochs ------")

    dataset_stage2 = FPC(
        max_epochs=epochs_stage2,
        dataset_dir=dataset_dir,
        split="train",
        tra_len=traLength,
        time_iterval=0.1,
        num_steps=numSteps,
    )
    loader2 = DataLoader(dataset=dataset_stage2, batch_size=batch_size, num_workers=5)
    
    train(simulator, loader2, optimizer, noise_std_stage2, use_teacher_forcing=False)

    end_time = time.time()  # Record the end time

    total_time = end_time - start_time
    print("Total training time: {:.2f} seconds".format(total_time))

    sys.stdout.flush()
    time.sleep(30)
    os.system("/usr/bin/shutdown")
