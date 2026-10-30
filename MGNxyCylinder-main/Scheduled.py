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

import argparse


dataset_dir = "dataset/data_downsampled"
batch_size = 7

noise_std_stage1 = 2e-2
noise_std_stage2 = 2e-3

print_batch = 50
save_batch = 50

epochs_stage1 = 7
epochs_stage2 = 3

traLength = 100
numSteps = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
simulator = Simulator(message_passing_num=12, node_input_size=11, edge_input_size=3, device=device)

transformer = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])

# Load the pre-trained model
# simulator.load_checkpoint(ckpdir="checkpoint/preSim.pth")

# optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-5)
# print('Optimizer initialized')


def train(model: Simulator, dataloader, optimizer, noise_std, use_teacher_forcing=True):

    for batch_index, graph in enumerate(dataloader):

        graph = transformer(graph)
        graph = graph.cuda()

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

        for step in range(num_steps):

            start_idx = step * 2
            end_idx = (step + 1) * 2
            target_velocity = graph.future_velocities[:, start_idx:end_idx]

            # current_velocity[boundary_mask] = target_velocity[boundary_mask]

            # Prepare the graph data
            graph.x[:, 1:3] = current_velocity
            graph.y = target_velocity

            # Determine whether to add noise
            if use_teacher_forcing or step == 0:
                # Add noise to the current velocity
                velocity_sequence_noise = get_velocity_noise(
                    graph, noise_std=noise_std, device=device
                )
            else:
                # No noise added
                velocity_sequence_noise = torch.zeros_like(current_velocity).to(device)

            # Clone the graph before passing it to the model
            graph_clone = graph.clone()
            # Run the model
            predicted_acc, target_acc = model(graph_clone, velocity_sequence_noise)

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
                # current_velocity = current_velocity

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if batch_index % print_batch == 0:
            print("batch %d [loss %.2e]" % (batch_index, total_loss.item()))

        if batch_index % save_batch == 0:
            model.save_checkpoint(savedir=checkpoint_path)


# if __name__ == '__main__':

#     dataset_fpc = FPC(
#         max_epochs=3,
#         dataset_dir=dataset_dir,
#         split="train",
#         tra_len=100,
#         time_iterval=0.03,
#         num_steps=8,
#     )
#     train_loader = DataLoader(dataset=dataset_fpc, batch_size=batch_size, num_workers=10)
#     transformer = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])

#     start_time = time.time()  # Record the start time
#     train(simulator, train_loader, optimizer, use_teacher_forcing=False)
#     end_time = time.time()  # Record the end time

#     total_time = end_time - start_time
#     print("Total training time: {:.2f} seconds".format(total_time))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_root", type=str, default="checkpoint_test")
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
        time_iterval=0.03,
        num_steps=numSteps,
    )
    loader1 = DataLoader(dataset=dataset_stage1, batch_size=batch_size, num_workers=10)

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
        time_iterval=0.03,
        num_steps=numSteps,
    )
    loader2 = DataLoader(dataset=dataset_stage2, batch_size=batch_size, num_workers=10)
    
    train(simulator, loader2, optimizer, noise_std_stage2, use_teacher_forcing=False)

    end_time = time.time()  # Record the end time

    total_time = end_time - start_time
    print("Total training time: {:.2f} seconds".format(total_time))
    # sys.stdout.flush()
    # time.sleep(30)
    # os.system("/usr/bin/shutdown")