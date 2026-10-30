# from dataset.fpcMulti import FPC
# from model.simulator_GNN_LSTM import Simulator
# import torch
# from utils.noise import get_velocity_noise
# import time
# import os
# import sys
# import copy
# from utils.utils import NodeType
# from torch_geometric.loader import DataLoader
# import torch_geometric.transforms as T


# dataset_dir = "dataset/data_downsampled"
# batch_size = 10
# noise_std = 1e-3

# print_batch = 50
# save_batch = 50


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# simulator = Simulator(message_passing_num=10, node_input_size=11, edge_input_size=3, device=device)

# # Load the pre-trained model
# # simulator.load_checkpoint(ckpdir="checkpoint/preSim.pth")

# optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-5)
# print('Optimizer initialized')


# def train(model: Simulator, dataloader, optimizer, use_teacher_forcing=True):

#     for batch_index, graph in enumerate(dataloader):

#         graph = transformer(graph)
#         graph = graph.cuda()

#         node_type = graph.x[:, 0] #"node_type, cur_v, pressure, time"
#         current_velocity = graph.x[:, 1:3]  # Current velocity at time t

#         # Mask for NORMAL and OUTFLOW nodes
#         mask = torch.logical_or(
#             node_type == NodeType.NORMAL, node_type == NodeType.OUTFLOW
#         )
#         # Mask for boundary nodes (INLET, WALL, CYLINDER)
#         boundary_mask = torch.logical_not(mask)

#         total_loss = 0.0
#         num_steps = graph.future_velocities.shape[1] // 2

#         hidden = None  # ← 初始化 LSTM 隐状态

#         for step in range(num_steps):

#             start_idx = step * 2
#             end_idx = (step + 1) * 2
#             target_velocity = graph.future_velocities[:, start_idx:end_idx]

#             # Prepare the graph data
#             graph.x[:, 1:3] = current_velocity
#             graph.y = target_velocity

#             # Determine whether to add noise
#             if use_teacher_forcing or step == 0:
#                 # Add noise to the current velocity
#                 velocity_sequence_noise = get_velocity_noise(
#                     graph, noise_std=noise_std, device=device
#                 )
#             else:
#                 # No noise added
#                 velocity_sequence_noise = torch.zeros_like(current_velocity).to(device)

#             # Clone the graph before passing it to the model
#             graph_clone = graph.clone()
#             # Run the model
#             predicted_acc, target_acc, hidden = model(
#                 graph_clone, velocity_sequence_noise, hidden
#             )

#             # Compute loss
#             errors = ((predicted_acc - target_acc) ** 2)[mask]
#             loss = torch.mean(errors)
#             total_loss += loss

#             # Update current_velocity
#             if use_teacher_forcing:
#                 # Use ground truth
#                 current_velocity = target_velocity
#             else:
#                 # Use predicted acceleration to update velocity
#                 velocity_update = model._output_normalizer.inverse(predicted_acc)

#                 current_velocity = current_velocity + velocity_update

#                 # Enforce boundary conditions
#                 current_velocity[boundary_mask] = target_velocity[boundary_mask]

#                 # Prevent gradient propagation through time
#                 current_velocity = current_velocity.detach()
#                 # current_velocity = current_velocity

#         # Backpropagation
#         optimizer.zero_grad()
#         total_loss.backward()
#         optimizer.step()

#         if batch_index % print_batch == 0:
#             print("batch %d [loss %.2e]" % (batch_index, total_loss.item()))

#         if batch_index % save_batch == 0:
#             model.save_checkpoint()


# if __name__ == '__main__':

#     dataset_fpc = FPC(
#         max_epochs=3,
#         dataset_dir=dataset_dir,
#         split="train",
#         tra_len=100,
#         time_iterval=0.03,
#         num_steps=5,
#     )
#     train_loader = DataLoader(dataset=dataset_fpc, batch_size=batch_size, num_workers=10)
#     transformer = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])

#     start_time = time.time()  # Record the start time
#     train(simulator, train_loader, optimizer, use_teacher_forcing=False)
#     end_time = time.time()  # Record the end time

#     total_time = end_time - start_time
#     print("Total training time: {:.2f} seconds".format(total_time))

#     sys.stdout.flush()
#     time.sleep(30)
#     os.system("/usr/bin/shutdown")



from dataset.fpcMulti import FPC
# from model.simulator_GNN_LSTM import Simulator
from model.simulator_GCRN import Simulator
# from model.simulator_GNN_LSTM_GNN import Simulator
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
# noise_std = 1e-3

print_batch = 50
save_batch = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# simulator = Simulator(message_passing_num=10, node_input_size=11, edge_input_size=3, device=device)

#optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-5)
#print('Optimizer initialized')


# def train(model: Simulator, dataloader, optimizer, use_teacher_forcing=True):
# def train(model: Simulator, dataloader, optimizer, tf_ratio: float = 1.0):
def train(model: Simulator, dataloader, optimizer, use_teacher_forcing=True, tf_noise_std=2e-2, ar_noise_std=2e-3):
    for batch_index, graph in enumerate(dataloader):

        graph = transformer(graph)
        graph = graph.cuda()

        node_type = graph.x[:, 0]  # "node_type, cur_v, pressure, time"
        current_velocity = graph.x[:, 1:3]

        mask = torch.logical_or(
            node_type == NodeType.NORMAL, node_type == NodeType.OUTFLOW
        )
        boundary_mask = torch.logical_not(mask)

        total_loss = 0.0
        num_steps = graph.future_velocities.shape[1] // 2
        hidden = None

        for step in range(num_steps):

            start_idx = step * 2
            end_idx = (step + 1) * 2
            target_velocity = graph.future_velocities[:, start_idx:end_idx]

            graph.x[:, 1:3] = current_velocity
            graph.y = target_velocity

            # if use_teacher_forcing or step == 0:
            #     velocity_sequence_noise = get_velocity_noise(
            #         graph, noise_std=noise_std, device=device
            #     )
            # else:
            #     velocity_sequence_noise = torch.zeros_like(current_velocity).to(device)

            # # —— 按比例决定本步是否使用 TF（step==0 保持稳态，仍走 TF）——
            # use_tf_step = (step == 0) or (torch.rand(1).item() < tf_ratio)

            # # 噪声逻辑：TF 或 step==0 时加噪，否则为 0
            # if use_tf_step:
            #     velocity_sequence_noise = get_velocity_noise(
            #         graph, noise_std=noise_std, device=device
            #     )
            # else:
            #     velocity_sequence_noise = torch.zeros_like(current_velocity).to(device)
            

            # ===== 修改噪声逻辑 =====
            if use_teacher_forcing:
                velocity_sequence_noise = get_velocity_noise(
                    graph, noise_std=tf_noise_std, device=device
                )
            else:
                if step == 0:
                    velocity_sequence_noise = get_velocity_noise(
                        graph, noise_std=ar_noise_std, device=device
                    )
                else:
                    velocity_sequence_noise = torch.zeros_like(current_velocity).to(device)
            # =======================

            graph_clone = graph.clone()
            predicted_acc, target_acc, hidden = model(
                graph_clone, velocity_sequence_noise, hidden
            )

            errors = ((predicted_acc - target_acc) ** 2)[mask]
            loss = torch.mean(errors)
            total_loss += loss

            # if use_tf_step:
            #     # Teacher Forcing: 直接用 GT 作为下一步输入
            #     current_velocity = target_velocity
            # else:
            #     # Autoregressive: 用预测增量更新
            #     velocity_update = model._output_normalizer.inverse(predicted_acc)
            #     current_velocity = current_velocity + velocity_update
            #     # 边界点仍用 GT 覆盖
            #     current_velocity[boundary_mask] = target_velocity[boundary_mask]
            #     current_velocity = current_velocity.detach()

            if use_teacher_forcing:
                current_velocity = target_velocity
            else:
                velocity_update = model._output_normalizer.inverse(predicted_acc)
                current_velocity = current_velocity + velocity_update
                current_velocity[boundary_mask] = target_velocity[boundary_mask]
                current_velocity = current_velocity.detach()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if batch_index % print_batch == 0:
            print("batch %d [loss %.2e]" % (batch_index, total_loss.item()))

        if batch_index % save_batch == 0:
            model.save_checkpoint(savedir=checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_root", type=str, default="checkpoint_minLSTM")
    args = parser.parse_args()

    # 构建最终保存路径：<save_root>/final_model.pth
    os.makedirs(args.save_root, exist_ok=True)
    checkpoint_path = os.path.join(args.save_root, "simulator.pth")
    print(f"Checkpoint path: {checkpoint_path}")
    
    total_epochs = 10
    tf_lr = 1e-4              # Teacher-forcing 学习率
    ar_lr = 1e-5              # Autoregressive 学习率
    dataset_fpc = FPC(
        max_epochs=1,
        dataset_dir=dataset_dir,
        split="train",
        tra_len=100,
        time_iterval=0.03,
        num_steps=10,
    )
    train_loader = DataLoader(dataset=dataset_fpc, batch_size=batch_size, num_workers=10)
    transformer = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])

    # simulator = Simulator(message_passing_num=12, node_input_size=11, edge_input_size=3, device=device, use_gru=False)
    # simulator = Simulator(message_passing_num=12, node_input_size=11, edge_input_size=3, device=device, use_gconv_lstm=True)
    simulator = Simulator(message_passing_num=12, node_input_size=11, edge_input_size=3, device=device, use_minlstm=True)
    # simulator = Simulator(message_passing_num=12, node_input_size=11, edge_input_size=3, device=device)
    optimizer = torch.optim.Adam(simulator.parameters(), lr=tf_lr)

    start_time = time.time()

    tf_noise_std = 2e-2     # Teacher Forcing 阶段噪声
    ar_noise_std = 2e-3     # Autoregressive 阶段噪声

    for epoch in range(total_epochs):
        use_teacher_forcing = epoch < 7  # 前几个 epoch 使用 teacher forcing
        # 动态设置 learning rate
        current_lr = tf_lr if use_teacher_forcing else ar_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        current_noise_std = tf_noise_std if use_teacher_forcing else ar_noise_std

        print(f"\n=== Epoch {epoch+1}/{total_epochs} | "
              f"{'Teacher Forcing' if use_teacher_forcing else 'Autoregressive'} | LR = {current_lr:.1e} | Noise std = {current_noise_std:.1e} ===")

        train(simulator, train_loader, optimizer, use_teacher_forcing=use_teacher_forcing, tf_noise_std=tf_noise_std,
              ar_noise_std=ar_noise_std)

    # 计划：1-7 epoch: TF=1.0；第8 epoch: TF=0.5；第9-10 epoch: TF=0.0
    # for epoch in range(total_epochs):
    #     if epoch < 7:
    #         tf_ratio = 1.0
    #     elif epoch == 7:
    #         tf_ratio = 0.5
    #     else:
    #         tf_ratio = 0.0

    #     # 学习率随阶段切换
    #     current_lr = tf_lr if tf_ratio > 0.0 else ar_lr
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = current_lr

    #     print(f"\n=== Epoch {epoch+1}/{total_epochs} | "
    #           f"Mode = {'TF' if tf_ratio==1.0 else ('Mixed' if tf_ratio>0.0 else 'AR')} | "
    #           f"TF ratio = {tf_ratio:.2f} | LR = {current_lr:.1e} ===")

    #     train(simulator, train_loader, optimizer, tf_ratio=tf_ratio)

    #     if hasattr(dataset_fpc, "shuffle_file"):
    #         dataset_fpc.shuffle_file()
        
    # for epoch in range(total_epochs):
    #     use_teacher_forcing = epoch < 7  # 前几个 epoch 使用 teacher forcing
    #     # 动态设置 learning rate
    #     current_lr = tf_lr if use_teacher_forcing else ar_lr
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = current_lr

    #     print(f"\n=== Epoch {epoch+1}/{total_epochs} | "
    #           f"{'Teacher Forcing' if use_teacher_forcing else 'Autoregressive'} | LR = {current_lr:.1e} ===")

    #     train(simulator, train_loader, optimizer, use_teacher_forcing=use_teacher_forcing)

    #     if hasattr(dataset_fpc, "shuffle_file"):
    #         dataset_fpc.shuffle_file()

    end_time = time.time()
    print("Total training time: {:.2f} seconds".format(end_time - start_time))

    # sys.stdout.flush()
    # time.sleep(30)
    # os.system("/usr/bin/shutdown")


# 课程学习 Inverse
# from dataset.fpcMulti import FPC
# from model.simulator_GNN_LSTM import Simulator
# # from model.simulator_GCRN import Simulator
# # from model.simulator_GNN_LSTM_GNN import Simulator
# import torch
# from utils.noise import get_velocity_noise
# import time
# import os
# import sys
# import copy
# from utils.utils import NodeType
# from torch_geometric.loader import DataLoader
# import torch_geometric.transforms as T
# import math

# dataset_dir = "dataset/data_downsampled"
# batch_size = 7
# noise_std = 1e-3

# print_batch = 50
# save_batch = 50

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # simulator = Simulator(message_passing_num=10, node_input_size=11, edge_input_size=3, device=device)

# #optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-5)
# #print('Optimizer initialized')

# def train(model: Simulator, dataloader, optimizer, teacher_forcing_ratio=1.0):
# # def train(model: Simulator, dataloader, optimizer, use_teacher_forcing=True):
#     for batch_index, graph in enumerate(dataloader):

#         graph = transformer(graph)
#         graph = graph.cuda()

#         node_type = graph.x[:, 0]  # "node_type, cur_v, pressure, time"
#         current_velocity = graph.x[:, 1:3]

#         mask = torch.logical_or(
#             node_type == NodeType.NORMAL, node_type == NodeType.OUTFLOW
#         )
#         boundary_mask = torch.logical_not(mask)

#         total_loss = 0.0
#         num_steps = graph.future_velocities.shape[1] // 2
#         hidden = None

#         for step in range(num_steps):

#             start_idx = step * 2
#             end_idx = (step + 1) * 2
#             target_velocity = graph.future_velocities[:, start_idx:end_idx]

#             graph.x[:, 1:3] = current_velocity
#             graph.y = target_velocity

#             # if use_teacher_forcing or step == 0:
#             use_tf = (torch.rand(1).item() < teacher_forcing_ratio) or (step == 0)

#             if use_tf:
#                 velocity_sequence_noise = get_velocity_noise(
#                     graph, noise_std=noise_std, device=device
#                 )
#             else:
#                 velocity_sequence_noise = torch.zeros_like(current_velocity).to(device)

#             graph_clone = graph.clone()
#             predicted_acc, target_acc, hidden = model(
#                 graph_clone, velocity_sequence_noise, hidden
#             )

#             errors = ((predicted_acc - target_acc) ** 2)[mask]
#             loss = torch.mean(errors)
#             total_loss += loss

#             if use_tf:
#                 current_velocity = target_velocity
#             else:
#                 velocity_update = model._output_normalizer.inverse(predicted_acc)
#                 current_velocity = current_velocity + velocity_update
#                 current_velocity[boundary_mask] = target_velocity[boundary_mask]
#                 current_velocity = current_velocity.detach()

#         optimizer.zero_grad()
#         total_loss.backward()
#         optimizer.step()

#         if batch_index % print_batch == 0:
#             print("batch %d [loss %.2e]" % (batch_index, total_loss.item()))

#         if batch_index % save_batch == 0:
#             model.save_checkpoint()


# if __name__ == '__main__':
#     total_epochs = 10
#     tf_lr = 1e-4              # Teacher-forcing 学习率
#     ar_lr = 1e-5              # Autoregressive 学习率
#     dataset_fpc = FPC(
#         max_epochs=1,
#         dataset_dir=dataset_dir,
#         split="train",
#         tra_len=100,
#         time_iterval=0.03,
#         num_steps=10,
#     )
#     train_loader = DataLoader(dataset=dataset_fpc, batch_size=batch_size, num_workers=10)
#     transformer = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])

#     simulator = Simulator(message_passing_num=12, node_input_size=11, edge_input_size=3, device=device, use_gru=False)
#     # simulator = Simulator(message_passing_num=12, node_input_size=11, edge_input_size=3, device=device, use_gconv_lstm=True)
#     # simulator = Simulator(message_passing_num=12, node_input_size=11, edge_input_size=3, device=device, use_minlstm=True)
#     # simulator = Simulator(message_passing_num=12, node_input_size=11, edge_input_size=3, device=device)
#     optimizer = torch.optim.Adam(simulator.parameters(), lr=tf_lr)

#     start_time = time.time()

#     k = total_epochs / math.log(total_epochs)  # 合理的默认值（经验）

#     for epoch in range(total_epochs):
#         # ✅ Inverse Sigmoid: p(t) = k / (k + exp(t / k))
#         teacher_forcing_ratio = k / (k + math.exp(epoch / k))

#         current_lr = tf_lr if teacher_forcing_ratio > 0.5 else ar_lr
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = current_lr

#         print(f"\n=== Epoch {epoch+1}/{total_epochs} | TF ratio = {teacher_forcing_ratio:.5f} | LR = {current_lr:.1e} ===")

#         train(simulator, train_loader, optimizer, teacher_forcing_ratio=teacher_forcing_ratio)

#     # for epoch in range(total_epochs):
#     #     use_teacher_forcing = epoch < 3  # 前几个 epoch 使用 teacher forcing
#     #     # 动态设置 learning rate
#     #     current_lr = tf_lr if use_teacher_forcing else ar_lr
#     #     for param_group in optimizer.param_groups:
#     #         param_group['lr'] = current_lr

#     #     print(f"\n=== Epoch {epoch+1}/{total_epochs} | "
#     #           f"{'Teacher Forcing' if use_teacher_forcing else 'Autoregressive'} | LR = {current_lr:.1e} ===")

#     #     train(simulator, train_loader, optimizer, use_teacher_forcing=use_teacher_forcing)

#         if hasattr(dataset_fpc, "shuffle_file"):
#             dataset_fpc.shuffle_file()

#     end_time = time.time()
#     print("Total training time: {:.2f} seconds".format(end_time - start_time))

#     sys.stdout.flush()
#     time.sleep(30)
#     os.system("/usr/bin/shutdown")
