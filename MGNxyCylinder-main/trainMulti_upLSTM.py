from dataset.fpcMulti import FPC
import torch
from utils.noise import get_velocity_noise
import time
import os
import sys
import copy
from utils.utils import NodeType
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

# from model.simulator_gconvlstm import Simulator
from model.simulator_GNN_LSTM import Simulator
# from model.simulator_GCRN import Simulator
# from model.simulator_GNN_LSTM_GNN import Simulator

import argparse

import math

def save_best_overall(model, optimizer, save_path, best_loss, stage_name, batch_index, kind="avg"):
    """
    kind: "avg" or "batch" 用于标记保存的是哪种best
    """
    to_save = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_loss": best_loss,
        "best_stage": stage_name,
        "best_batch": batch_index,
        "best_kind": kind,
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(to_save, save_path)

dataset_dir = "dataset/data_downsampled"
batch_size = 7

noise_std_stage1 = 2e-2
noise_std_stage2 = 2e-3

print_batch = 50
save_batch = 50

epochs_stage1 = 7
epochs_stage2 = 3

traLength = 120
numSteps = 5

# # ✅ 两个best分别保存
# BEST_AVG_PATH   = "checkpoint/best_avg.pth"
# BEST_BATCH_PATH = "checkpoint/best_batch.pth"
# BEST_BATCH_LAST_PATH = "checkpoint/best_batch_last.pth"

LAST_K_BATCHES = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# simulator = Simulator(message_passing_num=12, node_input_size=6, edge_input_size=5, device=device, use_gconv_lstm=True)
simulator = Simulator(message_passing_num=12, node_input_size=11, edge_input_size=3, device=device, use_gru=True)

transformer = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])


# def train(model: Simulator, dataloader, optimizer, noise_std, use_teacher_forcing=True):
def train(model: Simulator, dataloader, optimizer, noise_std, use_teacher_forcing=True,
          best_overall=None,
          save_best_avg_path=None,
          save_best_batch_path=None,
          save_best_batch_last_path=None,
          eval_every_batches=200,
          min_delta_avg=1e-5,     # avg best 的阈值（loss量级1e-3时建议 1e-5）
          min_delta_batch=0.0,    # batch best 的阈值（想少保存可设 1e-4）
          last_k_batches=200,
          stage_name="stage"):
    model.train()

    if save_best_avg_path is None or save_best_batch_path is None:
        raise ValueError("save_best_avg_path / save_best_batch_path must be provided.")

    # ✅ best_overall 同时维护 avg 与 batch 两套
    if best_overall is None:
        best_overall = {
            "best_avg": math.inf,
            "best_batch": math.inf,
            "best_batch_last": math.inf,
            # ✅ 用于尾段判断：一个滑动队列（只存最近 last_k_batches 的 loss）
            "tail_losses": []  # list of (batch_index, cur_loss)
        }

    running_loss = 0.0
    running_count = 0

    for batch_index, graph in enumerate(dataloader):

        graph = transformer(graph)
        graph = graph.cuda()

        # "node_type, rho, v1, v2, pres, Mach"
        node_type = graph.x[:, 0]
        current_velocity = graph.x[:, 1:3]  # Current  at time t

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
            
            current_velocity[boundary_mask] = target_velocity[boundary_mask]

            graph.x[:, 1:3] = current_velocity
            graph.y = target_velocity


            # Determine whether to add noise
            if use_teacher_forcing or step == 0:
                velocity_sequence_noise = get_velocity_noise(
                    graph, noise_std=noise_std, device=device
                )
            else:
                velocity_sequence_noise = torch.zeros_like(current_velocity).to(device)

            # Clone the graph before passing it to the model
            graph_clone = graph.clone()
            # Run the model
            predicted_acc, target_acc, hidden = model(
                graph_clone, velocity_sequence_noise, hidden
            )
            
            # Compute loss
            errors = ((predicted_acc - target_acc) ** 2)[mask]
            loss = torch.mean(errors)
            total_loss += loss


            # Update current_prims
            if use_teacher_forcing:
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
        
        cur_loss = total_loss.item()

        if batch_index % print_batch == 0:
            print("batch %d [loss %.2e]" % (batch_index, total_loss.item()))

        if batch_index % save_batch == 0:
            model.save_checkpoint(savedir=checkpoint_path)

            # model.save_checkpoint()

        # ==========================
        # ✅ 维护尾段队列（最近 last_k_batches）
        # ==========================
        tail = best_overall["tail_losses"]
        tail.append((batch_index, cur_loss))
        if len(tail) > last_k_batches:
            tail.pop(0)

        # =========================================================
        # ✅ 1) 全程最优（按单个 batch loss）——每个 batch 都可触发
        # =========================================================
        if cur_loss < best_overall["best_batch"] - min_delta_batch:
            best_overall["best_batch"] = cur_loss
            best_overall["best_batch_stage"] = stage_name
            best_overall["best_batch_at"] = batch_index

            save_best_overall(model, optimizer, save_best_batch_path,
                              best_overall["best_batch"], stage_name, batch_index, kind="batch")

            print(f"[BEST BATCH] ✅ loss={best_overall['best_batch']:.3e} @ {stage_name} batch {batch_index} saved -> {save_best_batch_path}")

        # 尾段最优（单 batch）
        # 注意：只有当尾段队列“已经满”时才保存，避免训练初期也被当成“尾段”
        if len(tail) == last_k_batches:
            # 当前 batch 是尾段队列的一员，若比尾段 best 更好就更新
            if cur_loss < best_overall["best_batch_last"] - min_delta_batch:
                best_overall["best_batch_last"] = cur_loss
                best_overall["best_batch_last_stage"] = stage_name
                best_overall["best_batch_last_at"] = batch_index

                save_best_overall(model, optimizer, save_best_batch_last_path,
                                  best_overall["best_batch_last"], stage_name, batch_index, kind="batch_last")

                print(f"[BEST BATCH LAST] ✅ loss={best_overall['best_batch_last']:.3e} (last {last_k_batches}) @ {stage_name} batch {batch_index} saved -> {save_best_batch_last_path}")


        # =========================================================
        # ✅ 2) 全程最优（按窗口平均 avg_loss）——每 eval_every_batches 触发一次
        # =========================================================
        running_loss += cur_loss
        running_count += 1

        if (batch_index + 1) % eval_every_batches == 0:
            avg_loss = running_loss / running_count
            running_loss = 0.0
            running_count = 0

            if avg_loss < best_overall["best_avg"] - min_delta_avg:
                best_overall["best_avg"] = avg_loss
                best_overall["best_avg_stage"] = stage_name
                best_overall["best_avg_at"] = batch_index

                save_best_overall(model, optimizer, save_best_avg_path,
                                  best_overall["best_avg"], stage_name, batch_index, kind="avg")

                print(f"[BEST AVG] ✅ avg_loss={best_overall['best_avg']:.3e} @ {stage_name} batch {batch_index} saved -> {save_best_avg_path}")

    return best_overall


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_root", type=str, default="checkpoint_LSTM_7_3")
    args = parser.parse_args()

    # 构建最终保存路径：<save_root>/final_model.pth
    # os.makedirs(args.save_root, exist_ok=True)
    # checkpoint_path = os.path.join(args.save_root, "simulator.pth")
    # print(f"Checkpoint path: {checkpoint_path}")
    # 构建最终保存路径：<save_root>/final_model.pth
    os.makedirs(args.save_root, exist_ok=True)
    checkpoint_path = os.path.join(args.save_root, "simulator.pth")
    best_avg_path   = os.path.join(args.save_root, "best_avg.pth")
    best_batch_path = os.path.join(args.save_root, "best_batch.pth")
    best_batch_last_path = os.path.join(args.save_root, "best_batch_last.pth")

    print(f"Checkpoint path: {checkpoint_path}")

    start_time = time.time()  # Record the start time
    
    # ✅ 全程共享 best（跨 stage1 + stage2）
    best_overall = {"best_avg": math.inf, "best_batch": math.inf, "best_batch_last": math.inf, "tail_losses": []}


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

    best_overall = train(simulator, loader1, optimizer, noise_std_stage1, use_teacher_forcing=True,
                         best_overall=best_overall,
                         save_best_avg_path=best_avg_path,
                         save_best_batch_path=best_batch_path,
                         save_best_batch_last_path=best_batch_last_path,
                         eval_every_batches=200,
                         min_delta_avg=1e-5,
                         min_delta_batch=1e-4,   # 想减少保存频率可改成 1e-4
                         last_k_batches=LAST_K_BATCHES,
                         stage_name="stage1")

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
    
    best_overall = train(simulator, loader2, optimizer, noise_std_stage2, use_teacher_forcing=False,
                         best_overall=best_overall,
                         save_best_avg_path=best_avg_path,
                         save_best_batch_path=best_batch_path,
                         save_best_batch_last_path=best_batch_last_path,
                         eval_every_batches=200,
                         min_delta_avg=1e-5,
                         min_delta_batch=1e-4,
                         last_k_batches=LAST_K_BATCHES,
                         stage_name="stage2")

    print("Best overall:", best_overall)

    end_time = time.time()  # Record the end time

    total_time = end_time - start_time
    print("Total training time: {:.2f} seconds".format(total_time))

    # sys.stdout.flush()
    # time.sleep(30)
    # os.system("/usr/bin/shutdown")
