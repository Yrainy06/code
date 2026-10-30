from dataset import FPC_ROLLOUT
from torch_geometric.loader import DataLoader
import torch
import argparse
from tqdm import tqdm
import h5py
import torch_geometric.transforms as T
from utils.utils import NodeType
import numpy as np
from model.simulator_GNN_LSTM import Simulator
import os


parser = argparse.ArgumentParser(description='Implementation of MeshGraphNets')
parser.add_argument("--gpu",
                    type=int,
                    default=0,
                    help="gpu number: 0 or 1")

parser.add_argument("--model_dir",
                    type=str,
                    default='checkpoint/simulator.pth')

parser.add_argument("--test_split", type=str, default="test")

parser.add_argument("--rollout_num", type=int, default=100)
parser.add_argument("--total_steps", type=int, default=100)

args = parser.parse_args()

# gpu devices
torch.cuda.set_device(args.gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def rollout_error(predicteds, targets):

    number_len = targets.shape[0]
    squared_diff = np.square(predicteds - targets)  # Shape: (number_len, num_nodes, num_features)

    # MSE at each time step (mean over nodes and features)
    mse_per_timestep = np.mean(squared_diff, axis=(1, 2))  # Shape: (number_len,)

    # cumulative RMSE up to each time step
    cumulative_mse = np.cumsum(mse_per_timestep)  # Shape: (number_len,)
    avg_cumulative_mse = cumulative_mse / np.arange(1, number_len + 1)
    cumulative_rmse = np.sqrt(avg_cumulative_mse)

    # overall average one-step RMSE
    avg_mse = np.mean(mse_per_timestep)
    avg_one_step_rmse = np.sqrt(avg_mse)

    for show_step in range(0, 10000, 1):
        if show_step < number_len:
            print("Cumulative RMSE @ step %d loss: %.2e" % (show_step, cumulative_rmse[show_step]))
        else:
            break

    print("Average One-Step RMSE over all time steps: %.2e" % avg_one_step_rmse)

    return cumulative_rmse, avg_one_step_rmse


@torch.no_grad()
def rollout(model, dataloader, total_steps, rollout_index=1):

    dataset.change_file(rollout_index)

    predicted_velocity = None
    mask = None
    predicteds = []
    targets = []

    crds = None
    face = None

    hidden = None  # ← 初始化 LSTM hidden state

    # Reset hidden state matrix for each layer
    # model.model.H = [None] * len(model.model.g_rnns)

    for step, graph in enumerate(tqdm(dataloader, total=total_steps)):

        if step % 5 == 0:  # 每 5 步清除 LSTM 隐状态
            hidden = None

        # Extract crds and face before transformer modifies the graph
        if crds is None:
            crds = graph.pos.cpu().numpy()
            face = graph.face.cpu().numpy()

        graph = transformer(graph)
        graph = graph.cuda()

        if mask is None:
            node_type = graph.x[:, 0]
            mask = torch.logical_or(node_type == NodeType.NORMAL, node_type == NodeType.OUTFLOW)
            mask = torch.logical_not(mask)

        if predicted_velocity is not None:
            graph.x[:, 1:3] = predicted_velocity.detach()
            # graph.x[:, 1:3] = predicted_velocity

        next_v = graph.y
        predicted_velocity, hidden = model(graph, velocity_sequence_noise=None, hidden=hidden)

        predicted_velocity[mask] = next_v[mask]

        predicteds.append(predicted_velocity.detach().cpu().numpy())
        targets.append(next_v.detach().cpu().numpy())

    result = [np.stack(predicteds), np.stack(targets)]

    os.makedirs("result", exist_ok=True)
    with h5py.File("result/result" + str(rollout_index) + ".h5", "w") as f:
        f.create_dataset("predicted_velocity", data=result[0])
        f.create_dataset("target_velocity", data=result[1])
        f.create_dataset("coordinates", data=crds)
        f.create_dataset("face", data=face)

    return result


if __name__ == "__main__":

    simulator = Simulator(message_passing_num=10, node_input_size=11, edge_input_size=3, device=device)
    simulator.load_checkpoint()
    simulator.eval()

    dataset_dir = "dataset/data_downsampled"
    dataset = FPC_ROLLOUT(dataset_dir, split=args.test_split)
    transformer = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])
    test_loader = DataLoader(dataset=dataset, batch_size=1)

    # Create lists to store metrics from each trajectory
    cumulative_rmse_list = []
    avg_one_step_rmse_list = []
    os.makedirs("metrics", exist_ok=True)

    for i in range(args.rollout_num):
        result = rollout(simulator, test_loader, args.total_steps, rollout_index=i)
        print("------------------------------------------------------------------")
        cumulative_rmse, avg_one_step_rmse = rollout_error(result[0], result[1])

        cumulative_rmse_list.append(cumulative_rmse)
        avg_one_step_rmse_list.append(avg_one_step_rmse)

    # After processing all trajectories, convert the lists to arrays
    cumulative_rmse_array = np.array(cumulative_rmse_list)  # Shape: (rollout_num, number_len)
    avg_one_step_rmse_array = np.array(avg_one_step_rmse_list)  # Shape: (rollout_num,)

    # Save all per-trajectory metrics into a single file
    per_trajectory_metrics_file = "metrics/per_trajectory_metrics.h5"
    with h5py.File(per_trajectory_metrics_file, "w") as f:
        f.create_dataset("cumulative_rmse", data=cumulative_rmse_array)
        f.create_dataset("avg_one_step_rmse", data=avg_one_step_rmse_array)

    # Compute the average cumulative_rmse and avg_one_step_rmse across all trajectories
    average_cumulative_rmse = np.mean(cumulative_rmse_array, axis=0)  # Shape: (number_len,)
    average_avg_one_step_rmse = np.mean(avg_one_step_rmse_array)  # Scalar

    # Save these average metrics into a file
    average_metrics_file = "metrics/average_metrics.h5"
    with h5py.File(average_metrics_file, "w") as f:
        f.create_dataset("average_cumulative_rmse", data=average_cumulative_rmse)
        f.create_dataset("average_avg_one_step_rmse", data=average_avg_one_step_rmse)

    # print the overall average metrics
    print("==================================================================")
    print("Average Cumulative RMSE over all trajectories:")
    for step, rmse in enumerate(average_cumulative_rmse):
        print(f"Step {step}: {rmse:.6e}")

    print(
        "Overall Average One-Step RMSE over all trajectories: %.6e"
        % average_avg_one_step_rmse
    )