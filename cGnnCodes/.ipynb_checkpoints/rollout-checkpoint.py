from dataset.fpcMulti import FPC_ROLLOUT
from torch_geometric.loader import DataLoader
import torch
import argparse
from tqdm import tqdm
import h5py
import torch_geometric.transforms as T
from utils.utils import NodeType
from utils.to_undirected import make_edges_undirected
import numpy as np
import os

from model.simulator import Simulator


parser = argparse.ArgumentParser(description='2D Euler')
parser.add_argument("--gpu",
                    type=int,
                    default=0,
                    help="gpu number: 0 or 1")

parser.add_argument("--model_dir",
                    type=str,
                    default='checkpoint/simulator.pth')

parser.add_argument("--test_split", type=str, default="test")

parser.add_argument("--rollout_num", type=int, default=20)

parser.add_argument("--total_steps", type=int, default=80)

args = parser.parse_args()

# gpu devices
torch.cuda.set_device(args.gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def rollout_error(predicteds, targets):
    """
    Args:
        predicteds: np.ndarray, shape (T, N, 4)
        targets:    np.ndarray, shape (T, N, 4)
        
    Returns:
        cumulative_rmse:     np.ndarray, shape (T, 4)
        avg_one_step_rmse:   np.ndarray, shape (4,)
    """
    T, N, F = targets.shape

    # squared error: shape (T, N, 4)
    sq_diff = (predicteds - targets) ** 2

    # MSE per time step *per variable* (mean over nodes only)
    # -> shape (T, 4)
    mse_per_t = sq_diff.mean(axis=1)

    # cumulative mean MSE up to each time step
    # cumsum over time, divide by step count -> shape (T,4)
    cum_mse = np.cumsum(mse_per_t, axis=0)
    counts = np.arange(1, T + 1)[:, None]  # shape (T,1)
    avg_cum_mse = cum_mse / counts

    # take square root -> cumulative RMSE per variable
    cumulative_rmse = np.sqrt(avg_cum_mse)  # shape (T,4)

    # overall one-step MSE per variable (average over all timesteps)
    avg_mse = mse_per_t.mean(axis=0)        # shape (4,)
    avg_one_step_rmse = np.sqrt(avg_mse)    # shape (4,)

    # pretty‐print
    var_names = ['rho', 'v1', 'v2', 'pres']
    for t in range(T):
        vals = ", ".join(f"{var_names[k]}={cumulative_rmse[t,k]:.2e}"
                         for k in range(F))
        print(f"Cumulative RMSE @ step {t}: {vals}")
    overall = ", ".join(f"{var_names[k]}={avg_one_step_rmse[k]:.2e}"
                        for k in range(F))
    print(f"Average One-Step RMSE: {overall}")

    return cumulative_rmse, avg_one_step_rmse


@torch.no_grad()
def rollout(model, dataloader, total_steps, rollout_index=1):

    dataset.change_file(rollout_index)

    predicted_prims = None
    mask = None
    predicteds = []
    targets = []

    for graph in tqdm(dataloader, total=total_steps):

        # copy directed (unique) edge_index and to_undirected
        graph = make_edges_undirected(graph)

        graph = transformer(graph)
        graph = graph.cuda()

        if mask is None:
            node_type = graph.x[:, 0]
            # ghost marker to update node_type
            # node_type = mark_ghost_nodes(node_type, graph.edge_index, num_layers=3)
            mask = (node_type == NodeType.NORMAL)
            mask = torch.logical_not(mask)

        next_v = graph.y

        if predicted_prims is not None:
            graph.x[:, 1:5] = predicted_prims.detach()

        graph.x[mask, 1:5] = next_v[mask]

        predicted_prims = model(graph, sequence_noise=None)

        predicted_prims[mask] = next_v[mask]

        predicteds.append(predicted_prims.detach().cpu().numpy())
        targets.append(next_v.detach().cpu().numpy())

    result = [np.stack(predicteds), np.stack(targets)]

    os.makedirs("result", exist_ok=True)
    with h5py.File("result/" + str(rollout_index + 1) + ".h5", "w") as f:
        f.create_dataset("predicteds", data=result[0])
        f.create_dataset("targets", data=result[1])

    return result


if __name__ == "__main__":

    simulator = Simulator(message_passing_num=12, node_input_size=6, edge_input_size=5, device=device)
    simulator.load_checkpoint()
    simulator.eval()

    dataset_dir = "dataset/data_downsampled"
    dataset = FPC_ROLLOUT(dataset_dir, split=args.test_split)
    test_loader = DataLoader(dataset=dataset, batch_size=1)

    transformer = T.Compose([T.NormalizeScale(), T.Cartesian(norm=False), T.Distance(norm=False)])

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
    cumulative_rmse_array = np.array(cumulative_rmse_list)      # Shape: (rollout_num, T, 4)
    avg_one_step_rmse_array = np.array(avg_one_step_rmse_list)  # Shape: (rollout_num, 4)

    # Save all per-trajectory metrics into a single file
    per_trajectory_metrics_file = "metrics/per_trajectory_metrics.h5"
    with h5py.File(per_trajectory_metrics_file, "w") as f:
        f.create_dataset("cumulative_rmse", data=cumulative_rmse_array)
        f.create_dataset("avg_one_step_rmse", data=avg_one_step_rmse_array)

    # Compute the average cumulative_rmse and avg_one_step_rmse across all trajectories
    average_cumulative_rmse    = np.mean(cumulative_rmse_array,    axis=0)  # Shape: (T, 4)
    average_avg_one_step_rmse  = np.mean(avg_one_step_rmse_array,  axis=0)  # Shape: (4,)

    # Save these average metrics into a file
    average_metrics_file = "metrics/average_metrics.h5"
    with h5py.File(average_metrics_file, "w") as f:
        f.create_dataset("average_cumulative_rmse",    data=average_cumulative_rmse)
        f.create_dataset("average_avg_one_step_rmse",  data=average_avg_one_step_rmse)

    # print the overall average metrics
    print("==================================================================")
    var_names = ['rho', 'v1', 'v2', 'pres']

    print("Average Cumulative RMSE over all trajectories:")
    for step, rmse_vec in enumerate(average_cumulative_rmse):
        vals = ", ".join(f"{var_names[k]}={rmse_vec[k]:.6e}" for k in range(4))
        print(f"Step {step}: {vals}")

    overall = ", ".join(f"{var_names[k]}={average_avg_one_step_rmse[k]:.6e}" 
                        for k in range(4))
    print("Overall Average One-Step RMSE over all trajectories:", overall)
