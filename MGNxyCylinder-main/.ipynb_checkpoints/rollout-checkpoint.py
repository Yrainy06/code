from dataset import FPC_ROLLOUT
from torch_geometric.loader import DataLoader
import torch
import argparse
from tqdm import tqdm
import h5py
import torch_geometric.transforms as T
from utils.utils import NodeType
import numpy as np
from model.simulator import SimulatorWithLSTM
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

parser.add_argument("--predict_steps", type=int, default=1)  # <<< 每次预测几步

args = parser.parse_args()

# gpu devices
torch.cuda.set_device(args.gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def rollout_error(predicteds, targets):
    
    number_len = targets.shape[0]
    squared_diff = np.square(predicteds - targets)  # Shape: (number_len, num_nodes, num_features)
    
    mse_per_timestep = np.mean(squared_diff, axis=(1, 2))  # Shape: (number_len,)
    
    # cumulative RMSE up to each time step
    cumulative_mse = np.cumsum(mse_per_timestep)
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

def fix_x(x, target_dim=11):
    """将输入特征 x 调整到 target_dim 维（不足时补零；多余时截断）"""
    if x.size(1) < target_dim:
        pad = torch.zeros(x.size(0), target_dim - x.size(1), device=x.device)
        x = torch.cat([x, pad], dim=1)
    elif x.size(1) > target_dim:
        x = x[:, :target_dim]
    return x

@torch.no_grad()
def rollout(model, dataloader, total_steps, rollout_index=1):
    # 根据 rollout_index 切换当前轨迹
    dataset.change_file(rollout_index)

    predicteds = []
    targets = []
    # hidden = None
    
    # 从原始图中提取 pos 与 face（避免 transformer 修改它们）
    data_iter = iter(dataloader)
    raw_first_graph = next(data_iter)

    crds = raw_first_graph.pos.cpu().numpy()
    if hasattr(raw_first_graph, 'face') and (raw_first_graph.face is not None):
        face = raw_first_graph.face.cpu().numpy()
    else:
        face = None

    # 初始化速度窗口：使用 future_velocities 的第一个时刻，否则采用目标速度 y
    init_future_vel = getattr(raw_first_graph, 'future_velocities', None)
    if init_future_vel is not None:
        init_vel = init_future_vel[:, 0, :]  # shape: [num_nodes, 2]
    else:
        init_vel = raw_first_graph.y  # shape: [num_nodes, 2]
    init_vel = init_vel.unsqueeze(1)  # shape: [num_nodes, 1, 2]
    velocity_window = init_vel.repeat(1, model.lstm_window, 1)
    # 将速度窗口移动到 device 上
    velocity_window = velocity_window.to(device)

    # 对 raw_first_graph 应用转换（这里只调用 FaceToEdge），随后恢复原始节点特征
    transformed_first = T.FaceToEdge()(raw_first_graph).to(device)
    transformed_first.x = fix_x(raw_first_graph.x.to(device), target_dim=11)
    if transformed_first.edge_attr is None:
        num_edges = transformed_first.edge_index.size(1)
        transformed_first.edge_attr = torch.zeros(num_edges, 3, device=device)

    target_velocity = transformed_first.y
    predicted_velocity = model(transformed_first, velocity_window, predict_steps=1)
    # predicted_velocity = predicted_velocity.detach()
    # predicted_velocity, hidden = model(transformed_first, velocity_window, predict_steps=1, hidden=hidden)
    velocity_window = torch.cat([velocity_window[:, 1:, :], predicted_velocity.unsqueeze(1)], dim=1)
    predicteds.append(predicted_velocity.detach().cpu().numpy())
    targets.append(target_velocity.detach().cpu().numpy())

    for raw_graph in tqdm(data_iter, total=total_steps - 1):
        raw_x = fix_x(raw_graph.x.to(device), target_dim=11)
        transformed_graph = T.FaceToEdge()(raw_graph).to(device)
        transformed_graph.x = raw_x  # 恢复调整后的原始节点特征
        if transformed_graph.edge_attr is None:
            num_edges = transformed_graph.edge_index.size(1)
            transformed_graph.edge_attr = torch.zeros(num_edges, 3, device=device)
        target_velocity = transformed_graph.y
        predicted_velocity = model(transformed_graph, velocity_window, predict_steps=1)
        # predicted_velocity = predicted_velocity.detach()
        # predicted_velocity, hidden = model(transformed_graph, velocity_window, predict_steps=1, hidden=hidden)
        velocity_window = torch.cat([velocity_window[:, 1:, :], predicted_velocity.unsqueeze(1)], dim=1)
        predicteds.append(predicted_velocity.detach().cpu().numpy())
        targets.append(target_velocity.detach().cpu().numpy())

    result = [np.stack(predicteds), np.stack(targets)]
    os.makedirs("result", exist_ok=True)
    with h5py.File("result/result" + str(rollout_index) + ".h5", "w") as f:
        f.create_dataset("predicted_velocity", data=result[0])
        f.create_dataset("target_velocity", data=result[1])
        f.create_dataset("coordinates", data=crds)
        if face is not None:
            f.create_dataset("face", data=face)
    return result

if __name__ == "__main__":

    simulator = SimulatorWithLSTM(message_passing_num=10, node_input_size=11, edge_input_size=3, device=device)
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