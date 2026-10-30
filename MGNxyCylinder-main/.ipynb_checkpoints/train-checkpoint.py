from dataset import FPC
from model.simulator import Simulator
import torch
from utils.noise import get_velocity_noise
import time
import os
import sys
from utils.utils import NodeType
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T


dataset_dir = "dataset/data_downsampled"
batch_size = 20
# noise_std=2e-2

print_batch = 100
save_batch = 100


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
simulator = Simulator(message_passing_num=15, node_input_size=11, edge_input_size=3, device=device)
optimizer= torch.optim.Adam(simulator.parameters(), lr=1e-4)
print('Optimizer initialized')


def train(model:Simulator, dataloader, optimizer):

    for batch_index, graph in enumerate(dataloader):

        graph = transformer(graph)
        graph = graph.cuda()

        node_type = graph.x[:, 0] #"node_type, cur_v, pressure, time"

        # 添加时间维度，假设序列长度为 1
        graph.x = graph.x.unsqueeze(1)  # [batch_size, seq_len=1, node_features]

        # velocity_sequence_noise = get_velocity_noise(graph, noise_std=noise_std, device=device)
        predicted_acc, target_acc = model(graph)

        # 去掉时间维度，恢复原始形状
        predicted_acc = predicted_acc.squeeze(1)
        target_acc = target_acc.squeeze(1)
        
        mask = torch.logical_or(node_type==NodeType.NORMAL, node_type==NodeType.OUTFLOW)
        
        errors = ((predicted_acc - target_acc)**2)[mask]
        loss = torch.mean(errors)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % print_batch == 0:
            print('batch %d [loss %.2e]'%(batch_index, loss.item()))

        if batch_index % save_batch == 0:
            model.save_checkpoint()

if __name__ == '__main__':

    dataset_fpc = FPC(
        max_epochs=50,
        dataset_dir=dataset_dir,
        split="train",
        tra_len=200,
        time_iterval=0.03,
    )
    train_loader = DataLoader(dataset=dataset_fpc, batch_size=batch_size, num_workers=10)
    transformer = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])

    start_time = time.time()  # Record the start time
    train(simulator, train_loader, optimizer)
    end_time = time.time()  # Record the end time

    total_time = end_time - start_time
    print("Total training time: {:.2f} seconds".format(total_time))

    sys.stdout.flush()
    time.sleep(30)
    os.system("/usr/bin/shutdown")
