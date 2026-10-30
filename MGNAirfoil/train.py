from dataset import FPC
from model.simulator import Simulator
import torch
from utils.noise import get_noise
import time
import os
import sys
from utils.utils import NodeType
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T


dataset_dir = "dataset/data_downsampled"
batch_size = 15
noise_std = torch.tensor([10.0, 10.0, 1e-2])

print_batch = 20
save_batch = 20


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
simulator = Simulator(message_passing_num=15, node_input_size=12, edge_input_size=3, device=device)
optimizer= torch.optim.Adam(simulator.parameters(), lr=1e-4)
print('Optimizer initialized')


def train(model:Simulator, dataloader, optimizer):

    for batch_index, graph in enumerate(dataloader):

        graph = transformer(graph)
        graph = graph.cuda()

        node_type = graph.x[:, 0] #"node_type, cur_v, pressure, time"
        sequence_noise = get_noise(graph, noise_std=noise_std, device=device)
        predicted_acc, target_acc = model(graph, sequence_noise)
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
        tra_len=120,
        time_iterval=0.04,
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
