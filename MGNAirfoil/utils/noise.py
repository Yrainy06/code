import torch
from utils.utils import NodeType

def get_noise(graph, noise_std, device):
    sequence = graph.x[:, 1:4]
    type = graph.x[:, 0]
    noise = torch.randn_like(sequence) * noise_std.to(device)
    mask = type!=NodeType.NORMAL
    noise[mask, :] = 0
    return noise.to(device)
