import torch
from utils.utils import NodeType


def Mach_re_full(graph, noise: torch.Tensor) -> torch.Tensor:
    # Mach non-dimension
    # [rho, v1, v2, pres]
    mach = graph.x[:, 5]

    vm = mach.clone().detach()

    noise[:, 0] = noise[:, 0] * vm
    noise[:, 1] = noise[:, 1] * vm
    noise[:, 3] = noise[:, 3] * (vm * vm)

    return noise


def get_noise(graph, mask, noise_std, device):
    sequence = graph.x[:, 1:5]

    noise = torch.randn_like(sequence) * noise_std.to(device)

    # noise = Mach_re_full(graph, noise)

    noise[mask, :] = 0

    return noise.to(device)


def get_noise_rel(graph, mask, noise_std, device):
    """
    Relative Gaussian jitter on NORMAL nodes only.
    noise_std: tensor of shape [4] giving relative std per feature.
    """
    # Extract the 4 state features
    sequence = graph.x[:, 1:5]                          # [N,4]

    # Compute relative noise: σ_i * |x_i| * N(0,1)
    rel_noise = torch.randn_like(sequence) * (noise_std.to(device) * sequence.abs())

    # Zero out noise on non‑NORMAL nodes
    rel_noise[mask, :] = 0

    return rel_noise.to(device)
