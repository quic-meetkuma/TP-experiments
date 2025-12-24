#!/usr/bin/env python3

import os
import sys
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.distributed.tensor.parallel.ddp import _pre_dp_module_transform
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh


# Command:
# QAIC_VISIBLE_DEVICES=60,61,62,63 torchrun --nproc_per_node=4 --master-port=1234 torch_tp_dp_v2.py --tp 2 --dp 2 --device qaic
# This code runs fine on QAIC.

# Code reference:
# 1. https://github.com/pytorch/pytorch/blob/7de041cb5a5817500b973eb32a70325187a83407/test/distributed/_composable/test_composability/test_2d_composability.py#L478

# Define your model
class MLPModule(torch.nn.Module):
    def __init__(self, device, bias=True):
        super().__init__()
        torch.manual_seed(5)
        self.net1 = torch.nn.Linear(10, 16, bias=bias, device=device)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(16, 4, bias=bias, device=device)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

    def reset_parameters(self):
        self.net1.reset_parameters()
        self.net2.reset_parameters()


def main():
    parser = argparse.ArgumentParser(description="Run TP + DP with PyTorch Distributed")
    parser.add_argument("--device", type=str, required=True, choices=["cuda", "qaic"],
                        help="Device type: 'cuda' or 'qaic'")
    parser.add_argument("--tp", type=int, required=True, help="Tensor Parallelism Degree")
    parser.add_argument("--dp", type=int, required=True, help="Data Parallelism Degree")
    args = parser.parse_args()

    device_type = args.device
    tp_degree = args.tp
    dp_degree = args.dp

    # Ensure WORLD_SIZE matches tp * dp
    world_size = int(os.getenv("WORLD_SIZE", 1))
    assert world_size == tp_degree * dp_degree, f"WORLD_SIZE ({world_size}) must equal tp * dp ({tp_degree} * {dp_degree})"

    rank = int(os.getenv("LOCAL_RANK", 0))
    backend = "qccl" if device_type == "qaic" else "nccl"

    # Initialize distributed process group
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
        device_id=rank,
    )

    if device_type == "qaic":
        torch.qaic.set_device(rank)
    elif device_type == "cuda":
        torch.cuda.set_device(rank)

    # Create model on specified device
    torch.manual_seed(0)
    model = MLPModule(device_type)

    # Setup 2D device mesh
    mesh_2d = init_device_mesh(
        device_type,
        (dp_degree, tp_degree),
        mesh_dim_names=("dp", "tp"),
    )

    # Parallelization plan
    parallelize_plan = {
        "net1": ColwiseParallel(),
        "net2": RowwiseParallel(),
    }

    # Apply tensor parallelism
    model = parallelize_module(model, mesh_2d["tp"], parallelize_plan)
    _pre_dp_module_transform(model)

    # Wrap with DDP using data parallel group
    dp_pg = mesh_2d.get_group(mesh_dim=0)
    model = DDP(model, process_group=dp_pg)

    # Dummy input
    dummy_input = torch.randn(32, 10, device=device_type)
    output = model(dummy_input)
    print(f"[Rank {rank}] Output shape: {output.shape}")

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()