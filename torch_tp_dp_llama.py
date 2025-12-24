#!/usr/bin/env python3

import os
import sys
import argparse
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    PrepareModuleInput,
    parallelize_module,
)
from torch.distributed.tensor import Shard, Replicate
from torch.distributed.tensor.parallel.ddp import _pre_dp_module_transform
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import timedelta


# Command:
# QAIC_VISIBLE_DEVICES=60,61,62,63 torchrun --nproc_per_node=4 --master-port=1234 torch_tp_dp_v2.py --tp 2 --dp 2 --device qaic
# Facing below error when running on QAIC:
# [rank2]: RuntimeError: Backend qccl does not support reduce_scatter_tensor_coalesced

# Code reference:
# 1. https://github.com/pytorch/pytorch/blob/7de041cb5a5817500b973eb32a70325187a83407/test/distributed/_composable/test_composability/test_2d_composability.py#L478
# 2. Torchtitan for llama parallelisation code

def setup_distributed(device_type: str):
    """Initialize distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if world_size == 1:
            raise RuntimeError("Please initiate distributed training via torchrun.")

        backend = "qccl" if device_type == "qaic" else "nccl"
        dist.init_process_group(backend=backend, timeout=timedelta(seconds=60))
        if device_type == "qaic":
            torch.qaic.set_device(local_rank)
        elif device_type == "cuda":
            torch.cuda.set_device(local_rank)
        else:
            raise ValueError(f"Unsupported device type: {device_type}")
        device = torch.device(f"{device_type}:{local_rank}")
    else:
        raise RuntimeError(
            "This script must be run with torchrun. Example:\n"
            "torchrun --nproc_per_node=8 torch_tp_dp_llama.py --device qaic --tp 2 --dp 2"
        )

    return device, rank, world_size


def verify_model_structure(model):
    """Verify that the model has the expected Llama structure."""
    has_embed_tokens = hasattr(model, "model") and hasattr(model.model, "embed_tokens")
    has_norm = hasattr(model, "model") and hasattr(model.model, "norm")
    has_layers = hasattr(model, "model") and hasattr(model.model, "layers")
    has_lm_head = hasattr(model, "lm_head")

    if not has_embed_tokens:
        print("Warning: model.model.embed_tokens not found")
    if not has_norm:
        print("Warning: model.model.norm not found")
    if not has_layers:
        print("Warning: model.model.layers not found")
    if not has_lm_head:
        print("Warning: model.lm_head not found")

    return has_embed_tokens and has_norm and has_layers and has_lm_head


def create_llama_parallelize_plan(use_sequence_parallel=True, loss_parallel=False):
    """
    Create parallelize plan for Llama 3.2 architecture (HuggingFace format).
    """
    plan = {}

    # Root-level plan
    root_plan = {}

    # Embedding layer
    root_plan["model.embed_tokens"] = RowwiseParallel(
        input_layouts=Replicate(),
        output_layouts=Shard(1) if use_sequence_parallel else Replicate(),
    )

    # Final norm layer
    if use_sequence_parallel:
        root_plan["model.norm"] = SequenceParallel()

    # Output head
    root_plan["lm_head"] = ColwiseParallel(
        input_layouts=Shard(1) if use_sequence_parallel else Replicate(),
        output_layouts=Shard(-1) if loss_parallel else Replicate(),
        use_local_output=not loss_parallel,
    )

    plan.update(root_plan)

    # Layer-level plan
    layer_plan = {}

    if use_sequence_parallel:
        layer_plan["model.layers.*.input_layernorm"] = SequenceParallel()
        layer_plan["model.layers.*.post_attention_layernorm"] = SequenceParallel()

    layer_plan["model.layers.*.self_attn.q_proj"] = ColwiseParallel()
    layer_plan["model.layers.*.self_attn.k_proj"] = ColwiseParallel()
    layer_plan["model.layers.*.self_attn.v_proj"] = ColwiseParallel()

    if use_sequence_parallel:
        layer_plan["model.layers.*.self_attn.o_proj"] = RowwiseParallel(
            output_layouts=Shard(1)
        )
        layer_plan["model.layers.*.self_attn"] = PrepareModuleInput(
            input_layouts=(Shard(1), None, None, None),
            desired_input_layouts=(Replicate(), None, None, None),
        )
    else:
        layer_plan["model.layers.*.self_attn.o_proj"] = RowwiseParallel()

    layer_plan["model.layers.*.mlp.gate_proj"] = ColwiseParallel()
    layer_plan["model.layers.*.mlp.up_proj"] = ColwiseParallel()

    if use_sequence_parallel:
        layer_plan["model.layers.*.mlp"] = PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        )
        layer_plan["model.layers.*.mlp.down_proj"] = RowwiseParallel(
            output_layouts=Shard(1)
        )
    else:
        layer_plan["model.layers.*.mlp.down_proj"] = RowwiseParallel()

    plan.update(layer_plan)
    return plan


def setup_tp_ddp_model(
    model,
    tp_size: int,
    dp_size: int,
    device_type: str,
    use_sequence_parallel: bool = True,
    use_loss_parallel: bool = False,
):
    """
    Set up Llama model with Tensor Parallelism + DDP.
    """
    world_size = dist.get_world_size()

    if world_size % tp_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by tp_size ({tp_size})"
        )

    mesh_2d = init_device_mesh(
        device_type,
        (dp_size, tp_size),
        mesh_dim_names=("dp", "tp"),
    )

    tp_mesh = mesh_2d["tp"]
    dp_mesh = mesh_2d["dp"]
    dp_pg = mesh_2d.get_group(mesh_dim=0)

    if dist.get_rank() == 0:
        print(f"Device mesh created: DP={dp_size}, TP={tp_size}")
        print(f"TP mesh size: {tp_mesh.size()}, DP mesh size: {dp_mesh.size()}")

    verify_model_structure(model)

    parallelize_plan = create_llama_parallelize_plan(
        use_sequence_parallel=use_sequence_parallel,
        loss_parallel=use_loss_parallel,
    )

    model = parallelize_module(model, tp_mesh, parallelize_plan)
    _pre_dp_module_transform(model)

    if device_type == "qaic":
        device_ids = [dist.get_rank() % torch.qaic.device_count()]
    elif device_type == "cuda":
        device_ids = [dist.get_rank() % torch.cuda.device_count()]
    else:
        raise RuntimeError("Unsupported device type.")

    model = DDP(model, process_group=dp_pg, device_ids=device_ids)

    return model, dp_pg, mesh_2d, dp_size


def fine_tune_llama32(args):
    """
    Fine-tuning loop for Llama 3.2 1B using TP+DDP.
    """
    device, rank, world_size = setup_distributed(args.device)

    model_name = "meta-llama/Llama-3.2-1B"
    tp_size = args.tp
    dp_size = args.dp
    use_sequence_parallel = True
    use_loss_parallel = False

    if rank == 0:
        print(f"Initializing Llama 3.2 1B fine-tuning")
        print(f"Model: {model_name}")
        print(f"World size: {world_size}")
        print(f"TP size: {tp_size}, DP size: {dp_size}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=None,
    ).to(device)

    model, dp_pg, mesh_2d, dp_size = setup_tp_ddp_model(
        model,
        tp_size=tp_size,
        dp_size=dp_size,
        device_type=args.device,
        use_sequence_parallel=use_sequence_parallel,
        use_loss_parallel=use_loss_parallel,
    )

    if rank == 0:
        print(f"Model parallelized:")
        print(f"  - Data Parallel groups: {dp_size}")
        print(f"  - Tensor Parallel size: {tp_size}")
        print(f"  - Sequence Parallel: {use_sequence_parallel}")
        print(f"  - Loss Parallel: {use_loss_parallel}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    model.train()
    num_epochs = 3
    batch_size = 2

    if rank == 0:
        print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        dp_rank = dist.get_rank(dp_pg)

        texts = [
            f"Example text {i} for DP rank {dp_rank} in epoch {epoch}"
            for i in range(batch_size)
        ]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        outputs = model(**inputs, labels=inputs["input_ids"])

        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if dist.get_rank(dp_pg) == 0:
            tp_rank = mesh_2d.get_local_rank("tp")
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"DP Group {dp_rank} | "
                f"TP Rank {tp_rank} | "
                f"Loss: {loss.item():.4f}"
            )

    if rank == 0:
        print("\nTraining completed!")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLaMA fine-tuning with TP+DDP")
    parser.add_argument("--device", type=str, required=True, choices=["cuda", "qaic"],
                        help="Device type: 'cuda' or 'qaic'")
    parser.add_argument("--tp", type=int, required=True, help="Tensor Parallelism Size")
    parser.add_argument("--dp", type=int, required=True, help="Data Parallelism Size")
    args = parser.parse_args()

    fine_tune_llama32(args)