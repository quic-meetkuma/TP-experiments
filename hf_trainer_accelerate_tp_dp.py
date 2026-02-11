import os
import torch
import argparse
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    AutoConfig,
)
from transformers import DataCollatorWithPadding
from accelerate.utils import ParallelismConfig
from torch.distributed.tensor.parallel.ddp import _pre_dp_module_transform
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from transformers.integrations.tensor_parallel import distribute_model
from peft import get_peft_model
from peft import LoraConfig
from transformers.integrations.tensor_parallel import (
    replace_layer_number_by_wildcard,
    ALL_PARALLEL_STYLES,
)

# Command:
# QAIC_VISIBLE_DEVICES=32,33,34,35 torchrun --nproc_per_node=4 --master-port=1234 hf_trainer_accelerate_tp_dp.py --force_device qaic --tp_size 2 --dp_size 2

# Status as of 8th Jan, 2026
# Only DDP:
#   Command: QAIC_VISIBLE_DEVICES=32,33,34,35 torchrun --nproc_per_node=2 --master-port=1234 hf_trainer_accelerate_tp_dp.py --force_device qaic --dp_size 2
#   Status: Works fine without any issues.
# Only TP:
#   Command: QAIC_VISIBLE_DEVICES=32,33,34,35 torchrun --nproc_per_node=2 --master-port=1234 hf_trainer_accelerate_tp_dp.py --force_device qaic --tp_size 2
#   Status: Works fine without any issues
# TP+DDP:
#   Command: QAIC_VISIBLE_DEVICES=32,33,34,35 torchrun --nproc_per_node=4 --master-port=1234 hf_trainer_accelerate_tp_dp.py --force_device qaic --tp_size 2 --dp_size 2
#   Status:
#       - With max_grad_norm set to 1.0 (default) value, we are getting attempting to unscale already unscaled params.
#         If we set max_grad_norm to None then we can get rid of that error.
#       - But with max_grad_norm set to None, we are now getting error "AssertionError: No inf checks were recorded prior to update.""
#       - TODO: Apart from above issue, need to make sure that the grads are divided by DDP rank rather than world size.
#               (Reference: https://github.com/meta-pytorch/torchtune/blob/44271b570af36cfda8ee20a4479d2652770378c0/recipes/full_finetune_distributed.py#L1037)

# Status as of 9th Jan, 2026
# TP+DDP:
#   Command: QAIC_VISIBLE_DEVICES=32,33,34,35 torchrun --nproc_per_node=4 --master-port=1234 hf_trainer_accelerate_tp_dp.py --force_device qaic --tp_size 2 --dp_size 2
#   Status:
#       - The previous issues were observed on GPU with accelerate==1.12.0 and transformers==4.57.3. These are installed
#         from open source. We now have a internal fork of accelerate==1.12.0 and transformers=5.0.0 with qaic backend changes.
#         Installing both from internal fork resolves the "No inf checks were recorded prior to update." issue. The issue with
#         max_grad_norm still remains to be debugged.
#       - This means now TP+DP works fine on QAIC as well as GPU with the internal fork of accelerate and transformers.
#   Note:
#       - Internal fork of transformers: https://github.com/quic-meetkuma/transformers/tree/qaic_support_transformer_20_12_2025 (Commit id: 9cd1f690c95cb526600dd0d4ab32bf7d4a58d720)
#       - Internal fork of accelerate: https://github.com/quic-meetkuma/accelerate/tree/v1.12.0-release-shubham-changes-dp-tp (Commit id: 4ebcbddc01be1b7441fc1ee9ba9b9fd474fdcb14)
#       - Use init.sh for installing the internal forks.

# Status as of 10th Feb, 2026
# TP+DDP:
#   Command: QAIC_VISIBLE_DEVICES=32,33,34,35 torchrun --nproc_per_node=4 --master-port=1234 hf_trainer_accelerate_tp_dp.py --force_device qaic --tp_size 2 --dp_size 2 --apply_peft
#   Status:
#       - With PEFT modification, now TP+DP along with LoRA layer works fine on QAIC with the internal fork of accelerate and transformers.
#   Note:
#       - Internal fork of transformers: https://github.com/quic-meetkuma/transformers/tree/qaic_support_transformer_v5.1-release (Commit id: 94c99d7e936d24ba0677a80fdc2f7cc9fdb7fc87)
#       - Internal fork of accelerate: https://github.com/quic-meetkuma/accelerate/tree/10_02_26_shubham_changes_dp_tp (Commit id: 43fc8626244b7d9b05b7df1fc09fac2ad0bfda12)
#       - Use init.sh for installing the internal forks.


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run tensor parallel training on GPU or QAIC"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1b",
        help="Model name or path",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./tp_output", help="Output directory"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Per-device batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--force_device",
        type=str,
        choices=["cuda", "qaic", "cpu"],
        default="auto",
        help="Force specific device type (auto will use best available)",
    )
    parser.add_argument(
        "--bf16", action="store_true", help="Use bfloat16 precision if available"
    )
    parser.add_argument(
        "--tp_size", type=int, default=1, help="TP degree for tensor parallelism"
    )
    parser.add_argument(
        "--dp_size",
        type=int,
        default=1,
        help="DP degree for Distributed data parallelism",
    )
    parser.add_argument(
        "--apply_peft", action="store_true", help="Apply PEFT (LoRA) to the model."
    )
    return parser.parse_args()


def setup_parallelism(tp_size, dp_size):
    """Set up tensor parallelism configuration."""
    parallelism_config = {}
    if tp_size is not None:
        parallelism_config["tp_size"] = tp_size
    if dp_size is not None:
        parallelism_config["dp_replicate_size"] = dp_size

    pc = ParallelismConfig(**parallelism_config)
    return pc


def create_training_arguments(args, pc):
    """Create training arguments with appropriate settings."""
    return TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=args.epochs,
        logging_steps=1,
        fp16=True,
        parallelism_config=pc,
        ddp_find_unused_parameters=False,
        remove_unused_columns=True,
        max_grad_norm=None,  # This is the root cause for unscale called twice. Will solve it later.
    )


def print_trainable_parameters(model) -> None:
    """
    Print the number of trainable parameters, all params and percentage of trainablke params.
    Args:
        model: The PyTorch model.
    """
    trainable_params, all_param = model.get_nb_trainable_parameters()
    print(
        f"Trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
    )


def load_tokenizer(model_name):
    """Load and configure tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def is_rowwise_parallel(param: torch.distributed.tensor.DTensor) -> bool:
    """Check if a DTensor is row-wise parallel."""
    if not isinstance(param, torch.distributed.tensor.DTensor):
        return False
    placements = param.placements
    if len(placements) != 1:
        # Till now only TP is applied. If DP is also applied, then the placements might be of len 2.
        return False
    tp_placement = placements[0]
    return tp_placement.is_shard() and tp_placement.dim == 0  # Row-wise sharding


def is_colwise_parallel(param: torch.distributed.tensor.DTensor) -> bool:
    """Check if a DTensor is column-wise parallel."""
    if not isinstance(param, torch.distributed.tensor.DTensor):
        return False
    placements = param.placements
    if len(placements) != 1:
        # Till now only TP is applied. If DP is also applied, then the placements might be of len 2.
        return False
    tp_placement = placements[0]
    return tp_placement.is_shard() and tp_placement.dim == 1  # Column-wise sharding


def update_peft_tp_plan(model):
    # If original layer has colwise then Lora-A --> colwise and Lora-B --> rowwise
    # If original layer has rowwise then Lora-A --> rowwise and Lora-B --> colwise
    peft_tp_plan = {}
    for name, schema in model.tp_plan.items():
        lora_a_name = "base_model.model." + name + ".lora_A.default"
        lora_b_name = "base_model.model." + name + ".lora_B.default"
        if schema == "rowwise":
            peft_tp_plan[lora_a_name] = "rowwise"
            peft_tp_plan[lora_b_name] = "colwise"
        elif schema == "colwise":
            peft_tp_plan[lora_a_name] = "colwise"
            peft_tp_plan[lora_b_name] = "lora_rowwise"
    model.tp_plan.update(peft_tp_plan)


def apply_tp_modification_for_peft(model, tp_mesh=None):
    if tp_mesh is None:
        return

    state_dict = model.state_dict()
    for name, param in state_dict.items():
        if ("lora_A.default" in name) or ("lora_B.default" in name):
            name_for_tp = name.replace(".weight", "")
            name_for_tp = replace_layer_number_by_wildcard(name_for_tp)
            if name_for_tp not in model.tp_plan:
                raise RuntimeError(
                    f"{name_for_tp} not found in model.tp_plan. Please include PEFT layers in tp_plan."
                )
            lora_plan = model.tp_plan[name_for_tp]

            empty_param = param.clone().to(device="meta")
            tp_layer_cls = ALL_PARALLEL_STYLES[lora_plan].__class__
            tp_layer = tp_layer_cls(
                device_mesh=tp_mesh,
                rank=tp_mesh.get_local_rank(),
                empty_param=empty_param.clone(),
            )
            module_path, _, param_name = name.rpartition(".")
            module_obj = model.get_submodule(module_path)

            # prepare_module_tp does same thing as distribute_model. Hence commented out.
            # Ideal order of opeartion would be prepare_module_tp followed by shard_tensor based on what HF's tensor parallel code.
            # tp_layer.prepare_module_tp(module_obj, tp_mesh)

            # Shard the param
            tp_layer.shard_tensor(param, tensor_idx=None, dtype=empty_param.dtype)
            setattr(getattr(module_obj, param_name), "data", param)


def apply_peft_to_model(model, tp_mesh=None, peft_params=None):
    peft_config = LoraConfig(
        r=8,  # rank
        lora_alpha=16,  # scaling
        lora_dropout=0.05,  # dropout
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],  # typical for Llama; adjust per model
    )
    # Add PEFT adapters to the model
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)

    if tp_mesh is None:
        return

    # Include PEFT parameters in TP plan and update model.tp_plan inplace.
    update_peft_tp_plan(model)

    # Register pre-forward and post-forward hooks to convert input/output DTensor
    # to tensor and vice-versa.
    distribute_model(
        model,
        tp_plan=model.tp_plan,
        distributed_config=None,
        device_mesh=tp_mesh,
        tp_size=tp_mesh.size(),
    )

    # Convert PEFT weights from torch.Tensor to DTensor and apply TP modifications
    apply_tp_modification_for_peft(model, tp_mesh)


def load_model(model_name, device_mesh, apply_peft=False):
    """Load model with tensor parallelism."""
    tp_enabled = False
    dp_enabled = False
    dp_size = 1
    tp_size = 1
    tp_mesh = None
    if "dp_replicate" in device_mesh.mesh_dim_names:
        dp_enabled = True
        dp_mesh = device_mesh["dp_replicate"]
        dp_size = dp_mesh.size()
    if "tp" in device_mesh.mesh_dim_names:
        tp_enabled = True
        tp_mesh = device_mesh["tp"]
        tp_size = tp_mesh.size()

    assert (
        tp_enabled or dp_enabled
    ), "Either TP or DP must be enabled for this experiment. Both are disabled. Check your device mesh."
    print(
        f"Loading model {model_name} with tensor parallelism (tp_size={tp_size}) and data parallelism (dp_size={dp_size})"
    )

    kwargs = {}
    if tp_enabled:
        kwargs["tp_plan"] = "auto"
        kwargs["tp_size"] = tp_mesh.size()
        kwargs["device_mesh"] = tp_mesh

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        **kwargs,
    )
    # Need to explicitly untie the embedding weights here to consider
    # this as separate params in further TP processing
    model.lm_head.weight = nn.Parameter(model.lm_head.weight.clone())

    if apply_peft:
        # Apply PEFT to the model and include PEFT layers in TP plan
        apply_peft_to_model(model, tp_mesh=tp_mesh)

    return model


def create_dummy_dataset():
    """Create a dummy dataset for testing."""
    dummy_data = {
        "text": [
            "This is a sample sentence for training.",
            "Tensor parallelism is a cool technique.",
            "We need to test the training loop.",
            "This example should run without errors.",
        ]
    }
    return Dataset.from_dict(dummy_data)


def tokenize_function(examples, tokenizer):
    """Tokenize dataset examples."""
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized


def initialize_distributed(args):
    WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
    LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))

    if args.force_device == "cuda":
        backend = "nccl"
    else:
        backend = "cpu:gloo,qaic:qccl"

    dist.init_process_group(
        backend=backend,  # "nccl" for GPUs, "gloo" for CPUs
        init_method="env://",  # how processes connect (env vars, file, tcp, etc.)
        world_size=WORLD_SIZE,  # total number of processes
        rank=LOCAL_RANK,  # unique ID for this process
    )


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()

    initialize_distributed(args)  # Initialize distributed training

    # Setup tensor parallelism
    pc = setup_parallelism(args.tp_size, args.dp_size)
    device_mesh = pc.build_device_mesh(args.force_device)
    print(f"Device Mesh: {device_mesh}")

    # Create training arguments
    training_args = create_training_arguments(args, pc)

    # Load tokenizer
    tokenizer = load_tokenizer(args.model_name)

    model = load_model(args.model_name, device_mesh, args.apply_peft)
    print(f"Model loaded on device: {model.device}")

    # Create dataset
    dataset = create_dummy_dataset()

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )

    # Create data collator
    data_collator = DataCollatorWithPadding(tokenizer, padding=True, max_length=128)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Train model
    trainer.train()
    print("Training complete!")


if __name__ == "__main__":
    main()
