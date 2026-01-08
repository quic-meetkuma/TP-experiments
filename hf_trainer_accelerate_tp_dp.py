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
    AutoConfig
)
from transformers import DataCollatorWithPadding
from accelerate.utils import ParallelismConfig
from torch.distributed.tensor.parallel.ddp import _pre_dp_module_transform
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


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

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run tensor parallel training on GPU or QAIC")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1b", help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="./tp_output", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--force_device", type=str, choices=["cuda", "qaic", "cpu"], default="auto",
                        help="Force specific device type (auto will use best available)")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision if available")
    parser.add_argument("--tp_size", type=int, help="TP degree for tensor parallelism")
    parser.add_argument("--dp_size", type=int, help="DP degree for Distributed data parallelism")
    return parser.parse_args()

def setup_parallelism(tp_size, dp_size):
    """Set up tensor parallelism configuration."""
    parallelism_config = {}
    if tp_size is not None:
        parallelism_config["tp_size"] = tp_size
    if dp_size is not None:
        parallelism_config["dp_replicate_size"] = dp_size

    pc = ParallelismConfig(**parallelism_config)
    print(f"{pc.total_size=}")
    return pc

def create_training_arguments(args, pc):
    """Create training arguments with appropriate settings."""
    # from trl.trainer.sft_config import SFTConfig
    # return SFTConfig(
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
        max_grad_norm=None,      # This is the root cause for unscale called twice. Will solve it later.
        # fsdp='no_shard',
        # fsdp_config={
        #     "fsdp_version" : 2,
        #     "reshard_after_forward" : True,
        #     "auto_wrap_policy" : "transformer_based_wrap",
        #     "state_dict_type" : "SHARDED_STATE_DICT",
        #     "activation_checkpointing" : False,
        #     "cpu_ram_efficient_loading" : True,
        # }
    )

def load_tokenizer(model_name):
    """Load and configure tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_model(model_name, device_mesh):
    """Load model with tensor parallelism."""
    # if hasattr(device_mesh, )
    tp_enabled = False
    dp_enabled = False
    dp_size = 1
    tp_size = 1
    if "dp_replicate" in device_mesh.mesh_dim_names:
        dp_enabled = True
        dp_mesh = device_mesh["dp_replicate"]
        dp_size = dp_mesh.size()
    if "tp" in device_mesh.mesh_dim_names:
        tp_enabled = True
        tp_mesh = device_mesh["tp"]
        tp_size = tp_mesh.size()

    assert tp_enabled or dp_enabled, "Either TP or DP must be enabled for this experiment. Both are disabled. Check your device mesh."
    print(f"Loading model {model_name} with tensor parallelism (tp_size={tp_size}) and data parallelism (dp_size={dp_size})")

    kwargs = {}
    if tp_enabled:
        kwargs["tp_plan"] = "auto"
        kwargs["tp_size"] = tp_mesh.size()
        kwargs["device_mesh"] = tp_mesh

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        **kwargs,
    )
    # Need to explicitly untie the embedding weights here to consider
    # this as separate params in further TP processing
    model.lm_head.weight = nn.Parameter(model.lm_head.weight.clone())

    # _pre_dp_module_transform(model)

    # Wrap with DDP using data parallel group
    # dp_pg = device_mesh.get_group(mesh_dim=0)
    # model = DDP(model, process_group=dp_pg)

    return model

def create_dummy_dataset():
    """Create a dummy dataset for testing."""
    dummy_data = {
        "text": [
            "This is a sample sentence for training.",
            "Tensor parallelism is a cool technique.",
            "We need to test the training loop.",
            "This example should run without errors."
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
        return_tensors="pt"
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
        backend=backend,       # "nccl" for GPUs, "gloo" for CPUs
        init_method="env://", # how processes connect (env vars, file, tcp, etc.)
        world_size=WORLD_SIZE,         # total number of processes
        rank=LOCAL_RANK,                # unique ID for this process
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

    # Load model
    # device_mesh = pc.get_device_mesh(args.force_device)

    # Setup 2D device mesh
    # mesh_2d = init_device_mesh(
    #     args.force_device,
    #     (args.dp_size, args.tp_size),
    #     # (args.dp_size),
    #     mesh_dim_names=("dp", "tp"),
    #     # mesh_dim_names=("dp",),
    # )

    model = load_model(args.model_name, device_mesh)
    print(f"Model loaded on device: {model.device}")

    # Create dataset
    dataset = create_dummy_dataset()

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )

    # Create data collator
    data_collator = DataCollatorWithPadding(tokenizer, padding=True, max_length=128)

    # from peft import LoraConfig
    # from trl import SFTTrainer
    # peft_config = LoraConfig(
    #     r=16,                      # rank
    #     lora_alpha=32,             # scaling factor
    #     lora_dropout=0.05,          # dropout
    #     bias="none",
    #     task_type="CAUSAL_LM",       # task type
    #     target_modules=["q_proj","v_proj"]
    # )


    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        # peft_config=peft_config,
    )

    # Train model
    trainer.train()
    print("Training complete!")

if __name__ == "__main__":
    main()