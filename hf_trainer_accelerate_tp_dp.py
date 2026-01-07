import os
import torch
import argparse
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

# Command:
# QAIC_VISIBLE_DEVICES=32,33,34,35 torchrun --nproc_per_node=4 --master-port=1234 run_tp_generic.py --force_device qaic --tp_size 2 --dp_size 2

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
    parallelism_config = {"tp_size": tp_size, "dp_replicate_size": dp_size}
    pc = ParallelismConfig(**parallelism_config)
    print(f"{pc.total_size=}")
    return pc

def create_training_arguments(args, pc):
    """Create training arguments with appropriate settings."""
    from trl.trainer.sft_config import SFTConfig
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
        remove_unused_columns=False,
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
    tp_mesh = device_mesh["tp"]
    dp_mesh = device_mesh["dp_replicate"]
    print(f"Loading model {model_name} with tensor parallelism (tp_size={tp_mesh.size()}) and data parallelism (dp_size={dp_mesh.size()})")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        tp_plan="auto",
        tp_size=tp_mesh.size(),
        device_mesh=tp_mesh
    )
    
    # for name, param in model.named_parameters():
    #     if name == "model.layers.6.mlp.gate_proj.weight":
    #         param.requires_grad = True
    #         # param = param.to(torch.float32)
    #         param.data = param.data.to(torch.float32)
    #     else:
    #         param.requires_grad = False
    print(f"Model tp size: {model.tp_size}")
            
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


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
    LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
    
    import torch.distributed as dist
    dist.init_process_group(
        backend="cpu:gloo,qaic:qccl",       # "nccl" for GPUs, "gloo" for CPUs
        init_method="env://", # how processes connect (env vars, file, tcp, etc.)
        world_size=WORLD_SIZE,         # total number of processes
        rank=LOCAL_RANK,                # unique ID for this process
    )
    
    # Setup tensor parallelism
    pc = setup_parallelism(args.tp_size, args.dp_size)
    device_mesh = pc.build_device_mesh(args.force_device)
    print(f"Device Mesh: {device_mesh}")

    # Create training arguments
    training_args = create_training_arguments(args, pc)
    # if args.dp_size > 0:
    #     training_args.fsdp_plugin_args["activation_checkpointing"] = False
    #     training_args.fsdp_plugin_args["state_dict_type"] = "SHARDED_STATE_DICT"
    #     training_args.fsdp_plugin_args["fsdp_version"] = 2
    #     training_args.fsdp_plugin_args["reshard_after_forward"] = True
    #     training_args.fsdp_plugin_args["auto_wrap_policy"] = "transformer_based_wrap"
    #     training_args.fsdp_plugin_args["cpu_ram_efficient_loading"] = True
    #     training_args.fsdp_plugin_args["forward_prefetch"] = None
    
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
    # data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    
    from peft import LoraConfig
    from trl import SFTTrainer
    peft_config = LoraConfig(
        r=16,                      # rank
        lora_alpha=32,             # scaling factor
        lora_dropout=0.05,          # dropout
        bias="none",
        task_type="CAUSAL_LM",       # task type
        target_modules=["q_proj","v_proj"]
    )

    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        processing_class=tokenizer,
        # peft_config=peft_config,
        # data_collator=data_collator,
    )

    # kwargs = {}
    # if training_args.ddp_find_unused_parameters is not None:
    #     kwargs["find_unused_parameters"] = training_args.ddp_find_unused_parameters
    # elif isinstance(model, PreTrainedModel):
    #     # find_unused_parameters breaks checkpointing as per
    #     # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021
    #     kwargs["find_unused_parameters"] = not model.is_gradient_checkpointing
    # else:
    #     kwargs["find_unused_parameters"] = True

    # if training_args.ddp_bucket_cap_mb is not None:
    #     kwargs["bucket_cap_mb"] = training_args.ddp_bucket_cap_mb

    # if training_args.ddp_broadcast_buffers is not None:
    #     kwargs["broadcast_buffers"] = training_args.ddp_broadcast_buffers

    # kwargs["process_group"] = dp_pg
    # from accelerate.utils import DistributedDataParallelKwargs
    # trainer.accelerator.ddp_handler = DistributedDataParallelKwargs(**kwargs)

    # Train model
    trainer.train()
    print("Training complete!")

if __name__ == "__main__":
    main()