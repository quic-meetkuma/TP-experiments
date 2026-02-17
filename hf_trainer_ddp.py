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
import torch.distributed as dist
from peft import get_peft_model
from peft import LoraConfig

# Command:
# QAIC_VISIBLE_DEVICES=32,33,34,35 torchrun --nproc_per_node=4 --master-port=1234 hf_trainer_ddp.py --force_device qaic

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
        "--apply_peft", action="store_true", help="Apply PEFT (LoRA) to the model."
    )
    return parser.parse_args()


def create_training_arguments(args):
    """Create training arguments with appropriate settings."""
    return TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=args.epochs,
        logging_steps=1,
        fp16=True,
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


def apply_peft_to_model(model):
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
    return model

    
def load_tokenizer(model_name):
    """Load and configure tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name, apply_peft):
    """Load model with tensor parallelism."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
    )
    # Need to explicitly untie the embedding weights here to consider
    # this as separate params in further TP processing
    model.lm_head.weight = nn.Parameter(model.lm_head.weight.clone())

    # Apply peft to the model
    if apply_peft:
        model = apply_peft_to_model(model)
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
        # truncation=True,
        # max_length=128,
        return_tensors="pt",
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()

    # Create training arguments
    training_args = create_training_arguments(args)

    # Load tokenizer
    tokenizer = load_tokenizer(args.model_name)

    model = load_model(args.model_name, args.apply_peft)
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
