"""
Fine-Tuning Script using QLoRA for Large Language Models (LLMs).

This script adapts a foundational model to a specific domain (Resident Evil)
using the modern Hugging Face ecosystem (Transformers, PEFT, Datasets, TRL).

Optimizations applied for mixed hardware (RTX 4070 Ti 12GB VRAM + 32GB RAM):
- Smart Offloading: Prioritizes 100% compute on the GPU and sends excess to system RAM.
- 4-bit Quantization (NF4) via BitsAndBytes.
- Gradient Checkpointing to reduce activation memory usage.
- Paged optimizer (paged_adamw_8bit) to manage states in RAM.
- Physical batch size reduced to 1 with gradient accumulation (16).
- Adjusted context length (max_length=512) and packing disabled.
"""

import warnings
# Suppress library warnings to keep the console output clean
warnings.filterwarnings("ignore")

import torch
import sys
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ==========================================
# 0. STRICT GPU CHECK
# ==========================================
if not torch.cuda.is_available():
    print("ERROR: PyTorch cannot detect your NVIDIA GPU.")
    print("Please make sure you have installed PyTorch with CUDA support.")
    sys.exit(1) # Stops execution to prevent accidentally using the CPU
else:
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")

# ==========================================
# 1. PATHS & MODEL CONFIGURATION
# ==========================================
MODEL_NAME = "BSC-LT/salamandra-7b-instruct"             # Base foundational model
OUTPUT_DIR = "./salamandra-finetuned"                    # Output directory for the LoRA adapter
DATASET_NAME = "DavidCaraballoBulnes/ResidentEvil-Data-Instruct" # Hugging Face dataset

# ==========================================
# 2. LOAD BASE MODEL IN 4-BIT WITH SMART OFFLOADING
# ==========================================
# Extreme model compression configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",             # NormalFloat 4, optimal distribution for weights
    bnb_4bit_compute_dtype=torch.bfloat16, # Compute in bfloat16 precision (compatible with RTX 4000)
    bnb_4bit_use_double_quant=True,        # Quantize constants for extra savings
)

# Strict memory limits: Fill the 12GB GPU first, send the rest to the 32GB RAM
max_memory_mapping = {
    0: "11.5GiB",   # Squeeze the RTX 4070 Ti (leaving a tiny margin for the OS)
    "cpu": "25GiB"  # Limit conventional RAM so your computer doesn't freeze
}

print("Loading base model and tokenizer with smart memory allocation...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",                     # Allow automatic distribution...
    max_memory=max_memory_mapping,         # ...but obey these strict manual limits
    trust_remote_code=True,                # Required to run model code from HuggingFace Hub
)

# Load and configure the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Unify padding token with end-of-sequence token
tokenizer.padding_side = "right"           # Right-padding is mandatory for causal modeling

# ==========================================
# 3. LoRA ARCHITECTURE CONFIGURATION
# ==========================================
# LoRA injects trainable low-rank matrices, freezing the original millions of parameters.
lora_config = LoraConfig(
    r=16,                    # Rank dimension: capacity/memory balance
    lora_alpha=32,           # Scaling factor (typically 2x the value of r)
    lora_dropout=0.05,       # Dropout rate for regularization
    bias="none",             # Original biases remain frozen
    task_type="CAUSAL_LM",   # Task nature: Autoregressive text generation
    target_modules=[         # Specific transformer layers to intercept and adapt
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj",
    ],
)

# Prepare model structure for quantized training
model = prepare_model_for_kbit_training(model)

# Attach LoRA adapter to the base model
model = get_peft_model(model, lora_config)

# Disable inference cache to ensure compatibility with gradient_checkpointing
model.config.use_cache = False 

# Parameter audit
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")

# ==========================================
# 4. DATASET PREPARATION & FORMATTING
# ==========================================
print("Loading and processing the dataset from Hugging Face...")
# Load dataset directly from Hugging Face Hub
dataset = load_dataset(DATASET_NAME, split="train")

def format_chat(example):
    """
    Converts the structured 'messages' array into a single pre-formatted text block
    according to the model's official chat template.
    """
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,              # Returns string; SFTTrainer handles tokenization internally
        add_generation_prompt=False, # Full training of system, user, and assistant roles
    )
    return {"text": text}

# Apply vector formatting to the entire dataset
dataset = dataset.map(format_chat)

# Standard Hold-out split: 90% Training, 10% Validation
dataset = dataset.train_test_split(test_size=0.1)

# ==========================================
# 5. SFT TRAINING LOOP CONFIGURATION
# ==========================================
# Detect hardware support for bfloat16 precision (natively supported by your RTX 4070 Ti)
is_bf16_supported = torch.cuda.is_bf16_supported()

# SFTConfig centralizes HuggingFace Trainer arguments and TRL-specific ones
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,                  # Full cycles over the data
    
    # Critical Memory Optimizations
    per_device_train_batch_size=1,       # Process a single example per physical step
    gradient_accumulation_steps=16,      # Delay weight updates, simulating a batch size of 16
    
    learning_rate=2e-4,                  # Standard optimizer step magnitude for PEFT
    weight_decay=0.01,                   # Overfitting control by limiting weight norm
    warmup_steps=50,                     # Gradual transition from 0 to target learning rate (2e-4)
    lr_scheduler_type="cosine",          # Cosine curve-based learning rate decay
    logging_steps=10,                    # Frequency (in steps) of printing loss metrics
    save_strategy="epoch",               # Persist checkpoints at the end of an epoch
    eval_strategy="epoch",               # Evaluate against test split at the end of an epoch
    bf16=is_bf16_supported,              # Leverage native bfloat16 of your Ada Lovelace architecture
    fp16=not is_bf16_supported,          # Fallback (won't be used in your case)
    gradient_checkpointing=True,         # Recompute activations instead of storing them in memory
    max_grad_norm=0.3,                   # Prevent numerical instability by clipping high gradients
    optim="paged_adamw_8bit",            # Vital for offloading: moves optimizer states to 32GB RAM
    report_to="none",                    # Disable external logs to cloud metric platforms
    
    # TRL specific parameters (Supervised Fine-Tuning)
    dataset_text_field="text",           # Mapping to the textual column of the dataset
    max_length=512,                      # Context length (Suitable for short QA pairs to save RAM/VRAM)
    packing=False,                       # Disabled to avoid attention saturation
)

# ==========================================
# 6. INITIALIZATION AND EXECUTION (SFTTrainer)
# ==========================================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

print("Starting the Fine-Tuning process...")
trainer.train()

# ==========================================
# 7. MODEL SAVING AND SHUTDOWN
# ==========================================
# Reset model state for optimal performance in subsequent inference
model.config.use_cache = True

print(f"Training completed. Saving adapter weights to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Process finished successfully!")