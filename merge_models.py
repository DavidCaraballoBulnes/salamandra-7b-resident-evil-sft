"""
Model Merging Script for LoRA Adapters.

This script merges a trained LoRA (Low-Rank Adaptation) adapter with its 
base foundational model. To prevent Out-Of-Memory (OOM) errors on systems 
with constrained VRAM (e.g., 12GB GPUs), the merge operation is explicitly 
offloaded to the system RAM (CPU).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==========================================
# 1. PATHS AND CONFIGURATION
# ==========================================
BASE_MODEL_NAME = "BSC-LT/salamandra-7b-instruct" # The foundational base model
ADAPTER_DIR = "./salamandra-finetuned"            # Directory containing the trained LoRA adapter
OUTPUT_DIR = "./salamandra-merged"                # Target directory for the standalone merged model

print(f"1. Loading the tokenizer from {ADAPTER_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

print(f"2. Loading the base model into system RAM (CPU)...")
# CRITICAL CONFIGURATION: device_map="cpu" is enforced to bypass 'accelerate' OOM errors.
# This guarantees the 7B parameters are loaded into system RAM instead of VRAM.
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.bfloat16,   
    device_map="cpu",             # <--- FORCING EXCLUSIVE USE OF SYSTEM RAM
    trust_remote_code=True,       
)

print(f"3. Loading the LoRA adapter from {ADAPTER_DIR}...")
# Because the base model resides in the CPU, the PEFT adapter will seamlessly inherit 
# the CPU device placement, avoiding device mismatch conflicts.
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

print("4. Merging weights (Merge & Unload)...")
# This matrix multiplication operation is CPU-bound. It may take several minutes to complete.
merged_model = model.merge_and_unload()

print(f"5. Saving the unified model to: {OUTPUT_DIR}...")
merged_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Merge completed successfully! The standalone model is ready for inference.")