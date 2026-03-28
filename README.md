# 🦎 Salamandra 7B SFT - Resident Evil Edition (Optimized for 12GB VRAM)

This repository contains the necessary scripts to perform a Supervised Fine-Tuning (SFT) of the foundational model `BSC-LT/salamandra-7b-instruct` using **QLoRA**. The model has been specifically fine-tuned to answer questions and provide accurate, canonical information about the Resident Evil universe.

### 🔗 Hugging Face Resources
You can find the artifacts generated and used by this project directly on Hugging Face:
* **Dataset:** [DavidCaraballoBulnes/ResidentEvil-Data-Instruct](https://huggingface.co/datasets/DavidCaraballoBulnes/ResidentEvil-Data-Instruct)
* **Final Merged Model:** [DavidCaraballoBulnes/ResidentEvil-QA](https://huggingface.co/DavidCaraballoBulnes/ResidentEvil-QA)

---

## 💻 Hardware Specifications (Development Setup)
The project has been specifically designed to run on consumer hardware with video memory limitations, successfully training a 7-billion parameter model on a single 12GB VRAM GPU.
* **CPU:** AMD Ryzen 7 5700X (8-Core)
* **RAM:** 32.0 GB DIMM 3600 MT/s
* **GPU:** NVIDIA GeForce RTX 4070 Ti (12GB VRAM)

## 🚀 Optimizations Applied
To prevent *Out Of Memory* (OOM) errors in CUDA, the following techniques were implemented in the training pipeline:
* **4-bit Quantization (NF4):** Utilizing `BitsAndBytes` to drastically compress the base model's footprint.
* **Smart Offloading:** Strict configuration of `max_memory` to prioritize the 12GB of VRAM and overflow the excess into the system's 32GB RAM.
* **Gradient Checkpointing:** Recomputing activations to free up VRAM during the backward pass.
* **Paged Optimizer:** Using `paged_adamw_8bit` to manage optimizer states within standard RAM.
* **Batch Size Management:** A physical batch size of `1` combined with `16` gradient accumulation steps to stabilize learning.
* **Prompt Engineering:** The inference script uses strict system prompts and low temperature (0.2) to prevent AI hallucinations and ground the model in the official game canon.

Furthermore, the `merge_models.py` script performs the Merge & Unload of the LoRA adapter with the base model by **forcing exclusive use of the CPU** (`device_map="cpu"`). This evades memory spikes that would otherwise crash the GPU during this final step.

## 🛠️ Usage

### 1. Installation
**⚠️ Important:** Ensure you have exactly **Python 3.11.x** installed. Newer versions (like 3.12+) may cause severe CUDA/GPU compatibility issues with the underlying deep learning libraries (`torch`, `bitsandbytes`, etc.).

Install the dependencies by running:
```bash
pip install .
```

### 2. Training
The script is pre-configured to download the Resident Evil dataset directly from Hugging Face. Simply run:
```bash
python fine_tuning.py
```

### 3. Merging the Model
Once training is complete, merge the LoRA adapter with the base model:
```bash
python merge_models.py
```
The standalone, inference-ready model will be saved in the `salamandra-merged/` directory.

### 4. Inference & Testing
To test the fine-tuned model and ask questions about the Resident Evil universe, run the included inference script. It uses 4-bit quantization to run smoothly on consumer GPUs and applies Prompt Engineering to ensure factual accuracy:
```bash
python inference.py
```
*Note: The `BSC-LT/salamandra-7b-instruct` base model is highly optimized for the Spanish language. It is recommended to prompt the model in Spanish for the best and most accurate results.*
