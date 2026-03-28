"""
Inference Script for the Resident Evil QA Model.

This script demonstrates how to load the fine-tuned Salamandra 7B model
from Hugging Face and generate accurate answers about the Resident Evil universe.

Key Features:
- 4-bit Quantization: Ensures the 7B model fits comfortably in a 12GB VRAM GPU.
- Prompt Engineering: Uses strict system prompts to ground the model in the official canon.
- Factual Generation: Uses low temperature (0.2) to prevent AI hallucinations.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ==========================================
# 1. CONFIGURATION & MODEL LOADING
# ==========================================
# Target model hosted on Hugging Face Hub
MODEL_ID = "DavidCaraballoBulnes/ResidentEvil-QA"

print(f"Loading tokenizer from '{MODEL_ID}'...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Configuring 4-bit quantization (NF4) for memory-efficient inference...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print(f"Loading model into GPU with smart device mapping...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# ==========================================
# 2. INFERENCE FUNCTION (WITH PROMPT ENGINEERING)
# ==========================================
def ask_lore(question: str) -> None:
    """
    Sends a question to the model with strict system guardrails to prevent hallucinations.
    """
    print(f"\n[User]: {question}")
    
    # SYSTEM PROMPT: Establishes the AI's persona and sets strict boundaries.
    # We use Spanish here as the underlying BSC-LT/salamandra model is highly 
    # optimized for the Spanish language.
    system_prompt = (
        "Eres un archivero experto en la historia, los personajes y los virus "
        "del universo oficial de los videojuegos de Resident Evil (creado por Capcom). "
        "Tu misión es dar respuestas precisas, directas y basadas estrictamente en el canon. "
        "Reglas críticas: No inventes nombres de criaturas, no mezcles novelas con los juegos, "
        "y bajo ninguna circunstancia alucines información. Si no conoces la respuesta exacta, "
        "debes responder: 'No tengo información verificada sobre esto en los archivos de Umbrella'."
    )
    
    # Format the prompt using the model's official chat template
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,       # Limit response length to avoid rambling
            temperature=0.2,          # Low temperature enforces factual accuracy over creativity
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the newly generated tokens (ignoring the prompt structure)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    print(f"[Resident Evil QA]: {response}\n")

# ==========================================
# 3. EXECUTION
# ==========================================
if __name__ == "__main__":
    print("\n=== Resident Evil Lore AI Initialized ===")
    
    # Example questions in Spanish to test the model's native capabilities
    ask_lore("¿Quién es Albert Wesker y cuáles son sus objetivos principales?")
    ask_lore("Explica qué es el Virus-T y cómo afecta a los humanos.")
    
    # Uncomment the following block to enable an interactive chat in the terminal:
    # print("\n(Escribe 'salir' para terminar el programa)")
    # while True:
    #     user_input = input("\nHaz una pregunta sobre Resident Evil: ")
    #     if user_input.lower() in ['salir', 'exit', 'quit']:
    #         break
    #     ask_lore(user_input)