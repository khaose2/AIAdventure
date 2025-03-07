#!/usr/bin/env python
import os
import json
import sys
import logging
from pathlib import Path
from typing import Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# ------------------------------
# Logging Setup
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ------------------------------
# Configuration and File Paths
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
AI_PERSONALITY_FILE = CONFIG_DIR / "ai_personality.json"
MODEL_NAME = "JamAndTeaStudios/DeepSeek-R1-Distill-Qwen-7B-FP8-Dynamic"  # Adjust this if needed

# Ensure config folder exists.
CONFIG_DIR.mkdir(exist_ok=True)

# ------------------------------
# Device Setup
# ------------------------------
if torch.cuda.is_available():
    device = "cuda"
    try:
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA is available. Using GPU: {gpu_name}")
    except Exception as e:
        logger.warning("CUDA is available but could not determine device name.", exc_info=e)
else:
    device = "cpu"
    logger.info("CUDA not available. Falling back to CPU.")

# ------------------------------
# Personality Loading
# ------------------------------
def load_personality(filepath: Path) -> Dict[str, Any]:
    if filepath.exists():
        try:
            with filepath.open("r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Could not decode {filepath}. Using default personality.")
    # Default personality for main AI.
    personality = {
        "name": "DeepOracle",
        "traits": ["wise", "analytical", "strategic", "empathetic"],
        "style": "formal and thoughtful",
        "decision_making": "methodical and data-driven"
    }
    with filepath.open("w") as f:
        json.dump(personality, f, indent=4)
    return personality

personality = load_personality(AI_PERSONALITY_FILE)

# ------------------------------
# Model Initialization
# ------------------------------
logger.info("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    config=config,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True
)
# Do not call model.to(device) explicitly if offloading is used.
model.eval()
logger.info("Model loaded and set to eval mode.")

# Optionally compile the model for optimization (PyTorch 2.0+)
try:
    model = torch.compile(model)
    logger.info("Model successfully compiled with torch.compile.")
except Exception as e:
    logger.warning("torch.compile failed, continuing without compilation.", exc_info=e)

# ------------------------------
# Response Generation Function
# ------------------------------
def generate_response(prompt: str, personality: Dict[str, Any]) -> str:
    enriched_prompt = f"You are {personality['name']}. {prompt}"
    inputs = tokenizer(enriched_prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ------------------------------
# Main CLI Loop
# ------------------------------
def main() -> None:
    print("Simple Text-to-AI Test Application")
    print("Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = generate_response(user_input, personality)
        print("AI:", response, "\n")

if __name__ == "__main__":
    main()
