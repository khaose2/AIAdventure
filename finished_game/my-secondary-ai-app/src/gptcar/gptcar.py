#!/usr/bin/env python
import os
import time
import logging
from pathlib import Path
from typing import Dict, Any

import openai

# ------------------------------
# Logging Setup
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ------------------------------
# API Key Setup
# ------------------------------
try:
    openai.api_key = os.environ["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"]
    logger.info("OPENAI_API_KEY loaded from environment.")
except KeyError:
    logger.error("OPENAI_API_KEY is not set. Please set your OpenAI API key in the environment.")
    exit(1)

# ------------------------------
# Model and Inference Parameters
# ------------------------------
# Replace MODEL_NAME with the correct identifier for your ChatGPT 4o-mini fine-tunable model.
MODEL_NAME = "gpt-4o-mini-finetunable"  # Adjust if needed.  
# For example, if your model repository is public, you might use:
# MODEL_NAME = "JamAndTeaStudios/DeepSeek-R1-Distill-Qwen-7B-FP8-Dynamic"
# (Ensure your repo is accessible or pass a token via use_auth_token if needed.)

# We set parameters for faster inference:
MAX_TOKENS = 30       # Generate a short response for speed.
TEMPERATURE = 0.0     # Greedy decoding for deterministic and fast output.
TOP_P = 1.0
DO_SAMPLE = False

# ------------------------------
# Response Generation Function
# ------------------------------
def generate_response(user_prompt: str) -> str:
    """
    Send a request to the OpenAI API using the ChatCompletion endpoint with your chosen model.
    Returns the AI's reply as a string.
    """
    try:
        start_time = time.time()
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=DO_SAMPLE,
        )
        elapsed = time.time() - start_time
        logger.info(f"Inference took {elapsed:.2f} seconds.")
        return response.choices[0].message["content"].strip()
    except Exception as e:
        logger.error("Error during API call: %s", e)
        return "Sorry, an error occurred."

# ------------------------------
# Main CLI Loop
# ------------------------------
def main() -> None:
    print("ChatGPT 4o-mini Fine-Tunable Test Application")
    print("Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = generate_response(user_input)
        print("AI:", response, "\n")

if __name__ == "__main__":
    main()
