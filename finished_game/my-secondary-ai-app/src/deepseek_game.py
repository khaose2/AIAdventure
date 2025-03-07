#!/usr/bin/env python
import os
import json
import random
import time
import threading
import sys
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# ------------------------------
# Configuration and File Paths
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
EVENT_REGISTRY_FILE = BASE_DIR / "event_registry.json"
AI_PERSONALITY_FILE = CONFIG_DIR / "ai_personality.json"
PLAYER_PERSONALITY_FILE = CONFIG_DIR / "player_personality.json"
# Use the 7B model variant from the correct repository.
MODEL_NAME = "JamAndTeaStudios/DeepSeek-R1-Distill-Qwen-7B-FP8-Dynamic"
LOG_FILE = BASE_DIR / "deepseek.log"

# Ensure config folder exists.
CONFIG_DIR.mkdir(exist_ok=True)

# ------------------------------
# Logging Setup
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# ------------------------------
# Device Check
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
# Tokenization Cache
# ------------------------------
_token_cache: Dict[str, Any] = {}

def get_tokenized(text: str) -> Dict[str, torch.Tensor]:
    """Cache tokenized representations for static text."""
    if text in _token_cache:
        return _token_cache[text]
    tokens = tokenizer(text, return_tensors="pt")
    tokens = {k: v.to(device) for k, v in tokens.items()}
    _token_cache[text] = tokens
    return tokens

# ------------------------------
# Personality Loading and Updating
# ------------------------------
def load_personality(filepath: Path) -> Dict[str, Any]:
    if filepath.exists():
        try:
            with filepath.open("r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Could not decode {filepath}. Using default personality.")
    if filepath == AI_PERSONALITY_FILE:
        personality = {
            "name": "DeepOracle",
            "traits": ["wise", "analytical", "strategic", "empathetic"],
            "style": "formal and thoughtful",
            "decision_making": "methodical and data-driven",
            "knowledge": []
        }
    else:
        personality = {
            "name": "PlayerAI",
            "traits": ["curious", "resourceful", "adventurous", "adaptable"],
            "style": "engaging and informal",
            "decision_making": "creative and intuitive",
            "knowledge": []
        }
    with filepath.open("w") as f:
        json.dump(personality, f, indent=4)
    return personality

def update_personality(personality_file: Path, new_knowledge: str) -> None:
    personality = load_personality(personality_file)
    personality.setdefault("knowledge", []).append(new_knowledge)
    with personality_file.open("w") as f:
        json.dump(personality, f, indent=4)
    logger.info(f"Updated personality in {personality_file} with new knowledge.")

main_personality = load_personality(AI_PERSONALITY_FILE)
secondary_personality = load_personality(PLAYER_PERSONALITY_FILE)

# ------------------------------
# Event Registry Functions
# ------------------------------
def load_event_registry() -> Dict[str, Any]:
    if EVENT_REGISTRY_FILE.exists():
        try:
            with EVENT_REGISTRY_FILE.open("r") as file:
                return json.load(file)
        except json.JSONDecodeError:
            logger.error(f"Error decoding {EVENT_REGISTRY_FILE}. Reinitializing registry.")
            return {"events": []}
    return {"events": []}

def save_event_registry(registry: Dict[str, Any]) -> None:
    with EVENT_REGISTRY_FILE.open("w") as file:
        json.dump(registry, file, indent=4)

# ------------------------------
# Model Initialization
# ------------------------------
logger.info("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)  # If the repo is private, pass use_auth_token="YOUR_TOKEN_HERE"
config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)         # Similarly for config, if needed.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    config=config,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True
)
model.eval()
logger.info("Model loaded and set to eval mode.")

# Attempt to compile the model with torch.compile for optimized inference.
try:
    model = torch.compile(model)
    logger.info("Model successfully compiled with torch.compile.")
except Exception as e:
    logger.warning("torch.compile failed, continuing without model compilation.", exc_info=e)

# ------------------------------
# Dynamic Fine Tuning Function
# ------------------------------
def dynamic_fine_tune(training_text: str, event_id: int, num_steps: int = 1, lr: float = 1e-5) -> None:
    logger.info(f"Starting dynamic fine tuning for event {event_id}...")
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    inputs = tokenizer(training_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["labels"] = inputs["input_ids"].clone()
    for _ in range(num_steps):
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    model.eval()
    logger.info(f"Dynamic fine tuning complete for event {event_id}.")

# ------------------------------
# Response Generation Functions
# ------------------------------
def generate_response(prompt: str, personality: Dict[str, Any]) -> str:
    enriched_prompt = f"You are {personality['name']}. {prompt}"
    tokens = get_tokenized(enriched_prompt)
    outputs = model.generate(
        **tokens,
        max_length=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def generate_main_ai_response(prompt: str) -> str:
    return generate_response(prompt, main_personality)

def generate_secondary_ai_response(story_memory: list) -> str:
    personality_intro = (
        f"You are {secondary_personality['name']}, with traits: {', '.join(secondary_personality['traits'])}. "
        f"Your style is {secondary_personality['style']} and you make decisions in a {secondary_personality['decision_making']} way."
    )
    _ = get_tokenized(personality_intro)  # Cache the static part.
    memory_text = "\n".join(story_memory)
    prompt = f"{personality_intro}\n\nContext:\n{memory_text}\n\nResponse:"
    return generate_response(prompt, secondary_personality)

def format_dynamic_context() -> str:
    registry = load_event_registry()
    events = registry.get("events", [])
    lines = []
    if events:
        sorted_events = sorted(events, key=lambda e: int(e.get("id", 0)))
        recent = sorted_events[-3:]
        for idx, evt in enumerate(recent, 1):
            lines.append(f"Recent Scenario {idx}: {evt.get('response', 'No response yet')}")
        random_event = random.choice(events)
        lines.append(f"Random Twist: {random_event.get('response', 'No response yet')}")
    return "\n".join(lines)

# ------------------------------
# Core Hub Action (Direct Function Call)
# ------------------------------
def hub_action(prompt: str, mode: str) -> Dict[str, Any]:
    context = format_dynamic_context()
    combined_prompt = f"{context}\n\nUser Action:\n{prompt}" if context else prompt
    main_response = generate_main_ai_response(combined_prompt)
    
    registry = load_event_registry()
    event_id = len(registry.get("events", [])) + 1
    new_event = {"id": event_id, "action": prompt, "response": main_response}
    registry.setdefault("events", []).append(new_event)
    save_event_registry(registry)
    
    # Run fine tuning in background so as not to block.
    if event_id % 4 == 0:
        training_text = f"{prompt} {main_response}"
        threading.Thread(target=dynamic_fine_tune, args=(training_text, event_id), daemon=True).start()
    
    if mode == "user":
        return {"ai_response": main_response.strip()}
    elif mode == "ai_interaction":
        secondary_response = generate_secondary_ai_response([main_response])
        registry = load_event_registry()
        event_id = len(registry.get("events", [])) + 1
        new_event = {"id": event_id, "action": main_response, "response": secondary_response}
        registry.setdefault("events", []).append(new_event)
        save_event_registry(registry)
        return {
            "main_ai_response": main_response.strip(),
            "secondary_ai_response": secondary_response.strip()
        }
    else:
        return {"error": "Invalid mode."}

async def async_hub_action(prompt: str, mode: str) -> Dict[str, Any]:
    return await asyncio.to_thread(hub_action, prompt, mode)

async def async_input(prompt: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, input, prompt)

# ------------------------------
# Asynchronous Command-Line Interface (CLI)
# ------------------------------
async def async_print_help() -> None:
    help_text = """
Welcome to DeepSeek Medieval Dungeon Adventure!

Instructions:
- In Human Play, you guide your character through a medieval dungeon.
- In AI Play, the AIs converse automatically to craft the story.
- Type 'help' to display these instructions again.
- Type 'exit' to quit the game.

Examples:
- Human Play: "I cautiously open the creaky door." or "I go to the river and drink."
- AI Play: The AIs will exchange responses like:
    [Main AI]: "You enter a dim corridor, sensing danger."
    [Secondary AI]: "The corridor reeks of decay; perhaps poison lurks ahead."

Let's begin your adventure!
"""
    print(help_text)

async def async_print_events_table() -> None:
    registry = load_event_registry()
    events = registry.get("events", [])
    if not events:
        print("No events recorded yet.")
        return
    header = f"{'ID':<4} {'Action':<20} {'Response':<40}"
    print("\n=== Event Registry ===")
    print(header)
    print("-" * len(header))
    for event in events:
        print(f"{event.get('id', '?'):<4} {event.get('action','')[:18]:<20} {event.get('response','')[:38]:<40}")
    print("=" * len(header) + "\n")

async def async_human_play(mode: str) -> None:
    registry = load_event_registry()
    if not registry.get("events"):
        seed = "Welcome to the Medieval Dungeon Adventure! You find yourself in a dark, musty dungeon. What do you do?"
        print("Starting adventure with seed prompt...")
        try:
            result = await async_hub_action(seed, mode)
            if "ai_response" in result:
                dm_response = result["ai_response"]
                print("\nDM:", dm_response, "\n")
            else:
                logger.error("Error from hub_action: %s", result)
        except Exception as e:
            logger.error("Error sending seed prompt: %s", e)
    while True:
        await async_print_events_table()
        user_input = await async_input("Enter your action (or 'help' for instructions, 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("Exiting adventure. Farewell!")
            break
        elif user_input.lower() == "help":
            await async_print_help()
            continue
        try:
            result = await async_hub_action(user_input, mode)
            if mode == "user":
                ai_response = result.get("ai_response", "")
                print("\nDM:", ai_response, "\n")
            elif mode == "ai_interaction":
                ai_response = result.get("main_ai_response", "")
                sec_response = result.get("secondary_ai_response", "")
                print("\n[Main AI]:", ai_response)
                print("[Secondary AI]:", sec_response, "\n")
            else:
                print("Invalid mode in response.")
        except Exception as e:
            logger.error("Error during request: %s", e)

async def async_ai_play(mode: str, iterations: int = 10) -> None:
    seed = "Welcome to the Medieval Dungeon Adventure! You find yourself in a dark, mysterious dungeon. What happens next?"
    current_prompt = seed
    print("Starting AI Play conversation...\n")
    for i in range(iterations):
        try:
            result = await async_hub_action(current_prompt, mode)
            if not result:
                print("Error in hub_action result.")
                break
            main_response = result.get("main_ai_response", result.get("ai_response", ""))
            print(f"\n[Main AI]: {main_response}")
            registry = load_event_registry()
            new_event = {"id": len(registry.get("events", [])) + 1, "action": current_prompt, "response": main_response}
            registry.setdefault("events", []).append(new_event)
            save_event_registry(registry)
  
            secondary_response = result.get("secondary_ai_response", "")
            if secondary_response:
                print(f"\n[Secondary AI]: {secondary_response}")
                registry = load_event_registry()
                new_event = {"id": len(registry.get("events", [])) + 1, "action": main_response, "response": secondary_response}
                registry.setdefault("events", []).append(new_event)
                save_event_registry(registry)
                current_prompt = secondary_response
            else:
                current_prompt = main_response
            await asyncio.sleep(2)
        except Exception as e:
            logger.error("Error during AI conversation: %s", e)
            break

async def main_async() -> None:
    print("Starting DeepSeek Medieval Dungeon Adventure...", flush=True)
    await async_print_help()
    
    mode_choice = await async_input("Select play mode: Type 'y' for AI Play (AIs converse automatically) or 'n' for Human Play (you guide the adventure): ")
    if mode_choice.lower().startswith('y'):
        mode = "ai_interaction"
        print("AI Play enabled: AIs will converse automatically.\n")
        await async_ai_play(mode, iterations=10)
    elif mode_choice.lower().startswith('n'):
        mode = "user"
        print("Human Play enabled: You will guide the adventure.\n")
        await async_human_play(mode)
    else:
        print("Invalid option. Exiting.")
    print("Thank you for playing DeepSeek Medieval Dungeon Adventure!")

if __name__ == "__main__":
    asyncio.run(main_async())
