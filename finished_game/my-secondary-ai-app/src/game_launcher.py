import json
import random
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import aiohttp

# Configuration: model and file paths.
model_name = "cortecs/DeepSeek-R1-Distill-Qwen-32B-FP8-Dynamic"
AI_PERSONALITY_FILE = "config/ai_personality.json"
EVENT_REGISTRY_FILE = "event_registry.json"

# Initialize tokenizer and model with trust_remote_code enabled.
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True
)

# Create the FastAPI application.
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Event Registry Functions ---
def load_event_registry():
    """Load dynamic game events from file."""
    if os.path.exists(EVENT_REGISTRY_FILE):
        with open(EVENT_REGISTRY_FILE, "r") as file:
            return json.load(file)
    return {"events": []}

def save_event_registry(registry):
    """Save updated game events to file."""
    with open(EVENT_REGISTRY_FILE, "w") as file:
        json.dump(registry, file, indent=4)

# --- Personality Functions ---
def load_personality(personality_file):
    """Load personality data from the given JSON file."""
    if os.path.exists(personality_file):
        with open(personality_file, "r") as f:
            return json.load(f)
    return {
        "name": "DeepOracle",
        "traits": ["wise", "analytical", "strategic", "empathetic"],
        "style": "formal and thoughtful",
        "decision_making": "methodical and data-driven",
        "knowledge": []
    }

def update_personality(personality_file, new_knowledge):
    """Append new knowledge to the personality file."""
    personality = load_personality(personality_file)
    if "knowledge" not in personality:
        personality["knowledge"] = []
    personality["knowledge"].append(new_knowledge)
    with open(personality_file, "w") as f:
        json.dump(personality, f, indent=4)

# --- Dynamic Context ---
def format_dynamic_context():
    """
    Build context from the dynamic event registry.
    Uses the last three events and a random event to enrich the prompt.
    """
    registry = load_event_registry()
    events = registry.get("events", [])
    lines = []
    if events:
        # Sort events by their numeric 'id'.
        sorted_events = sorted(events, key=lambda e: int(e.get('id', 0)))
        recent = sorted_events[-3:]
        for idx, evt in enumerate(recent, 1):
            lines.append(f"Recent Scenario {idx}: {evt.get('response', 'No response yet')}")
        random_event = random.choice(events)
        lines.append(f"Random Twist: {random_event.get('response', 'No response yet')}")
    return "\n".join(lines)

# --- AI Response Generation ---
def generate_ai_response(prompt: str) -> str:
    """Generate an AI response using enriched dynamic context."""
    context = format_dynamic_context()
    enriched_prompt = f"{context}\n\nUser Action:\n{prompt}" if context else prompt
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(enriched_prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# --- Secondary AI Call ---
async def send_to_secondary_ai(prompt: str) -> str:
    """
    Send the main AI response to the secondary AI at localhost:8001.
    """
    url = "http://localhost:8001/ai_action/"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json={"story_memory": [prompt]}) as resp:
                data = await resp.json()
                return data.get("ai_response", "No response from secondary AI.")
        except Exception as e:
            return f"Error contacting secondary AI: {e}"

# --- Hub Action Endpoint ---
@app.post("/hub_action/")
async def hub_action(request: Request, background_tasks: BackgroundTasks):
    """
    Process incoming requests.
    Expects a JSON payload with "prompt" and "mode".
    Mode "user" returns only the main AI response.
    Mode "ai_interaction" additionally triggers the secondary AI.
    """
    data = await request.json()
    prompt = data.get("prompt", "")
    mode = data.get("mode", "user")
    if not prompt:
        return {"error": "Prompt is required."}
    
    main_response = generate_ai_response(prompt)
    
    # Update event registry.
    registry = load_event_registry()
    event_id = len(registry.get("events", [])) + 1
    new_event = {"id": event_id, "action": prompt, "response": main_response}
    registry.setdefault("events", []).append(new_event)
    save_event_registry(registry)
    
    # Every even-numbered event, update personality.
    if event_id % 2 == 0:
        knowledge_entry = f"Event #{event_id}: Learned from response: {main_response}"
        update_personality(AI_PERSONALITY_FILE, knowledge_entry)
    
    if mode == "user":
        return {"ai_response": main_response.strip()}
    elif mode == "ai_interaction":
        background_tasks.add_task(send_to_secondary_ai, main_response)
        return {"main_ai_response": main_response.strip(), "note": "Secondary AI interaction is being processed."}
    else:
        return {"error": "Invalid mode."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
