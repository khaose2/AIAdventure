import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, Request
import uvicorn

# File for secondary AI (player) personality.
PLAYER_PERSONALITY_FILE = "config/player_personality.json"

def load_personality(personality_file):
    """Load personality data from the given JSON file."""
    if os.path.exists(personality_file):
        with open(personality_file, "r") as f:
            return json.load(f)
    return {
        "name": "PlayerAI",
        "traits": ["curious", "resourceful", "adventurous", "adaptable"],
        "style": "engaging and informal",
        "decision_making": "creative and intuitive",
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

# Load the secondary AI personality.
personality = load_personality(PLAYER_PERSONALITY_FILE)

# Use the same quantized model.
model_name = "cortecs/DeepSeek-R1-Distill-Qwen-32B-FP8-Dynamic"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True
)

app = FastAPI()

def personalize_prompt(story_memory):
    """
    Build a prompt that includes the secondary AI's personality and provided story memory.
    """
    personality_intro = (
        f"You are {personality['name']}, the secondary AI with traits: {', '.join(personality['traits'])}. "
        f"Your style is {personality['style']} and you make decisions in a {personality['decision_making']} way."
    )
    memory_text = "\n".join(story_memory)
    return f"{personality_intro}\n\nContext:\n{memory_text}\n\nResponse:"

def generate_ai_response(prompt):
    """Generate a response from the AI model using the given prompt."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
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

@app.post("/ai_action/")
async def ai_action(request: Request):
    """
    Process requests.
    Expects a JSON payload with "story_memory". Builds a personalized prompt, generates a response,
    and updates personality every even-numbered knowledge entry.
    """
    data = await request.json()
    story_memory = data.get("story_memory", [])
    if not isinstance(story_memory, list):
        story_memory = [str(story_memory)]
    prompt = personalize_prompt(story_memory)
    ai_response = generate_ai_response(prompt)
    
    current_event = len(personality.get("knowledge", [])) + 1
    if current_event % 2 == 0:
        knowledge_entry = f"Event #{current_event}: Learned from response: {ai_response}"
        update_personality(PLAYER_PERSONALITY_FILE, knowledge_entry)
    
    return {"ai_response": ai_response.strip()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
