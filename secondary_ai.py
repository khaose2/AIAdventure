import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, Request
import uvicorn
import os

# Initialize AI components
model_name = "deepseek-ai/DeepSeek-R1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

app = FastAPI()

# Load personality file
PERSONALITY_FILE = "playerai_personality.json"

def load_personality():
    """Load AI personality from JSON file."""
    if os.path.exists(PERSONALITY_FILE):
        with open(PERSONALITY_FILE, "r") as file:
            return json.load(file)
    return {
        "name": "Jeff-AI",
        "traits": ["adventurous", "strategic thinker", "playful"],
        "style": "direct, concise, but with a sense of humor",
        "decision_making": "balanced between taking risks and careful planning"
    }

personality = load_personality()

def personalize_prompt(prompt):
    """Modify the prompt based on the personality data."""
    personality_intro = (
        f"You are {personality['name']}, an AI adventurer with the following traits: "
        f"{', '.join(personality['traits'])}. Your style is {personality['style']}. "
        f"Make decisions that reflect your personality and {personality['decision_making']}."
    )
    return f"{personality_intro}\n\n{prompt}"

def generate_ai_response(prompt):
    """Generate a response from the AI using the personalized prompt."""
    prompt = personalize_prompt(prompt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

@app.post("/ai_action/")
async def get_ai_action(request: Request):
    """Generate an AI response based on the game context and memory."""
    data = await request.json()
    story_memory = data.get("story_memory", [])
    context = "\n".join(story_memory[-3:])[-500:] if story_memory else ""

    # Create a prompt considering recent story memory
    prompt = (
        f"{context}\n"
        "As the AI adventurer, decide your next action in the game. "
        "Consider the current situation and choose an action that progresses the story."
    )
    
    response = generate_ai_response(prompt)
    return {"ai_response": response.strip()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
