import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import aiohttp  # For AI-to-AI interactions

# Initialize AI components
model_name = "deepseek-ai/DeepSeek-R1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
)

app = FastAPI()

# Add CORS for external interaction (adjust allowed origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def generate_ai_response(prompt):
    """Generates a response from the AI model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

async def send_to_secondary_ai(prompt):
    """Send a prompt to the secondary AI and get its response."""
    async with aiohttp.ClientSession() as session:
        async with session.post("http://localhost:8001/ai_action/", json={"story_memory": [prompt]}) as resp:
            data = await resp.json()
            return data.get("ai_response", "No response from secondary AI.")

@app.post("/hub_action/")
async def hub_action(request: Request, background_tasks: BackgroundTasks):
    """Central hub for user and AI interactions."""
    data = await request.json()
    prompt = data.get("prompt", "")
    mode = data.get("mode", "user")  # 'user' for user input, 'ai_interaction' for AI-to-AI

    if not prompt:
        return {"error": "Prompt is required."}

    # Generate AI response for the user or initiate AI-to-AI interaction
    if mode == "user":
        response = generate_ai_response(prompt)
        return {"ai_response": response.strip()}
    elif mode == "ai_interaction":
        # Generate response from the main AI
        main_response = generate_ai_response(prompt)

        # Send the main AI response to the secondary AI in the background
        background_tasks.add_task(send_to_secondary_ai, main_response)

        return {
            "main_ai_response": main_response.strip(),
            "note": "Secondary AI interaction is being processed."
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
