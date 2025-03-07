import uvicorn
from fastapi import FastAPI
from secondary_ai import get_ai_action

app = FastAPI()

@app.post("/ai_action/")
async def ai_action_endpoint(request):
    return await get_ai_action(request)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)