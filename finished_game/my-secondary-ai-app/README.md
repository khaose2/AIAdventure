# My Secondary AI App

## Overview
This project implements a secondary AI for a game using FastAPI. The AI is designed to enhance the gaming experience by providing dynamic responses based on the game context and a predefined personality.

## Project Structure
```
my-secondary-ai-app
├── src
│   ├── main.py                # Entry point for the FastAPI application
│   ├── secondary_ai.py        # Implementation of the secondary AI
│   └── config
│       └── playerai_personality.json  # AI personality traits and characteristics
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd my-secondary-ai-app
   ```

2. **Install dependencies:**
   Ensure you have Python 3.7 or higher installed. Then, run:
   ```
   pip install -r requirements.txt
   ```

3. **Run the application:**
   Start the FastAPI server by executing:
   ```
   uvicorn src.main:app --reload
   ```

## Usage
Once the server is running, you can interact with the AI by sending POST requests to the `/ai_action/` endpoint with a JSON payload containing the game context.

### Example Request
```json
{
    "story_memory": ["The hero enters the dark forest.", "A mysterious figure appears.", "The hero must decide whether to fight or flee."]
}
```

### Example Response
```json
{
    "ai_response": "As the AI adventurer, I suggest we cautiously approach the figure, ready to defend ourselves if necessary."
}
```

## AI Personality
The AI's personality is defined in `src/config/playerai_personality.json`. You can modify this file to change the AI's traits and decision-making style.

## Dependencies
- FastAPI
- Transformers
- Torch
- Uvicorn

## License
This project is licensed under the MIT License.