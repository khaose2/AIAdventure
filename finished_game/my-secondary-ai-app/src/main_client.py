import json
import os
import random
import requests
import time
import sys
import threading

EVENT_REGISTRY_FILE = "event_registry.json"

def load_event_registry():
    if not os.path.exists(EVENT_REGISTRY_FILE):
        return {"events": []}
    try:
        with open(EVENT_REGISTRY_FILE, "r") as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error decoding {EVENT_REGISTRY_FILE}: {e}")
        backup_filename = EVENT_REGISTRY_FILE + ".bak"
        os.rename(EVENT_REGISTRY_FILE, backup_filename)
        print(f"Existing {EVENT_REGISTRY_FILE} backed up as {backup_filename}. Reinitializing.")
        return {"events": []}

def save_event_registry(registry):
    with open(EVENT_REGISTRY_FILE, "w") as file:
        json.dump(registry, file, indent=4)

def print_events_table(registry):
    events = registry.get("events", [])
    if not events:
        print("No events recorded yet.")
        return
    header = f"{'ID':<4} {'Action':<20} {'Response':<40} {'Season':<8} {'Type':<12} {'Holiday':<10} {'Scene':<10} {'Location':<10}"
    print("\n=== Event Registry ===")
    print(header)
    print("-" * len(header))
    for event in events:
        print(f"{event.get('id', '?'):<4} {event.get('action','')[:18]:<20} {event.get('response','')[:38]:<40} "
              f"{event.get('season',''):<8} {event.get('event_type',''):<12} {event.get('holiday',''):<10} "
              f"{event.get('scene',''):<10} {event.get('location',''):<10}")
    print("=" * len(header) + "\n")

def print_help():
    help_text = """
Welcome to DeepSeek Medieval Dungeon Adventure!

Instructions:
- In Human Play, you guide your character through a medieval dungeon.
- In AI Play, the AIs converse automatically to craft the story.
- Type 'help' to see these instructions again.
- Type 'exit' to quit the game.

Examples:
- Human Play: "I cautiously open the creaky door." or "I go to the river and drink."
- AI Play: The AIs will exchange responses like:
    [Main AI]: "You enter a dim corridor, sensing danger."
    [Secondary AI]: "The corridor reeks of decay; perhaps poison lurks ahead."

Let's begin your adventure!
"""
    print(help_text)

def spinner(stop_event):
    """Displays a spinner in the terminal until stop_event is set."""
    while not stop_event.is_set():
        for ch in '|/-\\':
            sys.stdout.write(f'\rWorking... {ch}')
            sys.stdout.flush()
            if stop_event.wait(0.1):
                break
    sys.stdout.write('\rDone!         \n')

def human_play(mode):
    registry = load_event_registry()
    if not registry.get("events"):
        seed = "Welcome to the Medieval Dungeon Adventure! You find yourself in a dark, musty dungeon. What do you do?"
        print("Starting adventure with seed prompt...")
        stop_event = threading.Event()
        spinner_thread = threading.Thread(target=spinner, args=(stop_event,))
        spinner_thread.start()
        try:
            response = requests.post("http://localhost:8000/hub_action/", json={"prompt": seed, "mode": mode}, timeout=60)
        finally:
            stop_event.set()
            spinner_thread.join()
        if response.ok:
            dm_response = response.json().get("ai_response", response.json().get("main_ai_response", ""))
            print("\nDM:", dm_response, "\n")
            new_event = {"id": 1, "action": seed, "response": dm_response}
            registry.setdefault("events", []).append(new_event)
            save_event_registry(registry)
        else:
            print("Error from game launcher:", response.text)
    while True:
        print_events_table(registry)
        user_input = input("Enter your action (or 'help' for instructions, 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("Exiting adventure. Farewell!")
            break
        elif user_input.lower() == "help":
            print_help()
            continue
        payload = {"prompt": user_input, "mode": mode}
        stop_event = threading.Event()
        spinner_thread = threading.Thread(target=spinner, args=(stop_event,))
        spinner_thread.start()
        try:
            response = requests.post("http://localhost:8000/hub_action/", json=payload, timeout=60)
        finally:
            stop_event.set()
            spinner_thread.join()
        if response.ok:
            json_response = response.json()
            if "ai_response" in json_response:
                ai_response = json_response.get("ai_response", "")
                print("\nDM:", ai_response, "\n")
            elif "main_ai_response" in json_response:
                ai_response = json_response.get("main_ai_response", "")
                print("\nDM:", ai_response, "\n")
            else:
                print("Unexpected response format:", json_response)
                ai_response = "No valid response."
            new_event = {"id": len(registry.get("events", [])) + 1, "action": user_input, "response": ai_response}
            registry.setdefault("events", []).append(new_event)
            save_event_registry(registry)
        else:
            print("Error from game launcher:", response.text)

def ai_play(mode, iterations=10):
    seed = "Welcome to the Medieval Dungeon Adventure! You find yourself in a dark, mysterious dungeon. What happens next?"
    current_prompt = seed
    print("Starting AI Play conversation...\n")
    for i in range(iterations):
        # Main AI response with spinner.
        stop_event = threading.Event()
        spinner_thread = threading.Thread(target=spinner, args=(stop_event,))
        spinner_thread.start()
        for attempt in range(3):
            try:
                response_main = requests.post("http://localhost:8000/hub_action/", json={"prompt": current_prompt, "mode": mode}, timeout=60)
                if response_main.ok:
                    break
            except requests.exceptions.Timeout:
                print("Timeout waiting for main AI. Retrying...")
                if attempt == 2:
                    print("Failed after multiple attempts. Exiting AI conversation.")
                    stop_event.set()
                    spinner_thread.join()
                    return
        stop_event.set()
        spinner_thread.join()
        main_response = response_main.json().get("main_ai_response", response_main.json().get("ai_response", ""))
        print(f"\n[Main AI]: {main_response}")
        registry = load_event_registry()
        new_event = {"id": len(registry.get("events", [])) + 1, "action": current_prompt, "response": main_response}
        registry.setdefault("events", []).append(new_event)
        save_event_registry(registry)
  
        # Secondary AI response with spinner.
        stop_event = threading.Event()
        spinner_thread = threading.Thread(target=spinner, args=(stop_event,))
        spinner_thread.start()
        for attempt in range(3):
            try:
                response_secondary = requests.post("http://localhost:8001/ai_action/", json={"story_memory": [main_response]}, timeout=60)
                if response_secondary.ok:
                    break
            except requests.exceptions.Timeout:
                print("Timeout waiting for secondary AI. Retrying...")
                if attempt == 2:
                    print("Failed after multiple attempts. Exiting AI conversation.")
                    stop_event.set()
                    spinner_thread.join()
                    return
        stop_event.set()
        spinner_thread.join()
        secondary_response = response_secondary.json().get("ai_response", "")
        print(f"\n[Secondary AI]: {secondary_response}")
        registry = load_event_registry()
        new_event = {"id": len(registry.get("events", [])) + 1, "action": main_response, "response": secondary_response}
        registry.setdefault("events", []).append(new_event)
        save_event_registry(registry)
  
        current_prompt = secondary_response
        time.sleep(2)

def main():
    print("Starting DeepSeek Medieval Dungeon Adventure...", flush=True)
    print_help()
    
    while True:
        mode_choice = input("Select play mode: Type 'y' for AI Play (AIs converse automatically) or 'n' for Human Play (you guide the adventure): ")
        if mode_choice.lower().startswith('y'):
            mode = "ai_interaction"
            print("AI Play enabled: AIs will converse automatically.\n")
            break
        elif mode_choice.lower().startswith('n'):
            mode = "user"
            print("Human Play enabled: You will guide the adventure.\n")
            break
        elif mode_choice.lower() == "help":
            print_help()
        else:
            print("Invalid option. Please type 'y' or 'n' (or 'help' for instructions).")
    
    if mode == "user":
        human_play(mode)
    else:
        ai_play(mode, iterations=10)
    
    print("Thank you for playing DeepSeek Medieval Dungeon Adventure!")

if __name__ == "__main__":
    main()
