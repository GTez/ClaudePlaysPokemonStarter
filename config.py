# Configuration for the application
MODEL_NAME = "hf.co/unsloth/Qwen2.5-Coder-32B-Instruct-GGUF:Q6_K"
TEMPERATURE = 1.0
MAX_TOKENS = 4000

 
SYSTEM_PROMPT = """You are playing Pokemon Red. You can see the game screen and control the game by executing emulator commands.

Your goal is to play through Pokemon Red and eventually defeat the Elite Four. Make decisions based on what you see on the screen.

Before each action, explain your reasoning briefly, then use the emulator tool to execute your chosen commands.

The conversation history may occasionally be summarized to save context space. If you see a message labeled "CONVERSATION HISTORY SUMMARY", this contains the key information about your progress so far. Use this information to maintain continuity in your gameplay."""

SUMMARY_PROMPT = """I need you to create a detailed summary of our conversation history up to this point. This summary will replace the full conversation history to manage the context window.

Please include:
1. Key game events and milestones you've reached
2. Important decisions you've made
3. Current objectives or goals you're working toward
4. Your current location and Pokémon team status
5. Any strategies or plans you've mentioned

The summary should be comprehensive enough that you can continue gameplay without losing important context about what has happened so far."""


USE_NAVIGATOR = False

# Set this to your Ollama server if not using a local installation
# Example: "http://localhost:11434" 
OLLAMA_HOST = "http://10.0.8.15:11434"  # None for local, or set to remote Ollama URL