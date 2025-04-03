from .emulator import Emulator
from .memory_reader import PokemonRedReader
import ollama_client
import time
import os
import base64
import logging
import json
import copy
from PIL import Image
from io import BytesIO
from config import MODEL_NAME, TEMPERATURE, MAX_TOKENS, SYSTEM_PROMPT, SUMMARY_PROMPT, OLLAMA_HOST, USE_NAVIGATOR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MAX_HISTORY = 10  # Maximum number of message pairs to keep in history

AVAILABLE_TOOLS = [
    {
        "name": "press_buttons",
        "description": "Press a sequence of buttons on the Game Boy.",
        "input_schema": {
            "type": "object",
            "properties": {
                "buttons": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["a", "b", "start", "select", "up", "down", "left", "right"]
                    },
                    "description": "List of buttons to press in sequence. Valid buttons: 'a', 'b', 'start', 'select', 'up', 'down', 'left', 'right'"
                },
                "wait": {
                    "type": "boolean",
                    "description": "Whether to wait for a brief period after pressing each button. Defaults to true."
                }
            },
            "required": ["buttons"],
        },
    }
]

if USE_NAVIGATOR:
    AVAILABLE_TOOLS.append({
        "name": "navigate_to",
        "description": "Automatically navigate to a position on the map grid. The screen is divided into a 9x10 grid, with the top-left corner as (0, 0). This tool is only available in the overworld.",
        "input_schema": {
            "type": "object",
            "properties": {
                "row": {
                    "type": "integer",
                    "description": "The row coordinate to navigate to (0-8)."
                },
                "col": {
                    "type": "integer",
                    "description": "The column coordinate to navigate to (0-9)."
                }
            },
            "required": ["row", "col"],
        },
    })


def get_screenshot_base64(screenshot, upscale=None):
    """Convert a PIL Image to base64 encoding, optionally upscaling it."""
    if upscale and upscale > 1:
        width, height = screenshot.size
        screenshot = screenshot.resize((width * upscale, height * upscale), Image.NEAREST)
    
    buffered = BytesIO()
    screenshot.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

class SimpleAgent:
    """Simple agent that plays Pokémon using Ollama."""
    
    def __init__(self, rom_path, headless=False):
        self.emulator = Emulator(rom_path, headless=headless)
        self.running = False
        self.max_history = MAX_HISTORY
        self.message_history = []
        
        # Initialize the Ollama client
        self.client = ollama_client.OllamaClient(
            model_name=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            host=OLLAMA_HOST
        )
        
    
    def get_available_functions(self):
        """Create a dictionary mapping function names to the actual functions"""
        return {
            "press_buttons": self.emulator.press_buttons,
            "press": lambda button: self.emulator.press_buttons(buttons=[button])
        }
    
    def run(self, num_steps=5):
        """Run the agent for a specified number of steps."""
        self.emulator.initialize()
        self.running = True
        steps_completed = 0
        
        # Initial observation
        self._send_initial_observation()
        
        # Main loop
        while self.running and steps_completed < num_steps:
            # Get response from model
            response = self._get_next_action()
            
            # Process tools or text response
            self._process_response(response)
            
            # Increment counter
            steps_completed += 1
            
            # Summarize history if needed
            if len(self.message_history) > self.max_history * 2:
                self.summarize_history()
                
        return steps_completed
            
    def _send_initial_observation(self):
        """Send the initial observation to the model."""
        logger.info(f"[Agent] Initializing with first observation")
        
        # Take a screenshot
        screenshot = self.emulator.get_screenshot()
        screenshot_b64 = get_screenshot_base64(screenshot, upscale=2)
        
        # Build the initial message
        initial_message = {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "You are playing Pokémon Red. What do you want to do? You can press buttons like A, B, START, SELECT, UP, DOWN, LEFT, and RIGHT."
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": screenshot_b64,
                    },
                }
            ]
        }
        
        # Add to history
        self.message_history.append(initial_message)
        
    def _get_next_action(self):
        """Get the next action from the model."""
        logger.info(f"[Agent] Getting next action from model")
        
        # Take a new screenshot for this turn
        screenshot = self.emulator.get_screenshot()
        screenshot_b64 = get_screenshot_base64(screenshot, upscale=2)
        
        # Current game state information
        coordinates = self.emulator.get_coordinates()
        location = self.emulator.get_location()
        
        # Format the message with the new screenshot and game state
        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Current game state:\nLocation: {location}\nCoordinates: {coordinates}\n\nWhat button(s) would you like to press next?"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": screenshot_b64,
                    },
                }
            ]
        }
        
        # Add to history
        self.message_history.append(user_message)
        
        # Convert message format for Ollama
        messages = self._format_messages_for_ollama()
        
        # Get the response from Ollama
        response = self.client.chat_via_client(
            messages=messages,
            tools=AVAILABLE_TOOLS
        )
        
        return response
    
    def _format_messages_for_ollama(self):
        """Format messages for Ollama API.
        
        Ollama expects messages in a different format than Claude.
        This function converts the message_history to a format Ollama understands.
        """
        ollama_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        for msg in self.message_history:
            content = ""
            if isinstance(msg["content"], list):
                for item in msg["content"]:
                    if item["type"] == "text":
                        content += item["text"] + "\n"
                    elif item["type"] == "image":
                        content += "[IMAGE: Game Screenshot]\n"
            else:
                content = msg["content"]
                
            ollama_messages.append({"role": msg["role"], "content": content})
            
        return ollama_messages
        
    def _process_response(self, response):
        """Process the response from the model."""
        logger.info(f"[Agent] Processing model response")
        
        # Extract text content from response
        content = ""
        if "message" in response and "content" in response["message"]:
            content = response["message"]["content"]
        
        # Create assistant message for history
        assistant_message = {
            "role": "assistant",
            "content": content
        }
        
        # Add to history
        self.message_history.append(assistant_message)
        
        # Execute any tool calls
        available_functions = self.get_available_functions()
        results = self.client.call_tool_from_response(response, available_functions)
        
        # Log the results
        if results:
            for result in results:
                logger.info(f"[Agent] Tool called: {result['tool']} with args: {result['args']}")
                if 'error' in result:
                    logger.error(f"[Agent] Tool error: {result['error']}")
                else:
                    logger.info(f"[Agent] Tool result: {result['result']}")
        else:
            # If no tool was called, try to extract button presses from the text response
            logger.info(f"[Agent] No tool calls detected, parsing text response")
            self._parse_and_execute_text_response(content)
        
        # Wait a bit to see the results
        time.sleep(0.5)
    
    def _parse_and_execute_text_response(self, content):
        """Parse the text response for button presses and execute them."""
        # List of valid buttons
        valid_buttons = ["A", "B", "START", "SELECT", "UP", "DOWN", "LEFT", "RIGHT"]
        
        # Simple parsing to find button mentions
        content = content.upper()
        buttons_to_press = []
        
        for button in valid_buttons:
            if button in content:
                buttons_to_press.append(button)
        
        # If buttons were found, press them
        if buttons_to_press:
            logger.info(f"[Agent] Pressing buttons from text: {buttons_to_press}")
            self.emulator.press_buttons(buttons=buttons_to_press)
        else:
            logger.warning(f"[Agent] No buttons found in text response")

    def summarize_history(self):
        """Generate a summary of the conversation history and replace the history with just the summary."""
        logger.info(f"[Agent] Generating conversation summary...")
        
        # Get a new screenshot for the summary
        screenshot = self.emulator.get_screenshot()
        screenshot_b64 = get_screenshot_base64(screenshot, upscale=2)
        
        # Format messages for summarization
        messages = self._format_messages_for_ollama()
        messages.append({
            "role": "user",
            "content": SUMMARY_PROMPT
        })
        
        # Get summary from model
        response = self.client.chat_via_client(messages=messages)
        
        # Extract the summary text
        summary_text = ""
        if "message" in response and "content" in response["message"]:
            summary_text = response["message"]["content"]
        
        logger.info(f"[Agent] Game Progress Summary:")
        logger.info(f"{summary_text}")
        
        # Replace message history with just the summary
        self.message_history = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"CONVERSATION HISTORY SUMMARY (representing {self.max_history} previous messages): {summary_text}"
                    },
                    {
                        "type": "text",
                        "text": "\n\nCurrent game screenshot for reference:"
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screenshot_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "You were just asked to summarize your playthrough so far, which is the summary you see above. You may now continue playing by selecting your next action."
                    },
                ]
            }
        ]
        
        logger.info(f"[Agent] Message history condensed into summary.")
    
    def stop(self):
        """Stop the agent."""
        self.running = False
        self.emulator.stop()
