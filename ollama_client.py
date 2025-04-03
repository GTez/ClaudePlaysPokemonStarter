import ollama
import os
from typing import List, Dict, Any, Optional

class OllamaClient:
    """Client for interacting with Gemma3 via Ollama"""
    
    def __init__(
        self, 
        model_name: str = "gemma3:27b", 
        temperature: float = 1.0, 
        max_tokens: int = 4000,
        host: Optional[str] = None
    ):
        """
        Initialize the Ollama client.
        
        Args:
            model_name: Name of the model to use
            temperature: Temperature for text generation
            max_tokens: Maximum tokens to generate
            host: Optional remote Ollama server host (e.g., "http://localhost:11434")
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Configure client to use remote server if specified
        if host:
            self.client = ollama.Client(
                host=host
            )
        else:
            # Use local Ollama server if no host is specified
            self.client = ollama.Client()

        
        # Check if model is available
        try:
            models = self.client.list()
            available_models = [model["model"] for model in models.get("models", [])]
            if model_name not in available_models:
                print(f"Warning: {model_name} not found in available models. You may need to pull it first.")
                print(f"Available models: {', '.join(available_models)}")
        except Exception as e:
            print(f"Could not check for model availability: {e}")
    
    def chat_via_client(
        self, 
        messages: List[Dict[str, str]], 
        tools: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Send a chat request to the Ollama model.
        
        Args:
            messages: List of messages in the conversation
            tools: Optional list of functions to provide as tools
            
        Returns:
            The response from the model
        """
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
                tools=tools if tools else None
            )
            return response
        except Exception as e:
            print(f"Error communicating with Ollama: {e}")
            return {"message": {"content": f"Error: {str(e)}"}}
    
    def call_tool_from_response(self, response, available_functions):
        """
        Execute tool calls from model response
        
        Args:
            response: The response from the model
            available_functions: Dictionary mapping function names to callable functions
            
        Returns:
            Tool execution results
        """
        results = []
        
        # Check if there are any tool calls in the response
        if hasattr(response.message, 'tool_calls') and response.message.tool_calls:
            for tool in response.message.tool_calls:
                function_name = tool.function.name
                function_args = tool.function.arguments
                
                # Get the function to call
                function_to_call = available_functions.get(function_name)
                if function_to_call:
                    try:
                        # Call the function with the provided arguments
                        result = function_to_call(**function_args)
                        results.append({
                            "tool": function_name,
                            "args": function_args,
                            "result": result
                        })
                    except Exception as e:
                        results.append({
                            "tool": function_name,
                            "args": function_args,
                            "error": str(e)
                        })
                else:
                    results.append({
                        "tool": function_name,
                        "args": function_args,
                        "error": f"Function {function_name} not found"
                    })
                    
        return results