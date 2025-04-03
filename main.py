import argparse
from agent.agent import SimpleAgent
import os
from config import MODEL_NAME, OLLAMA_HOST

def main():
    parser = argparse.ArgumentParser(description="Autonomous agent that plays Pokémon using LLMs")
    parser.add_argument("--rom", default="pokemon.gb", help="Path to the Pokémon ROM file")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps to run")
    parser.add_argument("--visible", action="store_true", help="Show the emulator window")
    parser.add_argument("--host", help="Ollama server host (e.g., http://localhost:11434)")
    args = parser.parse_args()
    
    # Update OLLAMA_HOST if provided
    if args.host:
        os.environ["OLLAMA_HOST"] = args.host
    
    # Print configuration information
    print(f"Starting Pokémon Agent with model: {MODEL_NAME}")
    print(f"Ollama host: {args.host or OLLAMA_HOST or 'local'}")
    print(f"ROM path: {args.rom}")
    print(f"Running for {args.steps} steps")

    # Initialize and run the agent
    agent = SimpleAgent(args.rom, headless=not args.visible)
    try:
        agent.run(num_steps=args.steps)
    except KeyboardInterrupt:
        print("Interrupted by user. Shutting down...")
    finally:
        agent.stop()
        print("Agent stopped")

if __name__ == "__main__":
    main()