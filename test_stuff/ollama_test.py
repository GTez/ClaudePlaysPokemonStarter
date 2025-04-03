import ollama
try:
    client = ollama.Client(host="http://10.0.8.15:11434")
    models = client.list()
    print("Connection successful!")
    print(models)
except Exception as e:
    print(f"Connection failed: {e}")