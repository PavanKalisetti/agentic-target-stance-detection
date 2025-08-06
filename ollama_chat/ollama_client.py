
import requests
import json

OLLAMA_API_URL = "http://localhost:11434/api/chat"

def stream_chat(model, prompt):
    """
    Streams the chat response from the Ollama API.
    """
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }

    with requests.post(OLLAMA_API_URL, json=data, stream=True) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                try:
                    json_data = json.loads(line)
                    if "content" in json_data["message"]:
                        yield json_data["message"]["content"]
                except json.JSONDecodeError:
                    
                    pass
