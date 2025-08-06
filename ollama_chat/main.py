
from ollama_client import stream_chat
from config import MODEL_NAME

def main():
    """
    Main function to run the chat application.
    """
    print(f"Using model: {MODEL_NAME}")
    print("Enter your prompt (or 'quit' to exit):")

    while True:
        try:
            prompt = input("> ")
            if prompt.lower() == 'quit':
                break

            for chunk in stream_chat(MODEL_NAME, prompt):
                print(chunk, end="", flush=True)
            print()  

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

if __name__ == "__main__":
    main()
