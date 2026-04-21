from langchain_ollama import OllamaLLM

def main():
    prompt = "Hello! How are you today?"

    # Use the model you actually have installed
    ollama = OllamaLLM(model="gemma3:4b")

    # Generate a response (takes a list of prompts)
    response = ollama.generate([prompt])

    # The generated text is in response.generations[0][0].text
    print(response.generations[0][0].text)

if __name__ == "__main__":
    main()
