"""Ollama Client - Simple client initialization with URL.

Reference: https://pypi.org/project/ollama/
"""

from ollama import Client

# Initialize the client with the URL
client = Client(host="http://localhost:11434")

# ==================== All Ollama Client Methods ====================

print("1. CHAT - Generate a chat completion")
response = client.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": "Why is the sky blue?",
        },
    ],
)
print(response["message"]["content"])

print("2. CHAT - Streaming")
stream = client.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "Why is the sky blue?"}],
    stream=True,
)
for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)

print("3. GENERATE - Generate a completion")
response = client.generate(model="llama3.2", prompt="Why is the sky blue?")
print(response["response"])

print("4. GENERATE - Streaming")
stream = client.generate(
    model="llama3.2",
    prompt="Why is the sky blue?",
    stream=True,
)
for chunk in stream:
    print(chunk["response"], end="", flush=True)

print("5. LIST - List local models")
models = client.list()
print(models)

print("6. SHOW - Show model information")
model_info = client.show("llama3.2")
print(model_info)

print("7. CREATE - Create a model")
response = client.create(model="example", from_="llama3.2", system="You are Mario from Super Mario Bros.")
print(response)

print("8. COPY - Copy a model")
client.copy("llama3.2", "user/llama3.2")

print("9. DELETE - Delete a model")
client.delete("llama3.2")

print("10. PULL - Pull a model")
response = client.pull("llama3.2")
print(response)

print("11. PUSH - Push a model")
response = client.push("user/llama3.2")
print(response)

print("12. EMBED - Generate embeddings (single)")
embeddings = client.embed(model="llama3.2", input="The sky is blue because of rayleigh scattering")
print(embeddings)

print("13. EMBED - Generate embeddings (batch)")
embeddings = client.embed(
    model="llama3.2", input=["The sky is blue because of rayleigh scattering", "Grass is green because of chlorophyll"]
)
print(embeddings)

print("14. PS - List running models")
running = client.ps()
print(running)
