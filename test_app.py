from ollama import chat

conversation = [
    {"role": "user", "content": "Hello, how are you?"}
]
reply = chat(model='llama2:7b', messages=conversation)
print(reply.message.content)
