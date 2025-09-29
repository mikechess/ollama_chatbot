import ollama

# Use the generate function for a one-off prompt
result = ollama.generate(model='llama3.1:8b', prompt='Why is the sky blue?')
print(result['response'])
