# Simple AI Agent
import ollama 

""" Tool function: add two numbers """
def add_two_numbers(a: int, b: int) -> int:
    """
    Add two numbers and return the result.
    """
    return a + b

""" System prompt to inform the model about the tool is usage """
system_message = {
    "role": "system", 
    "content": "You are a helpful assistant. You can do math by calling a function 'add_two_numbers' if needed."
}
# User asks a question that involves a calculation
user_message = {
    "role": "user", 
    "content": "What is 10 + 10?"
}
messages = [system_message, user_message]

response = ollama.chat(
    model='llama3.1:8b', 
    messages=messages,
    tools=[add_two_numbers]  # pass the actual function object as a tool
)

if response.message.tool_calls:
    for tool_call in response.message.tool_calls:
        func_name = tool_call.function.name   # e.g., "add_two_numbers"
        args = tool_call.function.arguments   # e.g., {"a": 10, "b": 10}
        # If the function name matches and we have it in our tools, execute it:
        if func_name == "add_two_numbers":
            result = add_two_numbers(**args)
            print("Function output:", result)

""" (Continuing from previous code) """
available_functions = {"add_two_numbers": add_two_numbers}

""" Model's initial response after possibly invoking the tool """
assistant_reply = response.message.content
print("Assistant (initial):", assistant_reply)

""" If a tool was called, handle it """
for tool_call in (response.message.tool_calls or []):
    func = available_functions.get(tool_call.function.name)
    if func:
        result = func(**tool_call.function.arguments)
        # Provide the result back to the model in a follow-up message
        messages.append({"role": "assistant", "content": f"The result is {result}."})
        follow_up = ollama.chat(model='llama3.1:8b', messages=messages)
        print("Assistant (final):", follow_up.message.content)



