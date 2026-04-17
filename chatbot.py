from model import model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

messages = []

system_message = input("Give a system message : ")
messages.append(SystemMessage(content=system_message))

while(True):
    user_input = input("You : ")
    messages.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(messages)
    response = result.content

    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1]
        messages.append(AIMessage(content=response))
        print("Assistant :",response.strip())
        
print(messages)
messages.clear()