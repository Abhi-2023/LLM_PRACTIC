from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

chat_template = ChatPromptTemplate([
    ("system","You're a helpful {domain} expert"),
    ("human","Explain in simple term, what is {topic}")
])

prompt = chat_template.invoke({
    'domain':"Cricket expert",
    "topic":"Dusra"
})

print(prompt)