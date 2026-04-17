from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
from typing import Optional, Literal, Annotated
from pydantic import BaseModel

load_dotenv()
model = ChatAnthropic(model='claude-sonnet-4-6', max_tokens_to_sample=100)

for chunk in model.stream([
    HumanMessage(content='Write a short story about a spy')
]):
    print(chunk.content, end="", flush=True)

