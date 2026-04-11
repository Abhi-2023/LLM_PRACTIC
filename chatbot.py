from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore")

from transformers import logging
logging.set_verbosity_error()

load_dotenv()

def load_model():
    llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task='text-generation',
        pipeline_kwargs=dict(
            temperature=0.5,
            max_new_tokens=100,
            do_sample=False
        )
    )
    return ChatHuggingFace(llm=llm)

model = load_model()
chat_history = []
while(True):
    user_input = input("You : ")
    chat_history.append(user_input)
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    response = result.content

    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1]
        chat_history.append(response)
        print(response.strip())
    