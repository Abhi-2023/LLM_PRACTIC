from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore")

from transformers import logging
logging.set_verbosity_error()

# load_dotenv()

def load_model_tiny_gemma() -> ChatHuggingFace:
    llm = HuggingFacePipeline.from_model_id(
        model_id="google/gemma-2-2b-it",
        task="text-generation",
        pipeline_kwargs=dict(
            temperature=0.5,
            max_new_tokens=100,
            do_sample=False
        )
    )
    return ChatHuggingFace(llm=llm)
model = load_model_tiny_gemma()