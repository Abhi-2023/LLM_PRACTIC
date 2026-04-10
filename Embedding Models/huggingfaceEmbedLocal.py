from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

embeddingMod = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

vector = embeddingMod.embed_documents(documents)
vector = np.array(vector)
print(vector.shape)