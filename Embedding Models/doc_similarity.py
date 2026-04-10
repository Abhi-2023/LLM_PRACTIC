from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
embeddingMod = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

query = 'what is paris ?'

doc_vec = embeddingMod.embed_documents(documents)
query_vec = embeddingMod.embed_query(query)

scores = cosine_similarity([query_vec],doc_vec)
print(scores)

scores = scores.flatten()

index = np.argmax(scores)

print(query)
print(documents[index])