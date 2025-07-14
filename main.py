import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Load CSV
df = pd.read_csv("data/Training Dataset.csv")
texts = []

# Convert each row to text
for i, row in df.iterrows():
    content = ", ".join([f"{col}: {row[col]}" for col in df.columns])
    texts.append(Document(page_content=content))

# Create embeddings
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Build FAISS vector index
vectorstore = FAISS.from_documents(texts, embedding)
vectorstore.save_local("index_store")

print(" FAISS vector index saved successfully.")
