from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.llms import Ollama 
from transformers import pipeline

# Load FAISS vector index
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectorstore = FAISS.load_local("index_store", embedding, allow_dangerous_deserialization=True)

# Function to retrieve top matching documents
def retrieve_context(query):
    docs = vectorstore.similarity_search(query, k=5)
    return "\n".join([doc.page_content for doc in docs])

# Load lightweight HuggingFace text generation model

llm = Ollama(model="mistral")


# Generate answer
def ask(query):
    context = retrieve_context(query)
    prompt = f"""You are a helpful assistant.

Context:
{context}

Question: {query}

Answer in clear and detailed manner:"""
    
    result = llm.invoke(prompt)  #  CORRECT METHOD
    return result.strip()

# CLI usage
if __name__ == "__main__":
    while True:
        query = input("\n Ask a question (or type 'exit'): ")
        if query.lower() == 'exit':
            break
        print("\n Answer:", ask(query))
