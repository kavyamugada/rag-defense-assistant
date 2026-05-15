from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

DB_PATH = "vector_db"

# ✅ SAME EMBEDDING MODEL USED FOR SEARCH
embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

# ✅ LLM FOR ANSWERS (UNCHANGED)
llm = Ollama(model="llama3")

def ask_with_rag(question):
    docs = db.similarity_search(question, k=2)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are a defense SOP assistant.
Answer ONLY using the context below.
If the answer is not found, say "Not specified in SOP".

Context:
{context}

Question:
{question}
"""

    return llm.invoke(prompt)

# Test question
question = "Someone enters a restricted area without permission. What should be done immediately?"

print("🔹 Question:")
print(question)

print("\n🔹 LLM + RAG Answer:")
print(ask_with_rag(question))
