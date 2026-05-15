from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

DB_PATH = "vector_db"
PDF_PATH = "ICS100.pdf"

# Load PDF
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# VERY IMPORTANT:
# Chunk by meaning, not too small
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

docs = text_splitter.split_documents(documents)

# Embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

# Store
db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory=DB_PATH
)

db.persist()

print("✅ PDF ingested successfully")