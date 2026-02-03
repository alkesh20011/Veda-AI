import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


print("Reading your books...")
loader = DirectoryLoader('./data', glob="./*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200, 
    chunk_overlap=200,
    separators=["\n\n", "\n", "।", " ", ""] 
)
chunks = text_splitter.split_documents(documents)
print("Initializing Multilingual Embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
print("Indexing your library... this may take a minute.")
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)
print("Successfully indexed ")