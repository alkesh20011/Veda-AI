from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
llm = Ollama(model="llama3")

system_prompt = (
    "You are an ancient Vedic Scholar. Answer the user based ONLY on the provided context. "
    "If the context is in Hindi/Sanskrit, translate the core meaning into English. "
    "Always cite the source book name clearly.\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(db.as_retriever(), question_answer_chain)

print("Oracle is Ready.")
while True:
    user_input = input("\nAsk the Oracle (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    
    print("\nScholar's Answer: ", end="", flush=True)
    
    for chunk in rag_chain.stream({"input": user_input}):
        if "answer" in chunk:
            print(chunk["answer"], end="", flush=True)
            
    print("\n" + "-"*40)