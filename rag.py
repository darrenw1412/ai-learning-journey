from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()

# Load your document
loader = TextLoader("document.txt")
documents = loader.load()

# Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Convert to embeddings and store
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
vectorstore = Chroma.from_documents(chunks, embeddings)

# Connect to Groq
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# Build RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

# Ask questions about your document
print("RAG system ready! Ask questions about your document.")
while True:
    question = input("You: ")
    if question.lower() == "quit":
        break
    answer = qa_chain.invoke(question)
    print(f"AI: {answer['result']}")