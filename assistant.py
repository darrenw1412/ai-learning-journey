from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from ddgs import DDGS
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# Load your document for RAG
loader = TextLoader("document.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

# Memory
history = ChatMessageHistory()

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Darren's personal AI assistant. You have access to:
1. Darren's personal document (use this for personal questions)
2. Web search capability (use this for current information)
3. Conversation memory (you remember the whole conversation)

Always be helpful, friendly and concise."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

conversation = RunnableWithMessageHistory(
    chain,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="history"
)

def search_web(query):
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
        if results:
            return "\n".join([r['body'] for r in results])
        return "No results found"

def search_docs(query):
    docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs])

print("Personal AI Assistant ready! Type 'quit' to exit.")
print("Commands: 'web: your question' or 'doc: your question' or just ask normally")
print("-" * 50)

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    if user_input.startswith("web:"):
        query = user_input[4:].strip()
        result = search_web(query)
        print(f"AI (web search): {result}\n")

    elif user_input.startswith("doc:"):
        query = user_input[4:].strip()
        result = search_docs(query)
        print(f"AI (from document): {result}\n")

    else:
        response = conversation.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "darren"}}
        )
        print(f"AI: {response.content}\n")