from langchain_groq import ChatGroq #import Groq client
from langchain_community.chat_message_histories import ChatMessageHistory #Import ChatMessageHistory client from langchain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder #Import Promp template and 
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Tarot Reader."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

history = ChatMessageHistory()

conversation = RunnableWithMessageHistory(
    chain,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="history"
)

print("Chatbot is ready! Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response = conversation.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": "darren"}}
    )
    print(f"AI: {response.content}")