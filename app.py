import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from ddgs import DDGS
from dotenv import load_dotenv
import os

load_dotenv()

st.title("🤖 Jarvis — Personal AI Assistant")
st.caption("Built by Darren William | APU CS (AI) Student")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Jarvis, Darren William's personal AI assistant.
Your name is Jarvis. Never change your name.
Be helpful, friendly and concise."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

if "history" not in st.session_state:
    st.session_state.history = ChatMessageHistory()

if "messages" not in st.session_state:
    st.session_state.messages = []

conversation = RunnableWithMessageHistory(
    chain,
    lambda session_id: st.session_state.history,
    input_messages_key="input",
    history_messages_key="history"
)

def search_web(query):
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
        if results:
            return "\n".join([r['body'] for r in results])
        return "No results found"

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask Jarvis anything..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if user_input.startswith("web:"):
            query = user_input[4:].strip()
            response = search_web(query)
        else:
            result = conversation.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": "darren"}}
            )
            response = result.content

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})