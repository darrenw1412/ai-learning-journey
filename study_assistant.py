import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pypdf import PdfReader
from dotenv import load_dotenv
import os

load_dotenv()

st.title("📚 AI Study Assistant")
st.caption("Upload your lecture notes and get quizzed!")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def generate_questions(text):
    prompt = f"""You are a helpful study assistant. Generate exactly 5 multiple choice questions from the provided text. 
Format each question exactly like this:
Q: [question]
A) [option]
B) [option]
C) [option]
D) [option]
Answer: [correct letter]

Generate 5 quiz questions from this text:

{text[:3000]}"""
    
    response = llm.invoke(prompt)
    return response.content

def check_answer(question, user_answer, correct_answer):
    return user_answer.upper() == correct_answer.upper()

# File upload
uploaded_file = st.file_uploader("Upload your lecture notes", type=["pdf", "txt"])

if uploaded_file:
    # Extract text
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = uploaded_file.read().decode("utf-8")

    st.success(f"✅ File uploaded! ({len(text)} characters)")

    if st.button("Generate Quiz"):
        with st.spinner("Generating questions..."):
            st.session_state.quiz = generate_questions(text)
            st.session_state.score = 0
            st.session_state.answered = 0

if "quiz" in st.session_state:
    st.subheader("📝 Quiz Time!")

    questions = st.session_state.quiz.split("\n\n")

    for i, q in enumerate(questions):
        if q.strip() and q.startswith("Q:"):
            lines = q.strip().split("\n")
            question = lines[0].replace("Q: ", "")
            options = [l for l in lines if l.startswith(("A)", "B)", "C)", "D)"))]
            answer_line = [l for l in lines if l.startswith("Answer:")]

            if options and answer_line:
                correct = answer_line[0].replace("Answer: ", "").strip()
                st.write(f"**Question {i+1}:** {question}")
                user_answer = st.radio(
                    "Choose your answer:",
                    options,
                    key=f"q_{i}"
                )

                if st.button(f"Check Answer {i+1}", key=f"check_{i}"):
                    selected = user_answer[0]
                    if check_answer(question, selected, correct):
                        st.success("✅ Correct!")
                    else:
                        st.error(f"❌ Wrong! The correct answer is {correct}")
                st.divider()