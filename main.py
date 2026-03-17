from groq import Groq#Importing groq from Groq
from dotenv import load_dotenv#importing dotenv to make sure that this file can run a .env file
import os#to have an access on the system

load_dotenv()#load the .env file that exist in the folder

client = Groq(api_key=os.getenv("GROQ_API_KEY"))#create a variable named 'client' that contains the Groq API Key, and using os to get the .env file

questions = str(input("Enter your question: "))

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "You are a helpful AI tutor for a computer science student."},
        {"role": "user", "content": questions}
    ]
)

print(response.choices[0].message.content)#Print the first answer of the question that already inputted/setted up