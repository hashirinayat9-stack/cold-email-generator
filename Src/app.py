# 1. Create the Streamlit App File (app.py)
# This MUST be the first line in this cell.
import streamlit as st
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# --- API Key Setup ---

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=st.secrets["GOOGLE_API_KEY"]
)
# --- Streamlit UI ---
st.title("🤖 Personalized Cold Email Generator")
st.markdown("Use this tool to draft a cold email to a recruiter, powered by your RAG-based portfolio data.")

# --- RAG Setup ---
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)
# Define the correct local path with the nested folder name
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Joining the paths to get the full correct path:

local_db_path = os.path.join(root_path, "resume_portfolio-db", "resume_portfolio") 

# Check if the database path exists before trying to load it
if not os.path.isdir(local_db_path):
    st.error(f"Error: The Chroma DB path was not found: {local_db_path}. Please check Google Drive mount and path.")
    st.stop()

# Initialize Chroma and Retriever
resume_db = Chroma(persist_directory=local_db_path, embedding_function=embeddings)
retriever = resume_db.as_retriever()

# Document formatting function
def format_docs(docs):
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

# Prompt definition
prompt_str = """You are an expert cold email writer. Your task is to write a personalized, professional cold email to a recruiter for the provided job description.
MANDATORY: Use the context below to highlight relevant skills and experience that match the job description.
Output ONLY the email content. Do not include any introductory phrases or greetings to yourself.

Context (Candidate's Portfolio Data):
{context}

Job Description (The position the candidate is applying for):
{job_description}

Email:
"""
prompt = ChatPromptTemplate.from_template(prompt_str)

# Define the RAG chain components
retrieval = RunnableParallel(
    {"context": retriever | format_docs, "job_description": RunnablePassthrough()}
)
chain = retrieval | prompt | llm | StrOutputParser()

# --- Input Form ---
with st.form("cold_email_form"):
    job_description_text = st.text_area(
        "Paste the Job Description Here:",
        "Example: Seeking a Senior Data Scientist with 5+ years experience in Python, NLP, and model deployment on AWS.",
        height=300
    )
    submitted = st.form_submit_button("Generate Email")

# --- Run Chain and Output ---
if submitted and job_description_text:
    with st.spinner('Generating personalized email using RAG...'):
        try:
            out = chain.invoke(job_description_text)
            st.subheader("Generated Email")
            st.code(out, language='text')
        except Exception as e:
            st.error(f"An error occurred during chain execution: {e}")