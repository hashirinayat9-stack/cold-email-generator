import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Use caching to prevent the app from slowing down on every click
@st.cache_resource
def initialize_rag():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    drive_db_path = "/content/drive/MyDrive/resume_portfolio"
    resume_db = Chroma(persist_directory=drive_db_path, embedding_function=embeddings)
    return resume_db.as_retriever()

@st.cache_resource
def get_llm(api_key):
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

st.title("🤖 Personalized Cold Email Generator")

if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("API Key missing!")
    st.stop()

# Initialize components
retriever = initialize_rag()
llm = get_llm(api_key)

prompt = ChatPromptTemplate.from_template("""
You are an expert cold email writer. Use the context to write a professional email.
Context: {context}
Job Description: {job_description}
Email:
""")

def format_docs(docs):
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

chain = (
    RunnableParallel({"context": retriever | format_docs, "job_description": RunnablePassthrough()})
    | prompt | llm | StrOutputParser()
)

with st.form("cold_email_form"):
    job_description_text = st.text_area("Paste Job Description:", height=300)
    submitted = st.form_submit_button("Generate Email")

if submitted and job_description_text:
    with st.spinner('Thinking...'):
        try:
            out = chain.invoke(job_description_text)
            st.subheader("Generated Email")
            st.code(out, language='text')
        except Exception as e:
            st.error(f"Error: {e}")
