import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
os.environ['HF_TOKEN']=os.getenv('HF_TOKEN')

llm=ChatGroq(groq_api_key=groq_api_key, model="llama-3.1-8b-instant")

prompt=ChatPromptTemplate.from_template(
    """
{context},
{input}
"""
)

def create_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")
        st.session_state.loader=PyPDFLoader("research_papers/Attention.pdf")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.splitter=RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        st.session_state.final_doc=st.session_state.splitter.split_documents(st.session_state.docs)
        st.session_state.vectors=Chroma.from_documents(st.session_state.final_doc, st.session_state.embeddings)

if st.button('Click to create vector embeddigns'):
    create_embeddings()
    st.write("Vector database created")

user_input=st.text_input("Enter your queries from document")

if user_input:
    doc_chain=create_stuff_documents_chain(llm, prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrivel_chain=create_retrieval_chain(retriever, doc_chain)

    response= retrivel_chain.invoke({"input":user_input})
    st.write(response['answer'])