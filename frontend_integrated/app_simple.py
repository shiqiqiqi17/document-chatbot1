import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Test RAG", layout="wide")
st.title("Test RAG System")

# Test basic functionality
st.write("Environment variables loaded:", os.getenv("OPENAI_API_KEY") is not None)

try:
    from langchain_openai import ChatOpenAI
    st.success("langchain_openai imported successfully")
except Exception as e:
    st.error(f"Error importing langchain_openai: {e}")

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    st.success("langchain_text_splitters imported successfully")
except Exception as e:
    st.error(f"Error importing langchain_text_splitters: {e}")

try:
    from langchain_community.document_loaders import PyPDFLoader
    st.success("PyPDFLoader imported successfully")
except Exception as e:
    st.error(f"Error importing PyPDFLoader: {e}")

try:
    from langchain_community.vectorstores import Chroma
    st.success("Chroma imported successfully")
except Exception as e:
    st.error(f"Error importing Chroma: {e}")

st.write("All imports completed!")