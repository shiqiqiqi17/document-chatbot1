import streamlit as st
import os

# 测试依赖是否安装成功
try:
    from langchain_core.documents import Document
    st.success("✅ langchain-core 安装成功")
except ImportError as e:
    st.error(f"❌ langchain-core 安装失败: {str(e)}")

try:
    from langchain_openai import ChatOpenAI
    st.success("✅ langchain-openai 安装成功")
except ImportError as e:
    st.error(f"❌ langchain-openai 安装失败: {str(e)}")

try:
    from langchain_community.document_loaders import PyPDFLoader
    st.success("✅ langchain-community 安装成功")
except ImportError as e:
    st.error(f"❌ langchain-community 安装失败: {str(e)}")

st.title("📚 Document Chatbot Test")
st.info("检查依赖安装状态")
