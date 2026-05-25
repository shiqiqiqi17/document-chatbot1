import os
os.environ["CHROMA_AUTO_FAISS"] = "1"
import streamlit as st
import os
import uuid
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Document Chatbot - RAG System",
    page_icon="📚",
    layout="wide"
)

class RAGSystem:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_BASE_URL"),
            model="text-embedding-3-small"
        )
        self.persist_directory = "./chroma_db"
        self._vectorstore = None
        self._chain = None
        self._chat_history = {}
        
    def get_llm(self):
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_BASE_URL"),
            temperature=0.1
        )

    def load_document(self, file_path: str) -> List[Document]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext in [".txt", ".md"]:
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        return loader.load()

    def split_documents(self, documents: List[Document]) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        return text_splitter.split_documents(documents)

    def create_vectorstore(self, documents: List[Document]):
        self._vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        self._vectorstore.persist()
        return self._vectorstore

    def get_vectorstore(self):
        if self._vectorstore is None and os.path.exists(self.persist_directory):
            self._vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        return self._vectorstore

    def add_documents(self, file_paths: List[str]):
        all_documents = []
        for file_path in file_paths:
            docs = self.load_document(file_path)
            split_docs = self.split_documents(docs)
            all_documents.extend(split_docs)
        
        vectorstore = self.get_vectorstore()
        if vectorstore is None:
            vectorstore = self.create_vectorstore(all_documents)
        else:
            vectorstore.add_documents(all_documents)
            vectorstore.persist()

    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self._chat_history:
            self._chat_history[session_id] = InMemoryChatMessageHistory()
        return self._chat_history[session_id]

    def create_chain(self):
        vectorstore = self.get_vectorstore()
        if vectorstore is None:
            raise ValueError("No documents loaded. Please upload documents first.")
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        llm = self.get_llm()

        template = """
You are a helpful AI assistant that answers questions based on the provided context.
If you cannot find the answer in the context, say "I cannot find the answer in the provided documents."

Context:
{context}

Question: {question}
"""

        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self._chain = (
            {
                "context": RunnableLambda(lambda x: retriever.get_relevant_documents(x["question"])) | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        return self._chain

    def chat(self, question: str, session_id: str) -> Dict[str, Any]:
        if self._chain is None:
            self.create_chain()

        history = self.get_session_history(session_id)
        history.add_user_message(question)

        result = self._chain.invoke({"question": question})
        
        history.add_ai_message(result)

        vectorstore = self.get_vectorstore()
        sources = []
        if vectorstore:
            docs = vectorstore.similarity_search(question, k=4)
            for doc in docs:
                sources.append({
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "N/A")
                })

        return {
            "answer": result,
            "sources": sources
        }


def init_session_state():
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = RAGSystem()
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = os.path.exists("./chroma_db")


def main():
    st.title("📚 Document Chatbot - RAG System")
    st.markdown("**RAG-based Intelligent Document Q&A System**")

    init_session_state()

    with st.sidebar:
        st.header("📁 Document Management")
        uploaded_files = st.file_uploader("Upload PDF or TXT files", 
                                        type=["pdf", "txt", "md"], 
                                        accept_multiple_files=True)
        
        if st.button("Process Documents"):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    temp_dir = "./temp_docs"
                    os.makedirs(temp_dir, exist_ok=True)
                    file_paths = []
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        file_paths.append(file_path)
                    
                    st.session_state.rag_system.add_documents(file_paths)
                    st.session_state.documents_loaded = True
                    st.success(f"Successfully processed {len(file_paths)} document(s)!")
                    
                    for fp in file_paths:
                        os.remove(fp)
            else:
                st.warning("Please upload at least one file first!")
        
        if st.session_state.documents_loaded:
            if st.button("Clear Documents"):
                import shutil
                if os.path.exists("./chroma_db"):
                    shutil.rmtree("./chroma_db")
                st.session_state.rag_system._vectorstore = None
                st.session_state.rag_system._chain = None
                st.session_state.documents_loaded = False
                st.session_state.messages = []
                st.success("Documents cleared!")

    if st.session_state.documents_loaded:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask a question about your documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = st.session_state.rag_system.chat(prompt, st.session_state.session_id)
                        st.markdown(result["answer"])
                        
                        if result["sources"]:
                            with st.expander("📖 View Sources"):
                                for i, source in enumerate(result["sources"], 1):
                                    st.markdown(f"**Source {i}:**")
                                    st.markdown(f"File: {source['source']}")
                                    if source['page'] != "N/A":
                                        st.markdown(f"Page: {source['page']}")
                                    st.markdown(f"Content preview:\n{source['content']}")
                                    st.markdown("---")
                        
                        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    else:
        st.info("Please upload and process documents in the sidebar to start the conversation!")
        st.image("https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=document%20chatbot%20interface%20with%20books%20and%20AI%20icon%20modern%20design&image_size=landscape_16_9", use_column_width=True)


if __name__ == "__main__":
    main()
