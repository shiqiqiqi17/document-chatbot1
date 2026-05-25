import os
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentService:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
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

    def load_and_split_documents(self, file_paths: List[str]) -> List[Document]:
        all_documents = []
        for file_path in file_paths:
            try:
                docs = self.load_document(file_path)
                for doc in docs:
                    doc.metadata["source"] = os.path.basename(file_path)
                all_documents.extend(docs)
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                continue

        if not all_documents:
            return []

        return self.text_splitter.split_documents(all_documents)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        return self.text_splitter.split_documents(documents)

    def process_files(self, file_paths: List[str]):
        from .vectorstore_service import vectorstore_service
        documents = self.load_and_split_documents(file_paths)
        if documents:
            vectorstore_service.add_documents(documents)


document_service = DocumentService()
