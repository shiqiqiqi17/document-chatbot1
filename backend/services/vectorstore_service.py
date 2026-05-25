import os
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


class VectorStoreService:
    def __init__(
        self,
        persist_directory: str = "./docs/chroma_db",
        embedding_model: str = "local"
    ):
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self._embeddings = None
        self._vectorstore = None
        self._embeddings_loaded = False

    @property
    def embeddings(self):
        if self._embeddings is None and not self._embeddings_loaded:
            self._embeddings_loaded = True
            try:
                if self.embedding_model == "local":
                    self._embeddings = HuggingFaceEmbeddings(
                        model_name="all-MiniLM-L6-v2",
                        model_kwargs={'device': 'cpu'},
                        encode_kwargs={'normalize_embeddings': True}
                    )
            except Exception as e:
                print(f"Failed to load embeddings: {e}")
                self._embeddings = None
        return self._embeddings

    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        self._vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        return self._vectorstore

    def get_vectorstore(self) -> Optional[Chroma]:
        if self._vectorstore is None:
            if os.path.exists(self.persist_directory):
                self._vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
        return self._vectorstore

    def add_documents(self, documents: List[Document]) -> None:
        vectorstore = self.get_vectorstore()
        if vectorstore is None:
            vectorstore = self.create_vectorstore(documents)
        else:
            vectorstore.add_documents(documents)
            vectorstore.persist()

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None
    ) -> List[Document]:
        vectorstore = self.get_vectorstore()
        if vectorstore is None:
            return []
        return vectorstore.similarity_search(query, k=k, filter=filter)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None
    ) -> List[Tuple[Document, float]]:
        vectorstore = self.get_vectorstore()
        if vectorstore is None:
            return []
        return vectorstore.similarity_search_with_score(query, k=k, filter=filter)

    def get_retriever(self, search_kwargs: Optional[dict] = None):
        vectorstore = self.get_vectorstore()
        if vectorstore is None:
            return None
        if search_kwargs is None:
            search_kwargs = {"k": 4}
        return vectorstore.as_retriever(search_kwargs=search_kwargs)

    def clear(self):
        if os.path.exists(self.persist_directory):
            import shutil
            shutil.rmtree(self.persist_directory)
        self._vectorstore = None


vectorstore_service = VectorStoreService(embedding_model="local")
