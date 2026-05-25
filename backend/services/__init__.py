from .document_service import DocumentService, DocumentService as _DocumentService
from .vectorstore_service import VectorStoreService, VectorStoreService as _VectorStoreService
from .llm_service import LLMService, LLMService as _LLMService

def get_document_service():
    return _DocumentService()

def get_vectorstore_service():
    return _VectorStoreService(embedding_model="local")

def get_llm_service():
    return _LLMService()

document_service = _DocumentService()
vectorstore_service = _VectorStoreService(embedding_model="local")
llm_service = _LLMService()

__all__ = [
    "DocumentService",
    "document_service",
    "get_document_service",
    "VectorStoreService",
    "vectorstore_service",
    "get_vectorstore_service",
    "LLMService",
    "llm_service",
    "get_llm_service"
]
