from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class ChatMessage(BaseModel):
    role: str = Field(default="user", description="Message role: user or assistant")
    content: str = Field(..., description="Message content")


class DocumentUploadResponse(BaseModel):
    success: bool
    message: str
    documents_processed: int = 0
    file_names: List[str] = []


class ChatRequest(BaseModel):
    question: str = Field(..., description="User's question")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation memory")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    session_id: str


class ConversationHistoryResponse(BaseModel):
    session_id: str
    messages: List[ChatMessage]
