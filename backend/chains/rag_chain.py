from typing import Optional, Dict, Any, List, Iterator, AsyncIterator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_core.runnables.config import RunnableConfig

from ..services.llm_service import llm_service
from ..services.vectorstore_service import vectorstore_service


system_prompt = """You are a helpful AI assistant that answers questions based on the provided context.

When answering:
1. Only use information from the provided context
2. If the context doesn't contain enough information, say so
3. Quote relevant parts of the context when appropriate
4. Be clear and concise in your answers

Context: {context}
"""


user_prompt = "{question}"


rag_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", user_prompt)
])


class RAGChainService(RunnableSerializable):
    """RAG Chain Service with LangServe support for high concurrency"""
    
    def __init__(
        self,
        return_source_documents: bool = True,
        verbose: bool = False
    ):
        super().__init__()
        self.return_source_documents = return_source_documents
        self.verbose = verbose
        self._chain = None

    def create_chain(self, llm=None):
        """创建 RAG 链"""
        if llm is None:
            llm = llm_service.get_llm()

        def get_context(input_dict):
            docs = vectorstore_service.similarity_search(
                input_dict["question"],
                k=4
            )
            if not docs:
                return "No relevant documents found."
            context = "\n\n".join([doc.page_content for doc in docs])
            return context

        chain = (
            RunnablePassthrough.assign(context=get_context)
            | rag_prompt
            | llm
            | StrOutputParser()
        )

        self._chain = chain
        return chain

    def get_chain(self):
        """获取当前链"""
        return self._chain

    def invoke(self, input_dict: Dict[str, Any], config: Optional[RunnableConfig] = None) -> str:
        """同步调用链"""
        if self._chain is None:
            self.create_chain()
        return self._chain.invoke(input_dict, config)
    
    async def ainvoke(self, input_dict: Dict[str, Any], config: Optional[RunnableConfig] = None) -> str:
        """异步调用链 - 支持高并发"""
        if self._chain is None:
            self.create_chain()
        return await self._chain.ainvoke(input_dict, config)
    
    def stream(self, input_dict: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Iterator[str]:
        """流式输出 - 支持实时响应"""
        if self._chain is None:
            self.create_chain()
        for chunk in self._chain.stream(input_dict, config):
            yield chunk
    
    async def astream(self, input_dict: Dict[str, Any], config: Optional[RunnableConfig] = None) -> AsyncIterator[str]:
        """异步流式输出 - 支持高并发实时响应"""
        if self._chain is None:
            self.create_chain()
        async for chunk in self._chain.astream(input_dict, config):
            yield chunk
    
    def batch(self, inputs: List[Dict[str, Any]], config: Optional[RunnableConfig] = None) -> List[str]:
        """批量调用 - 支持高吞吐量"""
        if self._chain is None:
            self.create_chain()
        return self._chain.batch(inputs, config)
    
    async def abatch(self, inputs: List[Dict[str, Any]], config: Optional[RunnableConfig] = None) -> List[str]:
        """异步批量调用 - 支持高并发批量处理"""
        if self._chain is None:
            self.create_chain()
        return await self._chain.abatch(inputs, config)

    def get_sources(self, question: str, k: int = 4) -> List[Dict[str, Any]]:
        """获取相关源文档"""
        docs = vectorstore_service.similarity_search(question, k=k)
        sources = []
        for i, doc in enumerate(docs):
            sources.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A")
            })
        return sources


rag_chain_service = RAGChainService()