from typing import Optional, Dict, Any, List
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.agents import create_openai_tools_agent, AgentExecutor

from ..services.llm_service import llm_service
from ..services.vectorstore_service import vectorstore_service
from ..services.document_service import document_service


class DocumentSearchTool(BaseTool):
    """用于搜索文档内容的工具"""
    name = "document_search"
    description = "搜索文档内容，用于回答用户关于上传文档的问题"
    
    def _run(self, query: str) -> str:
        docs = vectorstore_service.similarity_search(query, k=4)
        if not docs:
            return "没有找到相关文档"
        context = "\n\n".join([doc.page_content for doc in docs])
        return context


class DocumentStatsTool(BaseTool):
    """用于获取文档统计信息的工具"""
    name = "document_stats"
    description = "获取当前已上传文档的统计信息"
    
    def _run(self) -> str:
        vectorstore = vectorstore_service.get_vectorstore()
        if vectorstore:
            try:
                doc_count = len(vectorstore.get()["ids"])
            except:
                doc_count = 0
            return f"当前向量数据库中共有 {doc_count} 个文档片段"
        return "向量数据库未初始化"


class RAGToolService:
    """RAG工具服务"""
    
    def __init__(self):
        self.tools = [
            DocumentSearchTool(),
            DocumentStatsTool()
        ]
        self._agent_executor = None
    
    def get_tools(self):
        """获取所有工具"""
        return self.tools
    
    def create_agent(self, llm=None):
        """创建带有工具调用的Agent"""
        if llm is None:
            llm = llm_service.get_llm()
        
        system_prompt = """你是一个智能助手，可以使用工具来帮助回答用户问题。
        
可用工具：
1. document_search - 搜索文档内容
2. document_stats - 获取文档统计信息

当用户问关于上传文档的问题时，使用 document_search 工具。
当用户问有多少文档时，使用 document_stats 工具。
"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{input}"),
        ])
        
        agent = create_openai_tools_agent(llm, self.tools, prompt)
        self._agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)
        return self._agent_executor
    
    def invoke(self, input_dict: Dict[str, Any], config: Optional[Dict] = None):
        """调用工具执行"""
        if self._agent_executor is None:
            self.create_agent()
        return self._agent_executor.invoke(input_dict, config)


rag_tool_service = RAGToolService()