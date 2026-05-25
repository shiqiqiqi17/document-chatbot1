from typing import Optional, List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from ..services.llm_service import llm_service
from ..services.vectorstore_service import vectorstore_service


condense_question_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the conversation history, rephrase the following question to be a standalone question that contains all necessary context from the conversation.")
])


qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant that answers questions based on the provided context. 
If you cannot find the answer in the context, say "I don't have enough information to answer this question based on the provided documents."
Always be helpful and polite in your responses.

Context: {context}
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])


class ConversationalChainService:
    def __init__(
        self,
        return_source_documents: bool = True,
        verbose: bool = False
    ):
        self.return_source_documents = return_source_documents
        self.verbose = verbose
        self._chain = None
        self._history_chain = None

    def create_chain(self, llm=None, retriever=None):
        if llm is None:
            llm = llm_service.get_llm()

        if retriever is None:
            retriever = vectorstore_service.get_retriever()

        if retriever is None:
            raise ValueError("No retriever available. Please add documents first.")

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self._chain = (
            {
                "context": RunnableLambda(lambda x: retriever.get_relevant_documents(x["input"])) | format_docs,
                "input": RunnablePassthrough(),
                "chat_history": RunnablePassthrough()
            }
            | qa_prompt
            | llm
            | StrOutputParser()
        )
        return self._chain

    def get_chain(self):
        return self._chain

    def get_history_aware_chain(self, session_id: str, get_session_history):
        if self._chain is None:
            self.create_chain()

        self._history_chain = RunnableWithMessageHistory(
            self._chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        return self._history_chain

    def invoke(self, input_dict: Dict[str, Any], config: Optional[Dict] = None):
        if self._chain is None:
            self.create_chain()
        return self._chain.invoke(input_dict, config)

    def invoke_with_history(
        self,
        input_dict: Dict[str, Any],
        session_id: str,
        get_session_history
    ):
        history_chain = self.get_history_aware_chain(session_id, get_session_history)
        return history_chain.invoke(
            input_dict,
            config={"configurable": {"session_id": session_id}}
        )


conversational_chain_service = ConversationalChainService()