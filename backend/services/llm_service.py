import os
from typing import Optional, Any, Dict
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_core.language_models import BaseChatModel
from langchain_core.callbacks import CallbackManager


class LLMService:
    def __init__(
        self,
        model_provider: str = "openai",
        model_name: str = "deepseek-chat",
        temperature: float = 0.7,
        **kwargs
    ):
        self.model_provider = model_provider
        self.model_name = model_name
        self.temperature = temperature
        self.kwargs = kwargs
        self._llm: Optional[BaseChatModel] = None

    def get_llm(self, **overrides) -> BaseChatModel:
        if self._llm is None or overrides:
            temperature = overrides.get("temperature", self.temperature)

            if self.model_provider == "openai":
                api_key = overrides.get("api_key") or os.getenv("OPENAI_API_KEY") or "sk-98a167a653c54fda93cdfc2eaa2058d8"
                base_url = overrides.get("base_url") or os.getenv("OPENAI_BASE_URL") or "https://api.deepseek.com/v1"
                model_name = overrides.get("model_name") or os.getenv("OPENAI_MODEL") or "deepseek-chat"
                llm_kwargs = {
                    "model": model_name,
                    "temperature": temperature,
                    "api_key": api_key,
                    "streaming": overrides.get("streaming", False),
                    "callback_manager": overrides.get("callback_manager")
                }
                if base_url:
                    llm_kwargs["base_url"] = base_url
                self._llm = ChatOpenAI(**llm_kwargs)
            elif self.model_provider == "ollama":
                base_url = overrides.get("base_url") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                self._llm = Ollama(
                    base_url=base_url,
                    model=self.model_name,
                    temperature=temperature,
                    **self.kwargs
                )
            else:
                raise ValueError(f"Unknown model provider: {self.model_provider}")

        return self._llm

    def create_llm(self, **overrides) -> BaseChatModel:
        return self.get_llm(**overrides)


llm_service = LLMService()
