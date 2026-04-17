"""
Configurações centralizadas do projeto via variáveis de ambiente.
Usa pydantic-settings para validação automática.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Chave da API da OpenAI (necessária para Q2 - Chatbot)
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")

    # URL do banco SQLite (Q1 - Biblioteca)
    database_url: str = "sqlite+aiosqlite:///./library.db"

    # Modelo LLM a ser usado no chatbot
    llm_model: str = "gpt-4o-mini"

    # Modelo de embeddings local (Q3 - Vector Store)
    embedding_model: str = "all-MiniLM-L6-v2"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Instância global de configurações
settings = Settings()
