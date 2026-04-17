"""
Schemas para o endpoint de chatbot (Questão 2).
"""

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Mensagem enviada pelo usuário ao chatbot."""
    question: str = Field(
        ...,
        min_length=1,
        description="Pergunta sobre programação Python",
        examples=["Como criar uma lista em Python?"],
    )
    session_id: str = Field(
        default="default",
        description="ID da sessão para manter histórico de conversa",
        examples=["user-123"],
    )


class ChatResponse(BaseModel):
    """Resposta gerada pelo LLM."""
    answer: str = Field(..., description="Resposta do chatbot")
    session_id: str = Field(..., description="ID da sessão utilizada")
    model_used: str = Field(..., description="Modelo LLM que gerou a resposta")