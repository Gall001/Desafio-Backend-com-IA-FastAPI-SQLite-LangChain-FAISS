"""
Router da Questão 2 — Chatbot com IA Generativa.

Endpoints:
    POST /chatbot/ask          → Envia uma pergunta ao chatbot
    DELETE /chatbot/session/{id} → Limpa o histórico de uma sessão
"""

from fastapi import APIRouter, HTTPException, status

from app.config import settings
from app.schemas.chatbot import ChatMessage, ChatResponse
from app.services.chatbot_service import ask_chatbot, clear_session

router = APIRouter(prefix="/chatbot", tags=["🤖 Chatbot Python"])


@router.post(
    "/ask",
    response_model=ChatResponse,
    summary="Perguntar ao chatbot",
    description=(
        "Envia uma pergunta sobre Python ao chatbot alimentado por LLM (GPT-4o-mini via LangChain). "
        "O histórico de conversa é mantido por session_id, permitindo conversas multi-turno."
    ),
)
async def ask(payload: ChatMessage) -> ChatResponse:
    """
    Fluxo:
        1. Recupera (ou cria) a ConversationChain para o session_id
        2. Envia a pergunta ao LLM com o histórico de mensagens anteriores
        3. Armazena a resposta na memória da sessão
        4. Retorna a resposta ao cliente
    """
    try:
        answer = await ask_chatbot(
            question=payload.question,
            session_id=payload.session_id,
        )
        return ChatResponse(
            answer=answer,
            session_id=payload.session_id,
            model_used=settings.llm_model,
        )
    except Exception as e:
        # Erros comuns: API key inválida, rate limit, sem conexão
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Erro ao consultar o LLM: {str(e)}. Verifique a OPENAI_API_KEY no arquivo .env.",
        )


@router.delete(
    "/session/{session_id}",
    summary="Limpar histórico de sessão",
    description="Remove o histórico de conversa de uma sessão específica.",
)
async def clear(session_id: str) -> dict:
    removed = clear_session(session_id)
    return {
        "message": f"Sessão '{session_id}' removida." if removed else f"Sessão '{session_id}' não encontrada.",
        "removed": removed,
    }