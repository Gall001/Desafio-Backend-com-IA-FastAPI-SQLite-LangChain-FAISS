"""
Serviço de chatbot utilizando LangChain + OpenAI (Questão 2).

Arquitetura:
    - ConversationBufferWindowMemory: mantém as últimas N mensagens por sessão
    - ChatOpenAI: wrapper LangChain para GPT-4o-mini
    - ConversationChain: orquestra memória + LLM + prompt

O chatbot é especializado em Python — o system prompt o instrui a focar nesse tema.
"""

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder

from app.config import settings

# Dicionário que mapeia session_id → instância da chain
# Em produção, substituir por Redis para suportar múltiplos workers
_session_chains: dict[str, ConversationChain] = {}

# System prompt que especializa o bot em Python
SYSTEM_PROMPT = """Você é um assistente especialista em programação Python, amigável e didático.
Responda sempre em português do Brasil.
Forneça exemplos de código quando apropriado, usando blocos markdown.
Se a pergunta não for sobre Python ou programação, redirecione educadamente o usuário.
Seja conciso mas completo nas suas explicações."""


def _build_prompt() -> ChatPromptTemplate:
    """Constrói o template de prompt com system message e histórico."""
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ])


def _get_or_create_chain(session_id: str) -> ConversationChain:
    """
    Recupera a chain de conversa existente ou cria uma nova para a sessão.

    Mantém janela de 10 mensagens para não estourar o context window.
    """
    if session_id not in _session_chains:
        llm = ChatOpenAI(
            model=settings.llm_model,
            openai_api_key=settings.openai_api_key,
            temperature=0.7,  # Balanceia criatividade e coerência
        )

        memory = ConversationBufferWindowMemory(
            k=10,                   # Guarda as últimas 10 trocas de mensagem
            return_messages=True,   # Retorna objetos Message (necessário para ChatPromptTemplate)
        )

        chain = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=_build_prompt(),
            verbose=False,
        )
        _session_chains[session_id] = chain

    return _session_chains[session_id]


async def ask_chatbot(question: str, session_id: str = "default") -> str:
    """
    Envia uma pergunta ao chatbot e retorna a resposta.

    Args:
        question:   Texto da pergunta do usuário
        session_id: Identificador da sessão (mantém histórico separado por usuário)

    Returns:
        Resposta em texto gerada pelo LLM
    """
    chain = _get_or_create_chain(session_id)

    # ConversationChain.predict é síncrono; para produção usar ainvoke
    # Aqui usamos invoke que é seguro em contexto async para operações rápidas
    response = chain.invoke({"input": question})

    # O campo "response" contém a resposta do LLM
    return response.get("response", "Desculpe, não consegui gerar uma resposta.")


def clear_session(session_id: str) -> bool:
    """Remove o histórico de uma sessão específica."""
    if session_id in _session_chains:
        del _session_chains[session_id]
        return True
    return False