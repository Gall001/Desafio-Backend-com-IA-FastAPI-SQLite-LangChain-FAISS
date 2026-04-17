"""
Entrypoint da aplicação FastAPI — Backend Assessment.

Integra as três questões em uma única API:
    Q1 → /books          : Biblioteca Virtual (CRUD + busca)
    Q2 → /chatbot        : Chatbot Python com LangChain + GPT-4o-mini
    Q3 → /vector-store   : Busca Semântica com embeddings + FAISS

Execução:
    uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_db
from app.routers import books, chatbot, vector_store
from app.services.vector_store_service import load_sample_documents


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Eventos de ciclo de vida da aplicação:
    - startup: inicializa banco de dados e carrega documentos de exemplo no FAISS
    - shutdown: cleanup (se necessário)
    """
    # --- Startup ---
    print("🚀 Iniciando aplicação...")

    # Q1: Cria tabelas SQLite (se não existirem)
    await init_db()
    print("✅ Banco de dados SQLite inicializado.")

    # Q3: Carrega documentos de exemplo no vector store
    load_sample_documents()
    print("✅ Vector store pronto.")

    yield  # Aplicação rodando

    # --- Shutdown ---
    print("🛑 Encerrando aplicação.")


# ---------------------------------------------------------------------------
# Instância principal do FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="🐍 Backend AI Assessment",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — permite requests de qualquer origem (ajuste em produção)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Registra os routers das três questões
# ---------------------------------------------------------------------------
app.include_router(books.router)
app.include_router(chatbot.router)
app.include_router(vector_store.router)


@app.get("/", tags=["Root"], summary="Health check")
async def root():
    """Verifica se a API está no ar e lista os módulos disponíveis."""
    return {
        "status": "online",
        "message": "Backend AI Assessment API",
        "modules": {
            "Q1 - Biblioteca Virtual": "/books",
            "Q2 - Chatbot Python":     "/chatbot",
            "Q3 - Busca Semântica":    "/vector-store",
        },
        "docs": "/docs",
    }