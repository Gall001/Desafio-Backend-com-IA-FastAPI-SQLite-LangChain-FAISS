"""
Configuração do banco de dados assíncrono com SQLAlchemy + aiosqlite.
Fornece a sessão de banco como dependência injetável no FastAPI.
"""

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.config import settings
from app.models.book import Base

# Engine assíncrona — aiosqlite permite que o SQLite não bloqueie a event loop
engine = create_async_engine(
    settings.database_url,
    echo=False,          # Mude para True para ver queries no console
    future=True,
)

# Fábrica de sessões assíncronas
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Evita lazy-load após commit
)


async def init_db() -> None:
    """Cria as tabelas no banco caso ainda não existam."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    """
    Dependência FastAPI que fornece uma sessão de BD por request.
    Garante que a sessão seja fechada mesmo em caso de erro.
    """
    async with AsyncSessionLocal() as session:
        yield session