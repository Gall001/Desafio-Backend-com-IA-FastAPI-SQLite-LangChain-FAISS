"""
Modelo de banco de dados para a entidade Livro.
Utiliza SQLAlchemy com suporte assíncrono via aiosqlite.
"""

from datetime import date
from sqlalchemy import String, Text, Date
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Classe base para todos os modelos SQLAlchemy."""
    pass


class Book(Base):
    """
    Representa um livro na biblioteca virtual.

    Campos:
        id          - Chave primária auto-incrementada
        title       - Título do livro (indexado para busca rápida)
        author      - Nome do autor (indexado para busca rápida)
        published_at- Data de publicação
        summary     - Resumo/sinopse do livro
    """
    __tablename__ = "books"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    author: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    published_at: Mapped[date] = mapped_column(Date, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<Book id={self.id} title='{self.title}' author='{self.author}'>"