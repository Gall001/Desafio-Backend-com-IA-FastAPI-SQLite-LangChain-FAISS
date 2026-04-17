"""
Schemas Pydantic para validação de entrada/saída dos endpoints de livros.
Separa a camada de transporte (API) da camada de persistência (ORM).
"""

from datetime import date
from typing import Optional
from pydantic import BaseModel, Field


class BookCreate(BaseModel):
    """Schema para criação de um novo livro (request body)."""
    title: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Título completo do livro",
        examples=["Clean Code"],
    )
    author: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Nome completo do autor",
        examples=["Robert C. Martin"],
    )
    published_at: date = Field(
        ...,
        description="Data de publicação no formato YYYY-MM-DD",
        examples=["2008-08-01"],
    )
    summary: Optional[str] = Field(
        None,
        description="Resumo ou sinopse do livro",
        examples=["Um guia sobre boas práticas de programação."],
    )


class BookResponse(BookCreate):
    """Schema de resposta — inclui o id gerado pelo banco."""
    id: int = Field(..., description="Identificador único do livro")

    model_config = {"from_attributes": True}  # Permite converter ORM → Pydantic


class BookSearchResult(BaseModel):
    """Schema para resultados de busca — lista paginável de livros."""
    total: int = Field(..., description="Total de resultados encontrados")
    books: list[BookResponse]