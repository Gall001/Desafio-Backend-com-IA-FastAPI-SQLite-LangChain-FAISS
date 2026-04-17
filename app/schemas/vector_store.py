"""
Schemas para o endpoint de busca semântica (Questão 3).
"""

from pydantic import BaseModel, Field


class DocumentInput(BaseModel):
    """Documento a ser indexado no vector store."""
    content: str = Field(
        ...,
        min_length=10,
        description="Conteúdo textual do documento",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Metadados arbitrários (título, fonte, tags, etc.)",
        examples=[{"title": "Introdução ao FastAPI", "source": "blog"}],
    )


class SearchQuery(BaseModel):
    """Query para busca semântica."""
    query: str = Field(
        ...,
        min_length=3,
        description="Texto da consulta semântica",
        examples=["Como usar decorators em Python?"],
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Número de documentos mais similares a retornar",
    )


class SearchResult(BaseModel):
    """Documento encontrado pela busca semântica."""
    content: str
    metadata: dict
    score: float = Field(..., description="Score de similaridade (0 a 1, maior = mais similar)")


class SearchResponse(BaseModel):
    """Resposta da busca semântica com lista de resultados ranqueados."""
    query: str
    results: list[SearchResult]
    total_documents_indexed: int