"""
Router da Questão 1 — Biblioteca Virtual.

Endpoints:
    POST   /books/           → Cadastra um novo livro
    GET    /books/           → Lista todos os livros (paginado)
    GET    /books/search     → Busca por título e/ou autor
    GET    /books/{id}       → Busca livro por id
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.book import BookCreate, BookResponse, BookSearchResult
from app.services.book_service import BookService

router = APIRouter(prefix="/books", tags=["📚 Biblioteca"])


@router.post(
    "/",
    response_model=BookResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Cadastrar livro",
    description="Cadastra um novo livro na biblioteca virtual com título, autor, data e resumo.",
)
async def create_book(
    payload: BookCreate,
    db: AsyncSession = Depends(get_db),
) -> BookResponse:
    """
    Recebe os dados do livro, valida via Pydantic e persiste no SQLite.
    Retorna o livro criado com o id gerado automaticamente.
    """
    service = BookService(db)
    book = await service.create(payload)
    return book


@router.get(
    "/",
    response_model=BookSearchResult,
    summary="Listar todos os livros",
    description="Retorna todos os livros cadastrados com suporte a paginação.",
)
async def list_books(
    skip: int = Query(default=0, ge=0, description="Offset para paginação"),
    limit: int = Query(default=20, ge=1, le=100, description="Máximo de resultados"),
    db: AsyncSession = Depends(get_db),
) -> BookSearchResult:
    service = BookService(db)
    total, books = await service.list_all(skip=skip, limit=limit)
    return BookSearchResult(total=total, books=books)


@router.get(
    "/search",
    response_model=BookSearchResult,
    summary="Buscar livros",
    description=(
        "Busca livros por título e/ou autor (parcial, case-insensitive). "
        "Se ambos forem fornecidos, retorna livros que batem em qualquer um dos campos (OR)."
    ),
)
async def search_books(
    title: Optional[str] = Query(None, description="Fragmento do título"),
    author: Optional[str] = Query(None, description="Fragmento do nome do autor"),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> BookSearchResult:
    if not title and not author:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Informe ao menos um parâmetro de busca: 'title' ou 'author'.",
        )

    service = BookService(db)
    total, books = await service.search(title=title, author=author, skip=skip, limit=limit)
    return BookSearchResult(total=total, books=books)


@router.get(
    "/{book_id}",
    response_model=BookResponse,
    summary="Buscar livro por ID",
    description="Retorna um livro específico pelo seu identificador único.",
)
async def get_book(
    book_id: int,
    db: AsyncSession = Depends(get_db),
) -> BookResponse:
    service = BookService(db)
    book = await service.get_by_id(book_id)

    if not book:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Livro com id={book_id} não encontrado.",
        )

    return book