"""
Serviço de biblioteca — contém a lógica de negócio para CRUD de livros.
Separado do router para facilitar testes unitários independentes de HTTP.
"""

from typing import Optional
from sqlalchemy import select, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.book import Book
from app.schemas.book import BookCreate


class BookService:
    """Encapsula todas as operações de banco relacionadas a livros."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, data: BookCreate) -> Book:
        """
        Persiste um novo livro no banco de dados.

        Args:
            data: Dados validados do livro (BookCreate schema)

        Returns:
            Instância do livro recém-criado com o id gerado
        """
        book = Book(**data.model_dump())
        self.db.add(book)
        await self.db.commit()
        await self.db.refresh(book)  # Recarrega para obter o id gerado
        return book

    async def search(
        self,
        title: Optional[str] = None,
        author: Optional[str] = None,
        skip: int = 0,
        limit: int = 20,
    ) -> tuple[int, list[Book]]:
        """
        Busca livros por título e/ou autor usando LIKE (case-insensitive).

        Args:
            title:  Fragmento do título para filtrar (parcial, case-insensitive)
            author: Fragmento do autor para filtrar (parcial, case-insensitive)
            skip:   Offset para paginação
            limit:  Máximo de resultados retornados

        Returns:
            Tupla (total_encontrado, lista_de_livros)
        """
        # Monta a query base
        stmt = select(Book)
        count_stmt = select(func.count()).select_from(Book)

        # Aplica filtros apenas se os parâmetros foram fornecidos
        filters = []
        if title:
            filters.append(Book.title.ilike(f"%{title}%"))
        if author:
            filters.append(Book.author.ilike(f"%{author}%"))

        if filters:
            # OR: retorna livros que batem em título OU autor
            stmt = stmt.where(or_(*filters))
            count_stmt = count_stmt.where(or_(*filters))

        # Conta total antes de paginar
        total_result = await self.db.execute(count_stmt)
        total = total_result.scalar_one()

        # Aplica paginação
        stmt = stmt.offset(skip).limit(limit)
        result = await self.db.execute(stmt)
        books = list(result.scalars().all())

        return total, books

    async def get_by_id(self, book_id: int) -> Optional[Book]:
        """Retorna um livro pelo id ou None se não encontrado."""
        result = await self.db.execute(select(Book).where(Book.id == book_id))
        return result.scalar_one_or_none()

    async def list_all(self, skip: int = 0, limit: int = 20) -> tuple[int, list[Book]]:
        """Lista todos os livros com paginação."""
        return await self.search(skip=skip, limit=limit)