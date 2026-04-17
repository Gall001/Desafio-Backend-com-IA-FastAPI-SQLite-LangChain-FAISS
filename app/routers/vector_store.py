"""
Router da Questão 3 — Vector Store & Busca Semântica.

Endpoints:
    POST /vector-store/documents          → Indexa um documento
    POST /vector-store/documents/batch    → Indexa múltiplos documentos
    POST /vector-store/search             → Busca semântica por similaridade
    GET  /vector-store/stats              → Estatísticas do índice
    DELETE /vector-store/documents        → Limpa o índice
"""

from fastapi import APIRouter, status

from app.schemas.vector_store import DocumentInput, SearchQuery, SearchResponse, SearchResult
from app.services.vector_store_service import vector_store

router = APIRouter(prefix="/vector-store", tags=["🔍 Busca Semântica"])


@router.post(
    "/documents",
    status_code=status.HTTP_201_CREATED,
    summary="Indexar documento",
    description=(
        "Gera o embedding do documento e o armazena no índice FAISS em memória. "
        "O embedding é gerado com o modelo sentence-transformers/all-MiniLM-L6-v2."
    ),
)
async def add_document(doc: DocumentInput) -> dict:
    doc_id = vector_store.add_document(doc)
    return {
        "message": "Documento indexado com sucesso.",
        "doc_id": doc_id,
        "total_indexed": vector_store.total_documents,
    }


@router.post(
    "/documents/batch",
    status_code=status.HTTP_201_CREATED,
    summary="Indexar múltiplos documentos",
    description="Indexa uma lista de documentos em lote (mais eficiente que chamadas individuais).",
)
async def add_documents_batch(docs: list[DocumentInput]) -> dict:
    if not docs:
        return {"message": "Nenhum documento fornecido.", "indexed": 0}

    ids = vector_store.add_documents_batch(docs)
    return {
        "message": f"{len(ids)} documento(s) indexado(s) com sucesso.",
        "doc_ids": ids,
        "total_indexed": vector_store.total_documents,
    }


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Busca semântica",
    description=(
        "Converte a query em embedding e busca os documentos mais similares usando FAISS (IndexFlatIP). "
        "O score de similaridade é a cosine similarity normalizada (0 a 1, maior = mais relevante)."
    ),
)
async def search(query: SearchQuery) -> SearchResponse:
    """
    Fluxo de busca:
        1. Gera embedding da query com sentence-transformers
        2. Normaliza o vetor (L2) para usar cosine similarity
        3. FAISS calcula produto interno com todos os vetores indexados
        4. Retorna os top_k mais similares ordenados por score
    """
    results = vector_store.search(
        query=query.query,
        top_k=query.top_k,
    )

    return SearchResponse(
        query=query.query,
        results=results,
        total_documents_indexed=vector_store.total_documents,
    )


@router.get(
    "/stats",
    summary="Estatísticas do vector store",
    description="Retorna informações sobre o estado atual do índice FAISS.",
)
async def stats() -> dict:
    return {
        "total_documents": vector_store.total_documents,
        "embedding_dimension": vector_store.dimension,
        "embedding_model": "all-MiniLM-L6-v2",
        "index_type": "IndexFlatIP (cosine similarity)",
    }


@router.delete(
    "/documents",
    summary="Limpar índice",
    description="Remove todos os documentos do índice FAISS (útil para testes).",
)
async def clear_index() -> dict:
    vector_store.clear()
    return {"message": "Índice limpo com sucesso.", "total_indexed": 0}