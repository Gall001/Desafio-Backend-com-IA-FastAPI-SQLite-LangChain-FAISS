"""
Serviço de busca semântica com embeddings + FAISS (Questão 3).

Pipeline:
    1. Documento de texto é recebido via API
    2. sentence-transformers gera um embedding (vetor de floats)
    3. O vetor é armazenado no índice FAISS em memória
    4. Na busca, a query também é transformada em embedding
    5. FAISS calcula similaridade cosine e retorna os top-k mais próximos

Modelo de embedding: all-MiniLM-L6-v2
    - Leve (~80MB), rápido, multilíngue básico
    - 384 dimensões por embedding
    - Ótimo para busca semântica de propósito geral
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.schemas.vector_store import DocumentInput, SearchResult

# ---------------------------------------------------------------------------
# Estado global do vector store (in-memory)
# Em produção, serializar com faiss.write_index() e persistir em disco/S3
# ---------------------------------------------------------------------------

class VectorStoreService:
    """
    Encapsula o índice FAISS e os documentos associados.
    Uma única instância é compartilhada durante o ciclo de vida da aplicação.
    """

    def __init__(self):
        # Carrega o modelo de embeddings na inicialização
        # Primeira execução faz download automático do Hugging Face Hub
        print(f"[VectorStore] Carregando modelo de embeddings: {settings.embedding_model}")
        self.model = SentenceTransformer(settings.embedding_model)

        # Dimensão do espaço de embeddings (depende do modelo)
        self.dimension = self.model.get_sentence_embedding_dimension()

        # IndexFlatIP = produto interno (equivale a cosine similarity quando vetores normalizados)
        self.index = faiss.IndexFlatIP(self.dimension)

        # Armazena os documentos originais em paralelo ao índice FAISS
        # A posição no array corresponde ao índice no FAISS
        self.documents: list[DocumentInput] = []

        print(f"[VectorStore] Pronto. Dimensão dos embeddings: {self.dimension}")

    def _embed(self, text: str) -> np.ndarray:
        """
        Gera o embedding normalizado de um texto.
        A normalização L2 permite usar produto interno como cosine similarity.
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        # Normaliza para que ||v|| = 1 (necessário para cosine via IndexFlatIP)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype("float32")

    def add_document(self, doc: DocumentInput) -> int:
        """
        Indexa um documento no FAISS.

        Args:
            doc: Documento com conteúdo e metadados

        Returns:
            Índice do documento no vector store
        """
        embedding = self._embed(doc.content)

        # FAISS espera shape (n_vetores, dimensão)
        self.index.add(embedding.reshape(1, -1))
        self.documents.append(doc)

        doc_id = len(self.documents) - 1
        return doc_id

    def add_documents_batch(self, docs: list[DocumentInput]) -> list[int]:
        """
        Indexa múltiplos documentos de uma vez (mais eficiente que um por vez).
        """
        if not docs:
            return []

        texts = [doc.content for doc in docs]
        embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=32)

        # Normaliza cada vetor
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Evita divisão por zero
        embeddings = (embeddings / norms).astype("float32")

        self.index.add(embeddings)
        start_id = len(self.documents)
        self.documents.extend(docs)

        return list(range(start_id, start_id + len(docs)))

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """
        Busca os top_k documentos mais semanticamente similares à query.

        Args:
            query: Texto da consulta
            top_k: Número de resultados a retornar

        Returns:
            Lista de SearchResult ordenada por similaridade decrescente
        """
        if self.index.ntotal == 0:
            return []

        # Limita top_k ao total de documentos indexados
        k = min(top_k, self.index.ntotal)

        # Gera embedding da query
        query_embedding = self._embed(query).reshape(1, -1)

        # Busca no FAISS — retorna scores e índices dos k mais próximos
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS retorna -1 quando não encontra resultado
                continue

            doc = self.documents[idx]
            results.append(SearchResult(
                content=doc.content,
                metadata=doc.metadata,
                score=float(score),  # Score cosine normalizado: -1 a 1 (maior = mais similar)
            ))

        return results

    @property
    def total_documents(self) -> int:
        """Total de documentos indexados no vector store."""
        return self.index.ntotal

    def clear(self) -> None:
        """Limpa o índice e os documentos (útil para testes)."""
        self.index.reset()
        self.documents.clear()


# Instância singleton compartilhada pela aplicação
vector_store = VectorStoreService()


# ---------------------------------------------------------------------------
# Documentos de exemplo pré-carregados na inicialização
# Simulam posts de blog sobre Python
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENTS = [
    DocumentInput(
        content="Listas em Python são estruturas de dados mutáveis que permitem armazenar múltiplos itens. "
                "Você pode criar uma lista usando colchetes: minha_lista = [1, 2, 3]. "
                "Listas suportam índices negativos, slicing e métodos como append(), remove() e sort().",
        metadata={"title": "Introdução às Listas em Python", "source": "blog", "tags": ["python", "listas"]},
    ),
    DocumentInput(
        content="Decorators em Python são funções que modificam o comportamento de outras funções. "
                "São definidos com o símbolo @ antes da função. Exemplo: @staticmethod, @property, @functools.wraps. "
                "São amplamente usados em frameworks como Flask e FastAPI para rotas e autenticação.",
        metadata={"title": "Decorators em Python", "source": "blog", "tags": ["python", "decorators"]},
    ),
    DocumentInput(
        content="Programação assíncrona em Python usa async/await para executar tarefas concorrentes. "
                "O módulo asyncio gerencia a event loop. Funções assíncronas são declaradas com 'async def' "
                "e chamadas com 'await'. Ideal para operações I/O bound como requisições HTTP e banco de dados.",
        metadata={"title": "Async/Await em Python", "source": "blog", "tags": ["python", "async", "asyncio"]},
    ),
    DocumentInput(
        content="FastAPI é um framework web moderno e de alta performance para construir APIs com Python. "
                "Usa type hints para validação automática com Pydantic e gera documentação OpenAPI automaticamente. "
                "Suporta operações assíncronas nativamente e tem performance comparável ao Node.js.",
        metadata={"title": "Introdução ao FastAPI", "source": "blog", "tags": ["python", "fastapi", "api"]},
    ),
    DocumentInput(
        content="Generators em Python são funções que produzem valores sob demanda usando a palavra-chave yield. "
                "São eficientes em memória pois não armazenam todos os valores de uma vez. "
                "Generator expressions são similares a list comprehensions mas usam parênteses: (x**2 for x in range(10)).",
        metadata={"title": "Generators e Iteradores em Python", "source": "blog", "tags": ["python", "generators"]},
    ),
    DocumentInput(
        content="Context managers em Python gerenciam recursos com a instrução 'with'. "
                "Implementados via __enter__ e __exit__ ou com @contextmanager do módulo contextlib. "
                "O uso mais comum é para gerenciar arquivos: 'with open(arquivo) as f:' garante o fechamento automático.",
        metadata={"title": "Context Managers em Python", "source": "blog", "tags": ["python", "context-manager"]},
    ),
    DocumentInput(
        content="Type hints em Python adicionam anotações de tipo ao código sem alterar seu comportamento. "
                "São usados por ferramentas como mypy para checagem estática. "
                "Exemplos: def soma(a: int, b: int) -> int, list[str], dict[str, Any], Optional[str].",
        metadata={"title": "Type Hints em Python", "source": "blog", "tags": ["python", "typing", "type-hints"]},
    ),
    DocumentInput(
        content="Dataclasses simplificam a criação de classes que principalmente armazenam dados. "
                "Usando o decorator @dataclass, o Python gera automaticamente __init__, __repr__ e __eq__. "
                "Campos podem ter valores padrão e podem ser frozen (imutáveis) com @dataclass(frozen=True).",
        metadata={"title": "Dataclasses em Python", "source": "blog", "tags": ["python", "dataclasses", "oop"]},
    ),
]


def load_sample_documents() -> None:
    """Carrega os documentos de exemplo no vector store na inicialização da app."""
    if vector_store.total_documents == 0:
        ids = vector_store.add_documents_batch(SAMPLE_DOCUMENTS)
        print(f"[VectorStore] {len(ids)} documentos de exemplo carregados.")