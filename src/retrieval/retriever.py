# src/retrieval/retriever.py
import cohere
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from config import (
    COHERE_API_KEY, COHERE_EMBED_MODEL, COHERE_RERANK_MODEL,
    PINECONE_API_KEY, PINECONE_HYBRID_INDEX_NAME,
    TOP_K_RETRIEVAL, TOP_K_FETCH, HYBRID_ALPHA
)


def _hybrid_scale(
    dense: list[float],
    sparse: dict,
    alpha: float
) -> tuple[list[float], dict]:
    """
    Weight the dense and sparse vectors by alpha before passing to Pinecone.

    Pinecone combines the two scores as:
        final_score = dense_score + sparse_score

    So to implement  alpha * dense + (1-alpha) * sparse  we scale the vectors
    themselves. Pinecone then sums them and the weighting is preserved.

    alpha=1.0 → pure dense (semantic only, same as Phase 1)
    alpha=0.0 → pure sparse (BM25 keyword only)
    alpha=0.75 → our default: semantic-leaning but keyword-aware
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")

    scaled_dense = [v * alpha for v in dense]
    scaled_sparse = {
        "indices": sparse["indices"],
        "values":  [v * (1 - alpha) for v in sparse["values"]],
    }
    return scaled_dense, scaled_sparse


def retrieve(
    query: str,
    top_k: int = TOP_K_RETRIEVAL,
    filter_priority: str | None = None,
    alpha: float = HYBRID_ALPHA,
) -> list[dict]:
    """
    Hybrid retrieval (dense + BM25 sparse) followed by Cohere reranking.

    Stage 1 — Hybrid fetch (approximate, but fast):
      Encodes the query as both a dense vector (Cohere bi-encoder) and a
      sparse vector (BM25). Scales them by alpha, then queries Pinecone for
      the top TOP_K_FETCH candidates. Pinecone scores each stored vector as:
          score = alpha * cosine(query_dense, doc_dense)
                + (1-alpha) * bm25(query_sparse, doc_sparse)

    Stage 2 — Rerank (precise, but only runs on TOP_K_FETCH candidates):
      Cohere's reranker is a cross-encoder: it reads the full query and each
      chunk *together* in a single forward pass, giving it much richer context
      than the bi-encoder used in Stage 1. We ask it to pick the best top_k
      from among the candidates.

    The fetch-then-rerank pattern is the industry standard because:
    - Cross-encoders are too slow to run against the full index (millions of chunks)
    - Bi-encoders are fast but approximate
    - Combining them gives you the speed of one and the accuracy of the other
    """
    co    = cohere.Client(api_key=COHERE_API_KEY)
    index = Pinecone(api_key=PINECONE_API_KEY).Index(PINECONE_HYBRID_INDEX_NAME)
    bm25  = BM25Encoder.default()

    # --- Stage 1: encode query as dense + sparse ------------------------------
    dense_response = co.embed(
        texts      = [query],
        model      = COHERE_EMBED_MODEL,
        input_type = "search_query",
    )
    # Cohere SDK v5 type stubs annotate .embeddings as EmbedByTypeResponseEmbeddings
    # rather than list[list[float]], so Pylance rejects [0]. At runtime this is
    # a plain list — the type: ignore suppresses the false positive.
    dense_vec: list[float] = dense_response.embeddings[0]  # type: ignore[index]
    sparse_vec = bm25.encode_queries(query)   # note: encode_queries (not encode_documents)

    dense_scaled, sparse_scaled = _hybrid_scale(dense_vec, sparse_vec, alpha)

    # --- Stage 1: fetch from Pinecone --------------------------------------------
    pinecone_filter = {}
    if filter_priority:
        pinecone_filter["priority"] = {"$eq": filter_priority}

    results = index.query(
        vector           = dense_scaled,
        sparse_vector    = sparse_scaled,  # type: ignore[arg-type]
        top_k            = TOP_K_FETCH,         # fetch more than we need
        include_metadata = True,
        filter           = pinecone_filter if pinecone_filter else None,
    )

    candidates = [
        {
            "text":     match["metadata"]["text"],
            "score":    match["score"],
            "source":   match["metadata"].get("source", "unknown"),
            "page":     match["metadata"].get("page", "?"),
            "priority": match["metadata"].get("priority", "normal"),
        }
        for match in results.matches  # type: ignore[union-attr]
    ]

    if not candidates:
        return []

    # --- Stage 2: rerank with Cohere cross-encoder --------------------------------------------
    # The reranker receives the original query and each candidate chunk as text.
    # It returns relevance scores that are much more nuanced than vector cosine.
    rerank_response = co.rerank(
        query     = query,
        documents = [c["text"] for c in candidates],
        top_n     = top_k,
        model     = COHERE_RERANK_MODEL,
    )

    # rerank_response.results is sorted by relevance (best first).
    # Each result has an .index pointing back into our candidates list.
    reranked = []
    for r in rerank_response.results:
        chunk = candidates[r.index].copy()
        chunk["rerank_score"] = r.relevance_score   # keep for debugging
        reranked.append(chunk)

    return reranked
