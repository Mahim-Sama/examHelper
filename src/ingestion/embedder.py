# src/ingestion/embedder.py
import time
import cohere
from cohere.errors import TooManyRequestsError
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from .loader import Document
from config import (
    COHERE_API_KEY, COHERE_EMBED_MODEL,
    PINECONE_API_KEY, PINECONE_HYBRID_INDEX_NAME,
    PINECONE_CLOUD, PINECONE_REGION, EMBEDDING_DIMENSION
)
from rich import print
from rich.progress import track


def get_pinecone_index():
    """
    Create the hybrid Pinecone index if it doesn't exist, then return it.

    KEY DIFFERENCE from Phase 1: metric is now "dotproduct" instead of
    "cosine". Hybrid search (dense + sparse) requires dotproduct because
    Pinecone combines the two scores additively at query time. With cosine,
    the normalization step would break that addition.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if PINECONE_HYBRID_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating hybrid Pinecone index [cyan]{PINECONE_HYBRID_INDEX_NAME}[/cyan]...")
        pc.create_index(
            name      = PINECONE_HYBRID_INDEX_NAME,
            dimension = EMBEDDING_DIMENSION,
            metric    = "dotproduct",       # required for hybrid search
            spec      = ServerlessSpec(
                cloud  = PINECONE_CLOUD,
                region = PINECONE_REGION
            )
        )
        print("[green]Index created.[/green]")
    else:
        print(f"[yellow]Index '{PINECONE_HYBRID_INDEX_NAME}' already exists.[/yellow]")

    return pc.Index(PINECONE_HYBRID_INDEX_NAME)


def embed_and_store(chunks: list[Document], batch_size: int = 90) -> None:
    """
    Embed each chunk with Cohere (dense) + BM25 (sparse) and upsert to Pinecone.

    Why two representations?
    - Dense (Cohere): captures semantic meaning. "eigenvalue decomposition" and
      "matrix factorization" will be nearby even if they share no words.
    - Sparse (BM25): captures exact keyword matches. If a student asks about
      "BPTT" and the notes say "BPTT", sparse will catch it even if the dense
      embeddings are far apart due to abbreviation inconsistency.

    BM25Encoder.default() is pre-trained on MS-MARCO (Microsoft's large-scale
    QA dataset). It provides reasonable IDF weights without needing to fit on
    your specific corpus. Good enough for academic notes.
    """
    co    = cohere.Client(api_key=COHERE_API_KEY)
    index = get_pinecone_index()
    bm25  = BM25Encoder.default()

    # Track how many tokens have been sent in the current 60s window.
    TOKEN_LIMIT   = 90_000
    window_start  = time.monotonic()
    window_tokens = 0

    for batch_start in track(
        range(0, len(chunks), batch_size),
        description="Embedding and storing..."
    ):
        batch  = chunks[batch_start : batch_start + batch_size]
        texts  = [doc.text for doc in batch]

        # -- Sparse vectors (BM25) --------------------------------------------
        # encode_documents() returns a list of {"indices": [...], "values": [...]}
        # Each index maps to a token in BM25's vocabulary; the value is the
        # TF-IDF-style weight for that token in this specific document.
        # This is a sparse operation — most values are 0 and are omitted.
        sparse_vecs = bm25.encode_documents(texts)

        # -- Dense vectors (Cohere) --------------------------------------------
        batch_tokens = sum(len(t) // 4 for t in texts)

        elapsed = time.monotonic() - window_start
        if window_tokens + batch_tokens > TOKEN_LIMIT:
            sleep_for = max(0.0, 60.0 - elapsed)
            if sleep_for > 0:
                print(f"[cyan]Approaching token limit - pausing {sleep_for:.1f}s to reset window...[/cyan]")
                time.sleep(sleep_for)
            window_start  = time.monotonic()
            window_tokens = 0
        elif elapsed >= 60.0:
            window_start  = time.monotonic()
            window_tokens = 0

        for attempt in range(5):
            try:
                response = co.embed(
                    texts      = texts,
                    model      = COHERE_EMBED_MODEL,
                    input_type = "search_document",
                )
                break
            except TooManyRequestsError:
                wait = 60 * (attempt + 1)
                print(f"[yellow]Rate limit hit - waiting {wait}s before retry {attempt + 1}/5...[/yellow]")
                time.sleep(wait)
                window_start  = time.monotonic()
                window_tokens = 0
        else:
            raise RuntimeError("Cohere rate limit: all 5 retries exhausted.")

        window_tokens += batch_tokens
        dense_vecs = response.embeddings

        # -- Build and upsert vectors --------------------------------------------
        vectors = []
        for doc, dense, sparse in zip(batch, dense_vecs, sparse_vecs):
            vector_id = f"{doc.metadata['source']}_p{doc.metadata['page']}_c{doc.metadata['chunk_index']}"
            vectors.append({
                "id":           vector_id,
                "values":       dense,          # 1024-d dense embedding
                "sparse_values": sparse,        # BM25 sparse encoding
                "metadata": {
                    **doc.metadata,
                    "text": doc.text,
                }
            })

        index.upsert(vectors=vectors)

    print(f"[green]Successfully stored {len(chunks)} chunks in Pinecone (hybrid index).[/green]")
