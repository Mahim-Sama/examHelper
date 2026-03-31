# src/ingestion/embedder.py
import time
import cohere
from cohere.errors import TooManyRequestsError
from pinecone import Pinecone, ServerlessSpec
from .loader import Document
from config import (
    COHERE_API_KEY, COHERE_EMBED_MODEL,
    PINECONE_API_KEY, PINECONE_INDEX_NAME,
    PINECONE_CLOUD, PINECONE_REGION, EMBEDDING_DIMENSION
)
from rich import print
from rich.progress import track


def get_pinecone_index():
    """
    Create the Pinecone index if it doesn't exist, then return it.
    The dimension MUST match the embedding model's output size (1024
    for Cohere embed-english-v3.0). If you ever change models, you
    need to delete and recreate the index.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating Pinecone index [cyan]{PINECONE_INDEX_NAME}[/cyan]...")
        pc.create_index(
            name      = PINECONE_INDEX_NAME,
            dimension = EMBEDDING_DIMENSION,
            metric    = "cosine",          # cosine similarity for semantic search
            spec      = ServerlessSpec(
                cloud  = PINECONE_CLOUD,
                region = PINECONE_REGION
            )
        )
        print("[green]Index created.[/green]")
    else:
        print(f"[yellow]Index '{PINECONE_INDEX_NAME}' already exists.[/yellow]")

    return pc.Index(PINECONE_INDEX_NAME)


def embed_and_store(chunks: list[Document], batch_size: int = 90) -> None:
    """
    Embed chunks with Cohere and upsert them into Pinecone.

    We process in batches because:
    1. Cohere's API has a max batch size of 96 texts
    2. Pinecone upsert works best with batches of ~100 vectors

    input_type="search_document" tells Cohere these are documents
    being indexed (not queries). Cohere embed-v3 uses this to
    optimise the embedding for retrieval - always set this correctly.
    """
    co    = cohere.Client(api_key=COHERE_API_KEY)
    index = get_pinecone_index()

    # Proactive rate-limiter state for Cohere trial plan (100k tokens/min)
    TOKEN_LIMIT   = 90_000   # stay 10% below the hard 100k limit
    window_start  = time.monotonic()
    window_tokens = 0

    for batch_start in track(
        range(0, len(chunks), batch_size),
        description="Embedding and storing..."
    ):
        batch  = chunks[batch_start : batch_start + batch_size]
        texts  = [doc.text for doc in batch]

        # Estimate token cost (1 token ≈ 4 chars — good enough to pace requests)
        batch_tokens = sum(len(t) // 4 for t in texts)

        # If this batch would exceed the window budget, sleep until the window resets
        elapsed = time.monotonic() - window_start
        if window_tokens + batch_tokens > TOKEN_LIMIT:
            sleep_for = max(0.0, 60.0 - elapsed)
            if sleep_for > 0:
                print(f"[cyan]Approaching token limit — pausing {sleep_for:.1f}s to reset window...[/cyan]")
                time.sleep(sleep_for)
            window_start  = time.monotonic()
            window_tokens = 0
        elif elapsed >= 60.0:
            # Window already expired naturally — reset counters
            window_start  = time.monotonic()
            window_tokens = 0

        # Retry loop kept as a safety net in case the estimate is off
        for attempt in range(5):
            try:
                response = co.embed(
                    texts      = texts,
                    model      = COHERE_EMBED_MODEL,
                    input_type = "search_document",  # critical — not "search_query"
                )
                break
            except TooManyRequestsError:
                wait = 60 * (attempt + 1)
                print(f"[yellow]Rate limit hit — waiting {wait}s before retry {attempt + 1}/5...[/yellow]")
                time.sleep(wait)
                window_start  = time.monotonic()
                window_tokens = 0
        else:
            raise RuntimeError("Cohere rate limit: all 5 retries exhausted.")

        window_tokens += batch_tokens
        embeddings = response.embeddings

        # Build Pinecone vectors: (id, embedding, metadata)
        vectors = []
        for i, (doc, emb) in enumerate(zip(batch, embeddings)):
            vector_id = f"{doc.metadata['source']}_p{doc.metadata['page']}_c{doc.metadata['chunk_index']}"
            vectors.append({
                "id":       vector_id,
                "values":   emb,
                "metadata": {
                    **doc.metadata,
                    "text": doc.text,   # store text in metadata for retrieval
                }
            })

        index.upsert(vectors=vectors)

    print(f"[green]Successfully stored {len(chunks)} chunks in Pinecone.[/green]")