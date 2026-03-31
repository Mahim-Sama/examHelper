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
    Create the Pinecone index if it does not exist, then return it.
    The dimension must match the embedding model output size (1024
    for embed-english-v3.0). If you swap models, delete and recreate
    the index or the dimension mismatch will cause an error.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating Pinecone index [cyan]{PINECONE_INDEX_NAME}[/cyan]...")
        pc.create_index(
            name      = PINECONE_INDEX_NAME,
            dimension = EMBEDDING_DIMENSION,
            metric    = "cosine",
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
    Embed each chunk with Cohere and store the vectors in Pinecone.

    Batching is needed because Cohere's API accepts at most 96 texts
    per call and Pinecone upsert performs best around 100 vectors.

    input_type="search_document" tells Cohere these are passages being
    indexed, not a search query. Always use the right type or retrieval
    quality drops.
    """
    co    = cohere.Client(api_key=COHERE_API_KEY)
    index = get_pinecone_index()

    # Track how many tokens have been sent in the current 60s window.
    # The Cohere trial plan allows 100k tokens per minute, so we stay
    # under 90k to leave a small buffer before hitting the hard limit.
    TOKEN_LIMIT   = 90_000
    window_start  = time.monotonic()
    window_tokens = 0

    for batch_start in track(
        range(0, len(chunks), batch_size),
        description="Embedding and storing..."
    ):
        batch  = chunks[batch_start : batch_start + batch_size]
        texts  = [doc.text for doc in batch]

        # Rough token estimate: 1 token is about 4 characters on average
        batch_tokens = sum(len(t) // 4 for t in texts)

        elapsed = time.monotonic() - window_start
        if window_tokens + batch_tokens > TOKEN_LIMIT:
            # Sending this batch would go over the limit, so wait out
            # the rest of the current window before continuing
            sleep_for = max(0.0, 60.0 - elapsed)
            if sleep_for > 0:
                print(f"[cyan]Approaching token limit - pausing {sleep_for:.1f}s to reset window...[/cyan]")
                time.sleep(sleep_for)
            window_start  = time.monotonic()
            window_tokens = 0
        elif elapsed >= 60.0:
            # The window expired on its own, just reset the counters
            window_start  = time.monotonic()
            window_tokens = 0

        # Retry up to 5 times if the API still rejects the request.
        # This should rarely trigger since the window tracking above
        # keeps us below the limit proactively.
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
        embeddings = response.embeddings

        vectors = []
        for i, (doc, emb) in enumerate(zip(batch, embeddings)):
            vector_id = f"{doc.metadata['source']}_p{doc.metadata['page']}_c{doc.metadata['chunk_index']}"
            vectors.append({
                "id":       vector_id,
                "values":   emb,
                "metadata": {
                    **doc.metadata,
                    "text": doc.text,   # stored so retrieval can return the raw text
                }
            })

        index.upsert(vectors=vectors)

    print(f"[green]Successfully stored {len(chunks)} chunks in Pinecone.[/green]")