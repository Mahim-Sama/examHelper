# src/retrieval/retriever.py
import cohere
from pinecone import Pinecone
from config import (
    COHERE_API_KEY, COHERE_EMBED_MODEL,
    PINECONE_API_KEY, PINECONE_INDEX_NAME,
    TOP_K_RETRIEVAL
)


def retrieve(query: str, top_k: int = TOP_K_RETRIEVAL,
             filter_priority: str | None = None) -> list[dict]:
    """
    Embed the query and find the most semantically similar chunks.

    Note input_type="search_query" here — different from indexing!
    Cohere uses this to produce a query-optimised embedding that
    aligns better with document embeddings in the vector space.

    filter_priority: pass "exam_hint" to retrieve only from your
    priority notes — useful for last-minute cramming sessions.
    """
    co    = cohere.Client(api_key=COHERE_API_KEY)
    index = Pinecone(api_key=PINECONE_API_KEY).Index(PINECONE_INDEX_NAME)

    # Embed the query
    response  = co.embed(
        texts      = [query],
        model      = COHERE_EMBED_MODEL,
        input_type = "search_query",    # different from indexing!
    )
    query_vec = response.embeddings[0]

    # Build optional metadata filter
    pinecone_filter = {}
    if filter_priority:
        pinecone_filter["priority"] = {"$eq": filter_priority}

    # Search Pinecone
    results = index.query(
        vector          = query_vec,
        top_k           = top_k,
        include_metadata= True,
        filter          = pinecone_filter if pinecone_filter else None,
    )

    # Return clean list of matches with text and metadata
    return [
        {
            "text":     match["metadata"]["text"],
            "score":    match["score"],
            "source":   match["metadata"].get("source", "unknown"),
            "page":     match["metadata"].get("page", "?"),
            "priority": match["metadata"].get("priority", "normal"),
        }
        for match in results["matches"]
    ]