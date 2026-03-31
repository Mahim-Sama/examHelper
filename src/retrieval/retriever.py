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
    Embed the query and return the closest matching chunks from Pinecone.

    input_type="search_query" is different from what we use during
    indexing. Cohere produces a slightly different embedding for queries
    so they align better with the stored document embeddings.

    Pass filter_priority="exam_hint" to search only in your priority
    notes rather than the full textbook.
    """
    co    = cohere.Client(api_key=COHERE_API_KEY)
    index = Pinecone(api_key=PINECONE_API_KEY).Index(PINECONE_INDEX_NAME)

    response  = co.embed(
        texts      = [query],
        model      = COHERE_EMBED_MODEL,
        input_type = "search_query",
    )
    query_vec = response.embeddings[0]

    pinecone_filter = {}
    if filter_priority:
        pinecone_filter["priority"] = {"$eq": filter_priority}

    results = index.query(
        vector           = query_vec,
        top_k            = top_k,
        include_metadata = True,
        filter           = pinecone_filter if pinecone_filter else None,
    )

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