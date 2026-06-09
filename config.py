# config.py
from dotenv import load_dotenv
import os

load_dotenv()

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
COHERE_API_KEY    = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY  = os.getenv("PINECONE_API_KEY")

# Pinecone settings
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "exam-rag")
PINECONE_CLOUD      = "aws"
PINECONE_REGION     = "us-east-1"

# Cohere settings
# embed-english-v3.0 produces 1024-dimensional vectors - the dimension
# you'll set when creating your Pinecone index. Don't change this
# after you've already indexed documents or you'll get dimension errors.
COHERE_EMBED_MODEL  = "embed-english-v3.0"
EMBEDDING_DIMENSION = 1024

# Chunking settings - we'll explain these in Step 2
CHUNK_SIZE          = 512   # characters per chunk
CHUNK_OVERLAP       = 64    # overlap between chunks to preserve context

# Claude model
CLAUDE_MODEL        = "claude-sonnet-4-6"

# Retrieval settings
TOP_K_RETRIEVAL     = 8     # final number of chunks passed to the LLM after reranking

# --- Phase 2: Advanced Retrieval --------------------------------------------
# Hybrid search needs dotproduct metric, which is incompatible with the cosine
# index from Phase 1. We create a new index rather than mutating the old one.
PINECONE_HYBRID_INDEX_NAME = os.getenv("PINECONE_HYBRID_INDEX_NAME", "exam-rag-hybrid")

# Alpha controls the dense/sparse balance in hybrid search.
#   alpha=1.0  →  pure dense (semantic, like Phase 1)
#   alpha=0.0  →  pure sparse (BM25 keyword matching)
#   alpha=0.75 →  semantic-leaning but keyword-aware (good default)
HYBRID_ALPHA        = 0.75

# Fetch more candidates from Pinecone than you need, then let the reranker
# pick the best TOP_K_RETRIEVAL from among them. More headroom = better rerank.
TOP_K_FETCH         = 20

# Cohere cross-encoder reranker. Unlike the bi-encoder used for embedding,
# this model reads the query and each chunk *together* to score relevance.
# Much more accurate but too slow to run on the whole index — hence the
# fetch-then-rerank pattern.
COHERE_RERANK_MODEL = "rerank-english-v3.0"