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
# embed-english-v3.0 produces 1024-dimensional vectors — the dimension
# you'll set when creating your Pinecone index. Don't change this
# after you've already indexed documents or you'll get dimension errors.
COHERE_EMBED_MODEL  = "embed-english-v3.0"
EMBEDDING_DIMENSION = 1024

# Chunking settings — we'll explain these in Step 2
CHUNK_SIZE          = 512   # characters per chunk
CHUNK_OVERLAP       = 64    # overlap between chunks to preserve context

# Claude model
CLAUDE_MODEL        = "claude-sonnet-4-6"

# Retrieval settings
TOP_K_RETRIEVAL     = 8     # how many chunks to fetch from Pinecone