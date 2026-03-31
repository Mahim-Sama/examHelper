# src/ingestion/chunker.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .loader import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(docs: list[Document]) -> list[Document]:
    """
    Split each Document into smaller overlapping chunks.

    RecursiveCharacterTextSplitter tries to split on natural
    boundaries in this order: paragraphs → sentences → words → chars.
    This means it won't cut a sentence in half if it can avoid it.

    The overlap means chunk N and chunk N+1 share CHUNK_OVERLAP
    characters — so a concept that spans a chunk boundary still
    appears complete in at least one chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size        = CHUNK_SIZE,
        chunk_overlap     = CHUNK_OVERLAP,
        separators        = ["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in docs:
        splits = splitter.split_text(doc.text)
        for i, split_text in enumerate(splits):
            chunks.append(Document(
                text     = split_text,
                metadata = {
                    **doc.metadata,       # preserve all original metadata
                    "chunk_index": i,     # position within original document
                }
            ))

    print(f"[green]{len(docs)} documents → {len(chunks)} chunks[/green]")
    return chunks