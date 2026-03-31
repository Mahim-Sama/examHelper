# src/ingestion/chunker.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .loader import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(docs: list[Document]) -> list[Document]:
    """
    Split each Document into smaller overlapping pieces.

    RecursiveCharacterTextSplitter tries these split points in order:
    paragraph breaks, line breaks, sentence endings, spaces, then
    individual characters. It avoids cutting mid-sentence when possible.

    The overlap means consecutive chunks share some text so a concept
    that sits near a split boundary still appears fully in one chunk.
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
                    **doc.metadata,       # keep all original metadata
                    "chunk_index": i,     # position within the source page
                }
            ))

    print(f"[green]{len(docs)} documents -> {len(chunks)} chunks[/green]")
    return chunks