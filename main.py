# main.py
from pathlib import Path
from src.ingestion.loader   import load_folder
from src.ingestion.chunker  import chunk_documents
from src.ingestion.embedder import embed_and_store
from src.retrieval.retriever import retrieve
from src.generation.generator import answer_question, generate_problems, recap_topic
from rich import print
from rich.prompt import Prompt


def ingest(folder: str = "data/raw", priority: str = "normal"):
    """Load, chunk, embed, and store all documents in a folder."""
    docs   = load_folder(Path(folder), priority=priority)
    chunks = chunk_documents(docs)
    embed_and_store(chunks)


def ask(query: str):
    print(f"\n[bold cyan]Query:[/bold cyan] {query}")
    chunks = retrieve(query)
    print(f"[dim]Retrieved {len(chunks)} chunks from Pinecone[/dim]")
    answer = answer_question(query, chunks)
    print(f"\n[bold green]Answer:[/bold green]\n{answer}")


def problems(topic: str, n: int = 3):
    chunks = retrieve(topic)
    result = generate_problems(chunks, n=n)
    print(f"\n[bold green]Practice Problems:[/bold green]\n{result}")


def recap(topic: str):
    chunks = retrieve(topic)
    result = recap_topic(topic, chunks)
    print(f"\n[bold green]Exam Recap:[/bold green]\n{result}")


if __name__ == "__main__":
    print("[bold]Exam RAG System[/bold]")
    print("Commands: [ingest] [ask] [problems] [recap]\n")

    cmd = Prompt.ask("Command", choices=["ingest", "ask", "problems", "recap"])

    if cmd == "ingest":
        folder   = Prompt.ask("Folder", default="data/raw")
        priority = Prompt.ask("Priority", choices=["normal", "exam_hint"], default="normal")
        ingest(folder, priority)

    elif cmd == "ask":
        query = Prompt.ask("Your question")
        ask(query)

    elif cmd == "problems":
        topic = Prompt.ask("Topic")
        n     = int(Prompt.ask("How many problems", default="3"))
        problems(topic, n)

    elif cmd == "recap":
        topic = Prompt.ask("Topic")
        recap(topic)