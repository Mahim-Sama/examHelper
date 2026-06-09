# main.py
from pathlib import Path
import subprocess
import tempfile
from src.ingestion.loader   import load_folder
from src.ingestion.chunker  import chunk_documents
from src.ingestion.embedder import embed_and_store
from src.retrieval.retriever import retrieve
from src.generation.generator import answer_question, generate_problems, recap_topic
from pinecone import Pinecone
from rich import print
from rich.prompt import Prompt
from rich.table import Table
import config


def status():
    """
    Show what's stored in Pinecone and prove exam_hint filtering works.

    Two checks:
    1. Index stats — total vector count straight from Pinecone's metadata API.
       No embeddings are computed; this is just a fast metadata call.
    2. Filtered sample retrieval — runs a real hybrid query restricted to
       exam_hint chunks. If it returns results, your priority tag is live
       and filterable. If it returns 0, your ingest used priority="normal".
    """
    pc    = Pinecone(api_key=config.PINECONE_API_KEY)
    index = pc.Index(config.PINECONE_HYBRID_INDEX_NAME)

    # ── 1. Index stats ────────────────────────────────────────────────────────
    stats       = index.describe_index_stats()
    total       = stats.total_vector_count
    dimension   = stats.dimension
    print(f"\n[bold]Index:[/bold] [cyan]{config.PINECONE_HYBRID_INDEX_NAME}[/cyan]")
    print(f"  Vectors stored : [green]{total}[/green]")
    print(f"  Dimension      : {dimension}  (1024 = Cohere embed-english-v3.0)")
    print(f"  Metric         : dotproduct  (required for hybrid search)\n")

    # ── 2. Exam-hint filter check ─────────────────────────────────────────────
    print("[bold]Exam-hint filter check[/bold] (retrieves top 5 exam_hint chunks):")
    sample = retrieve("probability distribution", top_k=5, filter_priority="exam_hint")

    if not sample:
        print("[red]  No exam_hint chunks found.[/red] Did you ingest with --priority exam_hint?")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="dim", width=5)
    table.add_column("Source", max_width=35)
    table.add_column("Page", width=5)
    table.add_column("Priority", width=10)
    table.add_column("Rerank score", width=12)
    table.add_column("Preview (first 80 chars)", max_width=50)

    for i, chunk in enumerate(sample, 1):
        table.add_row(
            str(i),
            chunk["source"],
            str(chunk["page"]),
            f"[yellow]{chunk['priority']}[/yellow]",
            f"{chunk.get('rerank_score', 0):.4f}",
            chunk["text"][:80].replace("\n", " "),
        )

    print(table)
    print(f"\n[green]✓ exam_hint filter is working — {len(sample)} chunks returned.[/green]")


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
    save_recap_pdf(topic, result)


def save_recap_pdf(topic: str, markdown_text: str):
    """
    Save the recap as a PDF using Pandoc + MiKTeX (xelatex engine).
    Pandoc understands LaTeX math blocks like $$...$$ natively, so
    formulas render correctly instead of appearing as raw symbols.

    The markdown is written to a temp file, passed to pandoc, then
    the temp file is deleted. The final PDF goes to data/recaps/.
    """
    out_dir = Path("data/recaps")
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in topic).strip()
    out_path  = out_dir / f"{safe_name}.pdf"

    # Prepend a YAML front matter block so pandoc sets the title cleanly
    content = f"---\ntitle: 'Exam Recap: {topic}'\ngeometry: margin=2.5cm\nfontsize: 12pt\n---\n\n{markdown_text}"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        xelatex = Path.home() / "AppData/Local/Programs/MiKTeX/miktex/bin/x64/xelatex.exe"
        subprocess.run(
            [
                "pandoc", str(tmp_path),
                "-o", str(out_path.resolve()),
                f"--pdf-engine={xelatex}",
                "--highlight-style=tango",   # syntax highlighting for code blocks
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"[bold cyan]Saved recap PDF:[/bold cyan] {out_path.resolve()}")
    except subprocess.CalledProcessError as e:
        print(f"[red]Pandoc failed:[/red] {e.stderr}")
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    print("[bold]Exam RAG System[/bold]")
    print("Commands: [ingest] [ask] [problems] [recap] [status]\n")

    cmd = Prompt.ask("Command", choices=["ingest", "ask", "problems", "recap", "status"])

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

    elif cmd == "status":
        status()
