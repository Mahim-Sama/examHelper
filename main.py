# main.py
from pathlib import Path
import subprocess
import tempfile
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
