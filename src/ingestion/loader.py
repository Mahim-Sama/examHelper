# src/ingestion/loader.py
from pathlib import Path
from dataclasses import dataclass, field
import fitz
from PIL import Image
import pytesseract
from rich import print

@dataclass
class Document:
    """
    One page or file worth of text, plus where it came from.
    Metadata is attached here so it stays with the content all
    the way through chunking and into Pinecone.
    """
    text: str
    metadata: dict = field(default_factory=dict)


def load_pdf(path: Path, priority: str = "normal") -> list[Document]:
    """
    Pull text out of every page of a PDF.
    Each page becomes its own Document - chunking comes later.

    Pages are loaded one at a time so a large textbook does not
    load the entire file into memory at once. Page numbers are
    also captured this way for free.
    """
    docs = []
    pdf   = fitz.open(str(path))

    for page_num, page in enumerate(pdf, start=1):
        text = page.get_text("text").strip()
        if not text:          # skip empty or image-only pages
            continue
        docs.append(Document(
            text=text,
            metadata={
                "source":    path.name,
                "page":      page_num,
                "type":      "pdf",
                "priority":  priority,   # "normal" or "exam_hint"
            }
        ))

    print(f"[green]Loaded {len(docs)} pages from {path.name}[/green]")
    return docs


def load_text(path: Path, priority: str = "normal") -> list[Document]:
    """Load a plain .txt or .md file as a single Document."""
    text = path.read_text(encoding="utf-8").strip()
    return [Document(
        text=text,
        metadata={
            "source":   path.name,
            "page":     1,
            "type":     "text",
            "priority": priority,
        }
    )]


def load_image_ocr(path: Path, priority: str = "normal") -> list[Document]:
    """
    Run OCR on a handwritten note image and return the text.
    pytesseract is a wrapper around Tesseract. To install Tesseract
    on Windows grab it from https://github.com/UB-Mannheim/tesseract/wiki
    and add its folder to your system PATH.
    """
    img  = Image.open(str(path))
    text = pytesseract.image_to_string(img).strip()

    if not text:
        print(f"[yellow]Warning: OCR returned no text for {path.name}[/yellow]")
        return []

    return [Document(
        text=text,
        metadata={
            "source":   path.name,
            "page":     1,
            "type":     "handwritten",
            "priority": priority,
        }
    )]


def load_folder(folder: Path, priority: str = "normal") -> list[Document]:
    """
    Load all supported files from a folder, including subfolders.
    Pass priority="exam_hint" when loading your exam hints folder.
    """
    loaders = {
        ".pdf": load_pdf,
        ".txt": load_text,
        ".md":  load_text,
        ".png": load_image_ocr,
        ".jpg": load_image_ocr,
        ".jpeg":load_image_ocr,
    }
    docs = []
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() in loaders:
            print(f"Loading [cyan]{path.name}[/cyan]...")
            docs.extend(loaders[path.suffix.lower()](path, priority))
    return docs