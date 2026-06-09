# ExamHelper

A Retrieval-Augmented Generation (RAG) system for exam preparation. Load your course materials, then ask questions, generate practice problems, or create summarized recap PDFs — all grounded in your actual study content.

## How It Works

**Three-stage pipeline:**

1. **Ingest** — Load PDFs, text files, and handwritten note images (OCR). Chunks are embedded with Cohere (dense) and BM25 (sparse), then stored in Pinecone.
2. **Retrieve** — Hybrid search combines semantic similarity and keyword matching, followed by Cohere cross-encoder reranking.
3. **Generate** — Claude uses the retrieved context to answer questions, write practice problems, or produce a recap.

## Prerequisites

- Python 3.11+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (for handwritten image ingestion)
- [Pandoc](https://pandoc.org/) + [MiKTeX](https://miktex.org/) (for PDF recap generation)
- API keys: **Anthropic**, **Cohere**, **Pinecone**

## Setup

```powershell
# 1. Activate the virtual environment
.\.examragenv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create a .env file with your API keys (see Configuration below)
```

## Configuration

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=sk-ant-...
COHERE_API_KEY=...
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=exam-rag
PINECONE_HYBRID_INDEX_NAME=exam-rag-hybrid
```

Key settings in `config.py` (edit as needed):

| Setting | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K_FETCH` | `20` | Candidates fetched before reranking |
| `TOP_K_RETRIEVAL` | `8` | Final chunks returned after reranking |
| `HYBRID_ALPHA` | `0.75` | Semantic vs. keyword weight (1.0 = fully semantic) |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Claude model used for generation |

## Usage

```powershell
python main.py
```

This opens an interactive CLI with five commands:

### `ingest`
Load documents into the vector database. Accepts any folder path.

- Place regular course materials in `data/raw/`
- Place high-priority exam hints in `data/exam_hints/`
- Supported formats: `.pdf`, `.txt`, `.md`, `.jpg`, `.png`, `.jpeg` (images use OCR)

```
Command: ingest
Folder path: data/raw
Priority (normal/exam_hint): normal
```

### `ask`
Ask a question grounded in your ingested course materials.

```
Command: ask
Query: What is the difference between PDF and PMF?
```

### `problems`
Generate practice problems with detailed solutions for a given topic.

```
Command: problems
Topic: Probability Density Functions
Number of problems: 3
```

### `recap`
Generate a dense exam-prep summary and save it as a formatted PDF to `data/recaps/`.

```
Command: recap
Topic: Probability Mass Function
```

### `status`
Check Pinecone index stats and verify that exam-hint filtering is working correctly.

## Project Structure

```
examHelper/
├── main.py                  # CLI entry point
├── config.py                # All configurable settings
├── requirements.txt
├── .env                     # API keys (not committed)
├── src/
│   ├── ingestion/
│   │   ├── loader.py        # Load PDFs, text, images (OCR)
│   │   ├── chunker.py       # Split documents into overlapping chunks
│   │   └── embedder.py      # Embed with Cohere + store in Pinecone
│   ├── retrieval/
│   │   └── retriever.py     # Hybrid search + cross-encoder reranking
│   └── generation/
│       ├── generator.py     # Ask / problems / recap logic
│       └── prompts.py       # Claude system prompts
└── data/
    ├── raw/                 # Input: course documents
    ├── exam_hints/          # Input: high-priority exam material
    └── recaps/              # Output: generated recap PDFs
```

## Rate Limiting

- **Cohere**: Automatically pauses ingestion when approaching the 90K tokens/60s limit.
- **Anthropic**: Retries on overload (529) with exponential backoff, up to 5 attempts.

## Dependencies

| Package | Purpose |
|---|---|
| `anthropic` / `langchain-anthropic` | Claude LLM generation |
| `cohere` / `langchain-cohere` | Embeddings and reranking |
| `pinecone` / `pinecone-text` | Vector storage and hybrid search |
| `pymupdf` | PDF text extraction |
| `pytesseract` + `Pillow` | Handwritten note OCR |
| `rich` | CLI formatting |
| `python-dotenv` | `.env` loading |
