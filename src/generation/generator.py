# src/generation/generator.py
import time
import anthropic
from anthropic._exceptions import OverloadedError
from .prompts import QA_SYSTEM_PROMPT, PROBLEM_GENERATOR_PROMPT, TOPIC_RECAP_PROMPT
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL


client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _call_claude(system: str, user_content: str, max_tokens: int) -> str:
    """
    Send a request to Claude and return the response text.
    Retries up to 5 times if the API returns a 529 overloaded error,
    waiting a bit longer between each attempt.
    """
    for attempt in range(5):
        try:
            message = client.messages.create(
                model      = CLAUDE_MODEL,
                max_tokens = max_tokens,
                system     = system,
                messages   = [{"role": "user", "content": user_content}]
            )
            return message.content[0].text
        except OverloadedError:
            wait = 10 * (attempt + 1)
            print(f"[yellow]Anthropic overloaded - waiting {wait}s before retry {attempt + 1}/5...[/yellow]")
            time.sleep(wait)
    raise RuntimeError("Anthropic API overloaded: all 5 retries exhausted.")


def _build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered block with source info."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Source {i}: {chunk['source']}, page {chunk['page']}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def answer_question(query: str, chunks: list[dict]) -> str:
    context = _build_context(chunks)
    return _call_claude(
        system       = QA_SYSTEM_PROMPT,
        user_content = f"Context from course material:\n\n{context}\n\nQuestion: {query}",
        max_tokens   = 1024,
    )


def generate_problems(chunks: list[dict], n: int = 3) -> str:
    context = _build_context(chunks)
    return _call_claude(
        system       = PROBLEM_GENERATOR_PROMPT.format(n=n),
        user_content = f"Course material:\n\n{context}\n\nGenerate {n} practice problems.",
        max_tokens   = 2048,
    )


def recap_topic(topic: str, chunks: list[dict]) -> str:
    context = _build_context(chunks)
    return _call_claude(
        system       = TOPIC_RECAP_PROMPT,
        user_content = f"Course material on '{topic}':\n\n{context}\n\nGive me an exam recap.",
        max_tokens   = 1024,
    )