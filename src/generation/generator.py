# src/generation/generator.py
import anthropic
from .prompts import QA_SYSTEM_PROMPT, PROBLEM_GENERATOR_PROMPT, TOPIC_RECAP_PROMPT
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL


client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _build_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a readable context block.
    We include source and page so Claude can cite them.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Source {i}: {chunk['source']}, page {chunk['page']}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def answer_question(query: str, chunks: list[dict]) -> str:
    context = _build_context(chunks)
    message = client.messages.create(
        model      = CLAUDE_MODEL,
        max_tokens = 1024,
        system     = QA_SYSTEM_PROMPT,
        messages   = [{
            "role":    "user",
            "content": f"Context from course material:\n\n{context}\n\nQuestion: {query}"
        }]
    )
    return message.content[0].text


def generate_problems(chunks: list[dict], n: int = 3) -> str:
    context = _build_context(chunks)
    system  = PROBLEM_GENERATOR_PROMPT.format(n=n)
    message = client.messages.create(
        model      = CLAUDE_MODEL,
        max_tokens = 2048,
        system     = system,
        messages   = [{
            "role":    "user",
            "content": f"Course material:\n\n{context}\n\nGenerate {n} practice problems."
        }]
    )
    return message.content[0].text


def recap_topic(topic: str, chunks: list[dict]) -> str:
    context = _build_context(chunks)
    message = client.messages.create(
        model      = CLAUDE_MODEL,
        max_tokens = 1024,
        system     = TOPIC_RECAP_PROMPT,
        messages   = [{
            "role":    "user",
            "content": f"Course material on '{topic}':\n\n{context}\n\nGive me an exam recap."
        }]
    )
    return message.content[0].text