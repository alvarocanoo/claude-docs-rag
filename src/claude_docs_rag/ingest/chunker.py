"""Header-aware markdown chunker.

Strategy:
1. Split the page on H2/H3/H4 headers — each header starts a new section.
2. If a section is larger than `target_tokens`, split on blank-line paragraphs.
3. If consecutive sections are below `min_tokens`, merge them.
4. Within each chunk, prepend the cumulative header path as breadcrumb context
   (so the embedding sees "Build with Claude > Prompt caching > ..." not just the body).

Token counting uses tiktoken (cl100k_base) as a proxy — it's not Anthropic's
exact tokenizer, but it's deterministic, offline, and close enough for chunking
decisions. The embedding model's actual tokenizer handles its own limit.
"""

from __future__ import annotations

import re

import tiktoken

from claude_docs_rag.models import Chunk, RawPage

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

DEFAULT_TARGET_TOKENS = 600
DEFAULT_MIN_TOKENS = 120
DEFAULT_OVERLAP_TOKENS = 80


def _encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str, enc: tiktoken.Encoding | None = None) -> int:
    enc = enc or _encoder()
    return len(enc.encode(text, disallowed_special=()))


def _split_sections(markdown: str) -> list[tuple[list[str], str]]:
    """Return [(header_path, body)] where header_path is the cumulative chain."""
    lines = markdown.splitlines()
    sections: list[tuple[list[str], str]] = []

    header_stack: list[tuple[int, str]] = []  # (level, text)
    body_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(body_lines).strip()
        if body:
            sections.append(([h[1] for h in header_stack], body))

    for line in lines:
        m = _HEADER_RE.match(line)
        if m:
            flush()
            body_lines = []
            level = len(m.group(1))
            text = m.group(2).strip()
            # Pop stack to the parent of this level.
            while header_stack and header_stack[-1][0] >= level:
                header_stack.pop()
            header_stack.append((level, text))
        else:
            body_lines.append(line)

    flush()
    return sections


def _slice_with_overlap(
    text: str,
    target_tokens: int,
    overlap_tokens: int,
    enc: tiktoken.Encoding,
) -> list[str]:
    """Token-window slicing with overlap. Used only when a section is huge."""
    tokens = enc.encode(text, disallowed_special=())
    if len(tokens) <= target_tokens:
        return [text]

    step = max(target_tokens - overlap_tokens, 1)
    pieces: list[str] = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + target_tokens]
        if not window:
            break
        pieces.append(enc.decode(window))
        if start + target_tokens >= len(tokens):
            break
    return pieces


def chunk_page(
    page: RawPage,
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    if not page.content.strip():
        return []

    enc = _encoder()
    raw_sections = _split_sections(page.content)

    if not raw_sections:
        # Page has no headers — treat whole content as one section.
        raw_sections = [([page.title], page.content.strip())]

    # Pre-split oversized sections into windows.
    expanded: list[tuple[list[str], str]] = []
    for header_path, body in raw_sections:
        if _count_tokens(body, enc) <= target_tokens:
            expanded.append((header_path, body))
        else:
            for piece in _slice_with_overlap(body, target_tokens, overlap_tokens, enc):
                expanded.append((header_path, piece))

    # Merge consecutive tiny sections that share a parent header.
    merged: list[tuple[list[str], str]] = []
    buf_path: list[str] | None = None
    buf_body: list[str] = []
    buf_tokens = 0

    def flush_buffer() -> None:
        if buf_path is not None and buf_body:
            merged.append((buf_path, "\n\n".join(buf_body)))

    for header_path, body in expanded:
        body_tokens = _count_tokens(body, enc)
        if (
            buf_path is not None
            and body_tokens < min_tokens
            and buf_tokens + body_tokens <= target_tokens
            and (not buf_path or not header_path or header_path[:1] == buf_path[:1])
        ):
            buf_body.append(body)
            buf_tokens += body_tokens
            buf_path = header_path  # adopt the latest leaf for breadcrumb
        else:
            flush_buffer()
            buf_path = header_path
            buf_body = [body]
            buf_tokens = body_tokens

    flush_buffer()

    chunks: list[Chunk] = []
    for idx, (header_path, body) in enumerate(merged):
        breadcrumb = " > ".join(header_path) if header_path else page.title
        full_text = f"{breadcrumb}\n\n{body}".strip()
        chunks.append(
            Chunk(
                source_url=page.url,
                title=page.title,
                section_path=breadcrumb,
                chunk_index=idx,
                content=full_text,
                content_tokens=_count_tokens(full_text, enc),
            )
        )

    return chunks
