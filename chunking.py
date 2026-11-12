import re

CHUNK_SIZE = 600
HEAD = re.compile(r"^(#{1,4}\s+.+|제\s?\d+\s?조[^\n]{0,40})")


def windows(text: str, size: int, overlap: int | None = None) -> list[str]:
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap is None:
        overlap = min(80, size // 5)
    if not 0 <= overlap < size:
        raise ValueError("overlap must be non-negative and smaller than size")
    if len(text) <= size:
        return [text]

    step = size - overlap
    return [text[start:start + size] for start in range(0, len(text), step)]


def chunk_page(
    text: str, size: int = CHUNK_SIZE, heading: str = ""
) -> tuple[list[tuple[str, str]], str]:
    """문단 단위로 합치되 제목 경계를 존중하고 표는 통째로 보존.
    (chunks, 마지막 heading) 반환. chunks는 (heading, text) 리스트."""
    if size <= 0:
        raise ValueError("size must be positive")

    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    out: list[tuple[str, str]] = []
    buf = ""
    cur = heading

    def flush() -> None:
        nonlocal buf
        if buf.strip():
            out.append((cur, buf.strip()))
        buf = ""

    for b in blocks:
        m = HEAD.match(b)
        if m:
            flush()
            cur = m.group(1).lstrip("# ").strip()
        if b.lstrip().startswith("|"):           # 마크다운 표
            flush()
            out.append((cur, b))
            continue
        if len(b) > size:                         # 너무 긴 문단
            flush()
            out.extend((cur, part) for part in windows(b, size))
            continue
        if len(buf) + len(b) > size:
            flush()
        buf += ("\n\n" if buf else "") + b
    flush()
    return out, cur
