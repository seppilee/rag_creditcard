import hashlib
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from langchain_ollama import OllamaEmbeddings

DB_PATH = "cardkb.db"
emb = OllamaEmbeddings(model="bge-m3")
ARTICLE_SPLIT = re.compile(r"(?=^\s*제\s?\d+\s?조[\s(（])", re.M)
ARTICLE_LABEL = re.compile(r"\s*(제\s?\d+\s?조[^\n]{0,30})")
_cache = {"ids": [], "mat": None}


def conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.executescript("""
    CREATE TABLE IF NOT EXISTS docs(
        id INTEGER PRIMARY KEY, path TEXT UNIQUE, hash TEXT, title TEXT);
    CREATE TABLE IF NOT EXISTS chunks(
        id INTEGER PRIMARY KEY, doc_id INTEGER, page INTEGER,
        article TEXT, text TEXT, embedding BLOB);
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
        USING fts5(text, tokenize='trigram');
    """)
    return c


def windows(t: str, size: int = 800, overlap: int = 100) -> list[str]:
    if len(t) <= size:
        return [t]
    return [t[i:i + size] for i in range(0, len(t), size - overlap)]


def ingest(path: str) -> int:
    p = Path(path)
    h = hashlib.sha1(p.read_bytes()).hexdigest()
    with conn() as c:
        row = c.execute("SELECT id, hash FROM docs WHERE path=?", (str(p),)).fetchone()
        if row and row["hash"] == h:
            return 0  # 변경 없음
        if row:
            doc_id = row["id"]
            ids = [r[0] for r in c.execute("SELECT id FROM chunks WHERE doc_id=?", (doc_id,))]
            c.executemany("DELETE FROM chunks_fts WHERE rowid=?", [(i,) for i in ids])
            c.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            c.execute("UPDATE docs SET hash=? WHERE id=?", (h, doc_id))
        else:
            doc_id = c.execute("INSERT INTO docs(path, hash, title) VALUES(?,?,?)",
                               (str(p), h, p.stem)).lastrowid

        recs = []  # (page, article, text)
        for pageno, page in enumerate(PdfReader(str(p)).pages, 1):
            text = (page.extract_text() or "").strip()
            for piece in ARTICLE_SPLIT.split(text):
                piece = piece.strip()
                if len(piece) < 30:
                    continue
                m = ARTICLE_LABEL.match(piece)
                label = m.group(1).strip() if m else ""
                for w in windows(piece):
                    recs.append((pageno, label, w))
        if not recs:
            raise SystemExit(f"{p.name}: 텍스트 추출 실패 (스캔본이면 OCR 필요)")

        vecs = []
        for i in range(0, len(recs), 64):
            batch = recs[i:i + 64]
            vecs += emb.embed_documents([f"{p.stem} {a}\n{t}" for _, a, t in batch])
        for (pg, art, t), v in zip(recs, vecs):
            cid = c.execute(
                "INSERT INTO chunks(doc_id, page, article, text, embedding) VALUES(?,?,?,?,?)",
                (doc_id, pg, art, t, np.asarray(v, dtype=np.float32).tobytes()),
            ).lastrowid
            c.execute("INSERT INTO chunks_fts(rowid, text) VALUES(?,?)", (cid, t))
    _load_cache()
    return len(recs)


def _load_cache() -> None:
    with conn() as c:
        rows = c.execute("SELECT id, embedding FROM chunks").fetchall()
    if not rows:
        _cache.update(ids=[], mat=None)
        return
    mat = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    _cache.update(ids=[r["id"] for r in rows], mat=mat)


def search(query: str, k: int = 5) -> list[dict]:
    if _cache["mat"] is None:
        _load_cache()
    if _cache["mat"] is None:
        return []

    # 벡터 순위
    q = np.asarray(emb.embed_query(query), dtype=np.float32)
    q /= np.linalg.norm(q)
    order = np.argsort(-(_cache["mat"] @ q))[:20]
    vec_ids = [_cache["ids"][i] for i in order]

    # BM25(FTS5 trigram) 순위 - 3글자 이상 단어만 매칭됨
    terms = [t for t in re.findall(r"\w+", query) if len(t) >= 3]
    fts_ids = []
    if terms:
        match = " OR ".join(f'"{t}"' for t in terms)
        with conn() as c:
            try:
                fts_ids = [r[0] for r in c.execute(
                    "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
                    "ORDER BY rank LIMIT 20", (match,))]
            except sqlite3.OperationalError:
                pass

    # RRF 결합
    score: dict[int, float] = {}
    for ranking in (vec_ids, fts_ids):
        for r, cid in enumerate(ranking):
            score[cid] = score.get(cid, 0.0) + 1.0 / (60 + r)
    top = sorted(score, key=score.get, reverse=True)[:k]

    out = []
    with conn() as c:
        for cid in top:
            r = c.execute("""SELECT c.id, c.page, c.article, c.text, d.title
                             FROM chunks c JOIN docs d ON d.id=c.doc_id
                             WHERE c.id=?""", (cid,)).fetchone()
            out.append({"id": r["id"], "doc": r["title"], "page": r["page"],
                        "article": r["article"], "text": r["text"],
                        "score": score[cid]})
    return out


if __name__ == "__main__":
    for f in sys.argv[1:]:
        print(f"{Path(f).name}: {ingest(f)} chunks")