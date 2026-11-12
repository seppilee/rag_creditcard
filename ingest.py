import hashlib
import sys
from pathlib import Path

import pymupdf4llm
from elasticsearch import helpers


from common import es, emb, INDEX
from chunking import chunk_page



def ingest(path: str, category: str = "") -> int:
    p = Path(path)
    h = hashlib.sha1(p.read_bytes()).hexdigest()

    same = es.count(index=INDEX, query={"bool": {"filter": [
        {"term": {"doc": p.stem}}, {"term": {"doc_hash": h}}]}})["count"]
    if same:
        return 0                                         # 변경 없음
    es.delete_by_query(index=INDEX, query={"term": {"doc": p.stem}},
                       refresh=True, conflicts="proceed")  # 개정본이면 기존 청크 제거

    recs, heading = [], ""
    for pageno, pg in enumerate(pymupdf4llm.to_markdown(str(p), page_chunks=True), 1):
        chs, heading = chunk_page(pg["text"], heading=heading)
        recs += [(pageno, hd, t) for hd, t in chs if len(t) >= 30]
    if not recs:
        raise SystemExit(f"{p.name}: 텍스트 추출 실패 (스캔본이면 OCR 필요)")

    def ctx(hd, t):
        return f"{p.stem} > {hd}\n{t}" if hd else f"{p.stem}\n{t}"

    vecs = []
    for i in range(0, len(recs), 32):
        vecs += emb.embed_documents([ctx(hd, t) for _, hd, t in recs[i:i + 32]])

    actions = ({
        "_index": INDEX, "_id": f"{h[:12]}-{i}",
        "_source": {"doc": p.stem, "doc_hash": h, "category": category,
                    "page": pg, "heading": hd, "text": t, "vec": v},
    } for i, ((pg, hd, t), v) in enumerate(zip(recs, vecs)))
    helpers.bulk(es, actions)
    es.indices.refresh(index=INDEX)
    return len(recs)


if __name__ == "__main__":
    for f in sys.argv[1:]:
        print(f"{Path(f).name}: {ingest(f)} chunks")
