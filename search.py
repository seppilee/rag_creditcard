from langsmith import traceable

from common import es, INDEX, embed_query


@traceable(run_type="retriever", name="hybrid_search")
def search(query: str, k: int = 3, pool: int = 20, category: str | None = None) -> list[dict]:
    filt = [{"term": {"category": category}}] if category else []
    qv = list(embed_query(query))

    bm = es.search(index=INDEX, size=pool, source_excludes=["vec"], query={"bool": {
        "must": {"multi_match": {"query": query, "fields": ["heading^2", "text"]}},
        "filter": filt}})["hits"]["hits"]
    kn = es.search(index=INDEX, size=pool, source_excludes=["vec"], knn={
        "field": "vec", "query_vector": qv, "k": pool,
        "num_candidates": pool * 5, "filter": filt})["hits"]["hits"]

    score, src = {}, {}
    for ranking in (bm, kn):                              # RRF
        for r, h in enumerate(ranking):
            score[h["_id"]] = score.get(h["_id"], 0.0) + 1.0 / (60 + r)
            src[h["_id"]] = h["_source"]
    top = sorted(score, key=score.get, reverse=True)[:k]
    return [{"id": i, "doc": src[i]["doc"], "page": src[i]["page"],
             "article": src[i]["heading"], "text": src[i]["text"],
             "score": round(score[i], 4)} for i in top]
