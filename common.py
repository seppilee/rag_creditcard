import os
from functools import lru_cache

from elasticsearch import Elasticsearch
from langchain_ollama import OllamaEmbeddings

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
INDEX = "cardkb"  # 실제 인덱스가 아니라 alias (무중단 재색인용)
es = Elasticsearch(ES_URL)
emb = OllamaEmbeddings(model=os.getenv("EMB_MODEL", "bge-m3"))


@lru_cache(maxsize=4096)
def embed_query(q: str) -> tuple:
    # 같은 질문 반복 시 임베딩 호출 절약. 운영에서는 Redis 캐시로 교체
    return tuple(emb.embed_query(q))
