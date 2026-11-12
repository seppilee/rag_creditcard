from common import es, INDEX


def create(name: str = "cardkb-v1") -> None:
    if es.indices.exists(index=name):
        print("exists:", name)
        return
    es.indices.create(
        index=name,
        aliases={INDEX: {}},
        settings={
            "number_of_shards": 1,      # 운영: 데이터 크기에 맞춰 산정 (샤드당 10~50GB 기준)
            "number_of_replicas": 0,    # 운영: 1 이상
            "analysis": {
                "tokenizer": {"ko_tok": {
                    "type": "nori_tokenizer", "decompound_mode": "mixed",
                    "user_dictionary_rules": ["연회비", "연체이자율", "할부수수료"]}},
                "filter": {"ko_syn": {
                    "type": "synonym_graph",
                    "lenient": True,
                    "synonyms": ["이자율, 금리", "해지, 탈회"]}},
                "analyzer": {
                    "ko": {"type": "custom", "tokenizer": "ko_tok",
                           "filter": ["nori_part_of_speech", "nori_readingform", "lowercase"]},
                    "ko_search": {"type": "custom", "tokenizer": "ko_tok",
                                  "filter": ["nori_part_of_speech", "lowercase", "ko_syn"]},
                },
            },
        },
        mappings={"properties": {
            "doc": {"type": "keyword"},
            "doc_hash": {"type": "keyword"},
            "category": {"type": "keyword"},
            "page": {"type": "integer"},
            "heading": {"type": "text", "analyzer": "ko", "search_analyzer": "ko_search"},
            "text": {"type": "text", "analyzer": "ko", "search_analyzer": "ko_search"},
            "vec": {"type": "dense_vector", "dims": 1024, "index": True,
                    "similarity": "cosine",
                    "index_options": {"type": "int8_hnsw"}},  # 벡터 메모리 약 1/4
        }},
    )
    print("created:", name)


if __name__ == "__main__":
    create()
