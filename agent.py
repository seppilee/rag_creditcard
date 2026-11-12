import os

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.tools import tool

from search import search

os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "card-rag-service")
MODEL_NAME = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
MAX_CHARS = 500


@tool
def search_card_docs(query: str) -> str:
    """신한카드 약관/상품설명서를 검색합니다. 연회비, 한도, 수수료, 이자율,
    혜택, 연체, 해지 등 카드 관련 질문에 사용하세요. 핵심 용어를 포함해 구체적으로 쓰세요."""
    hits = search(query, k=3)
    if not hits:
        return "관련 내용을 찾지 못했습니다."
    return "\n\n".join(
        f"[{i}] ({h['doc']} p.{h['page']} {h['article']})\n{h['text'][:MAX_CHARS]}"
        for i, h in enumerate(hits, 1))


SYSTEM_PROMPT = """당신은 신한카드 약관/상품설명서 안내 도우미입니다.
- 카드 관련 질문은 search_card_docs로 검색하고, 검색된 내용만 근거로 답하세요.
- 검색은 최대 2회. 첫 검색으로 답할 수 있으면 재검색하지 마세요.
- 근거를 (문서명 p.쪽 조항) 형식으로 표시하세요. 없으면 모른다고 답하고 추측하지 마세요.
- 금액, 비율, 기간은 문서 표현 그대로 인용하세요.
- 마지막에 '정확한 내용은 최신 약관 또는 고객센터에서 확인하세요'를 덧붙이세요."""

agent = create_agent(ChatGroq(model=MODEL_NAME, temperature=0, reasoning_effort="low"),
                     [search_card_docs], system_prompt=SYSTEM_PROMPT)


def answer(q: str) -> str:
    r = agent.invoke({"messages": [("user", q)]},
                     config={"run_name": "card-rag", "tags": ["service"]})
    return r["messages"][-1].content
