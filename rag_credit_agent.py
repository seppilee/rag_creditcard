import os
import sys

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.tools import tool

from cardkb import search

os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "card-rag-agent")
for k in ("GROQ_API_KEY", "LANGSMITH_API_KEY"):
    assert os.environ.get(k), f"{k} 미설정"

MODEL_NAME = "openai/gpt-oss-120b"
TOP_K = 3
MAX_CHARS = 500  # chunk당 컨텍스트에 넣는 최대 글자 수


@tool
def search_card_docs(query: str) -> str:
    """신한카드 약관/상품설명서를 검색합니다. 연회비, 한도, 수수료, 이자율,
    혜택, 연체, 해지 등 카드 관련 질문에 사용하세요.
    검색어는 핵심 용어를 포함해 구체적으로 작성하세요."""
    hits = search(query, k=TOP_K)
    if not hits:
        return "관련 내용을 찾지 못했습니다."
    return "\n\n".join(
        f"[{i}] ({h['doc']} p.{h['page']} {h['article']})\n{h['text'][:MAX_CHARS]}"
        for i, h in enumerate(hits, 1)
    )


SYSTEM_PROMPT = """당신은 신한카드 약관/상품설명서 안내 도우미입니다.
- 카드 관련 질문은 search_card_docs로 검색하고, 검색된 문서 내용만 근거로 답하세요.
- 검색은 최대 2회까지만 하세요. 첫 검색으로 답할 수 있으면 재검색하지 마세요.
- 답변마다 근거를 (문서명 p.쪽 조항) 형식으로 표시하세요.
- 검색 결과에 없으면 모른다고 답하세요. 추측 금지.
- 금액, 비율, 기간은 문서 표현 그대로 인용하세요.
- 마지막에 '정확한 내용은 최신 약관 또는 고객센터에서 확인하세요'를 한 줄로 덧붙이세요.
- 답변은 간결하게 작성하세요."""

model = ChatGroq(model=MODEL_NAME, temperature=0, reasoning_effort="low")
agent = create_agent(model, [search_card_docs], system_prompt=SYSTEM_PROMPT)


def ask(q: str) -> None:
    print(f"\n❓ {q}")
    for step in agent.stream(
        {"messages": [("user", q)]},
        stream_mode="updates",
        config={"run_name": "card-rag-agent", "tags": ["card-rag", MODEL_NAME]},
    ):
        for node, upd in step.items():
            for m in upd["messages"]:
                if m.type == "ai" and m.tool_calls:
                    for tc in m.tool_calls:
                        print(f"  🔎 {tc['name']}({tc['args']})")
                elif m.type == "ai":
                    u = m.usage_metadata or {}
                    print(f"💬 {m.content}")
                    print(f"   tokens in/out: {u.get('input_tokens')}/{u.get('output_tokens')}")


if __name__ == "__main__":
    qs = sys.argv[1:] or ["해외서비스 수수료?"]
    for q in qs:
        ask(q)