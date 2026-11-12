import os
import sys
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 키는 셸에서 export 해 둔 환경변수를 사용 (GROQ_API_KEY, LANGSMITH_API_KEY)
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "rag-agent-practice")
for k in ("GROQ_API_KEY", "LANGSMITH_API_KEY"):
    assert os.environ.get(k), f"{k} 미설정"

MODEL_NAME = "openai/gpt-oss-120b"
DOCS_DIR = Path("./docs")  # .md/.txt를 여기에 넣으면 자동 로드


# ── 1. 문서 로드 (없으면 샘플) ──────────────────────────────
def load_docs() -> list[Document]:
    docs = []
    if DOCS_DIR.exists():
        for p in DOCS_DIR.rglob("*"):
            if p.suffix in {".md", ".txt"}:
                docs.append(Document(page_content=p.read_text(encoding="utf-8"),
                                     metadata={"source": str(p)}))
    if docs:
        return docs
    return [
        Document(page_content=(
            "반품 정책: 상품 수령 후 7일 이내 반품 신청 가능합니다. "
            "단순 변심의 경우 왕복 배송비는 고객 부담입니다. "
            "상품 불량·오배송은 판매자가 배송비를 부담합니다. "
            "개봉한 식품, 속옷류는 반품이 불가합니다."),
            metadata={"source": "sample/return_policy.md"}),
        Document(page_content=(
            "배송 안내: 오후 11시 이전 주문 시 다음 날 도착합니다. "
            "도서산간 지역은 1~2일 추가 소요되며 추가 배송비 3,000원이 부과됩니다. "
            "제주 지역은 추가 배송비 5,000원입니다."),
            metadata={"source": "sample/shipping_guide.md"}),
        Document(page_content=(
            "회원 등급: 일반, 실버, 골드, VIP. 골드 이상은 월 4회 무료 반품 쿠폰을 제공합니다. "
            "VIP는 모든 주문 무료배송이며 전용 상담 채널을 이용할 수 있습니다."),
            metadata={"source": "sample/membership.md"}),
    ]


# ── 2. 청킹 + 임베딩 + 인덱싱 ───────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(load_docs())

embeddings = OllamaEmbeddings(model="bge-m3")
store = InMemoryVectorStore.from_documents(chunks, embeddings)
print(f"[index] {len(chunks)} chunks")


# ── 3. 툴 ──────────────────────────────────────────────────
@tool
def search_knowledge_base(query: str) -> str:
    """사내 정책/FAQ 지식베이스를 검색합니다. 반품, 배송, 회원 등급 등
    정책 질문에 사용하세요. 검색어는 질문의 핵심 키워드 중심으로 구체적으로 작성하세요."""
    results = store.similarity_search_with_score(query, k=4)
    if not results:
        return "관련 문서를 찾지 못했습니다."
    return "\n\n".join(
        f"[{i}] (source={d.metadata.get('source')}, score={s:.3f})\n{d.page_content}"
        for i, (d, s) in enumerate(results, 1)
    )


@tool
def calculate_shipping_cost(weight_kg: float) -> str:
    """상품의 무게(kg)를 입력받아 기본 배송비를 계산합니다."""
    return f"상품 무게 {weight_kg}kg의 기본 배송비는 {int(weight_kg * 3000)}원입니다."


# ── 4. 에이전트 ─────────────────────────────────────────────
SYSTEM_PROMPT = """당신은 쇼핑몰 고객지원 에이전트입니다.
- 정책/규정 질문은 반드시 search_knowledge_base로 검색한 뒤 그 내용만 근거로 답하세요.
- 검색 결과가 부족하면 다른 검색어로 한 번 더 검색하고, 그래도 없으면 모른다고 답하세요.
- 답변 끝에 근거로 쓴 source를 표시하세요.
- 배송비 계산은 calculate_shipping_cost를 사용하세요."""

model = ChatGroq(model=MODEL_NAME, temperature=0)
agent = create_agent(model, [search_knowledge_base, calculate_shipping_cost],
                     system_prompt=SYSTEM_PROMPT)


def ask(q: str) -> None:
    print(f"\n❓ {q}")
    for step in agent.stream({"messages": [("user", q)]}, stream_mode="updates",
                             config={"run_name": "rag-agent", "tags": ["rag", MODEL_NAME]}):
        for node, upd in step.items():
            for m in upd["messages"]:
                if m.type == "ai" and m.tool_calls:
                    for tc in m.tool_calls:
                        print(f"  🔎 {tc['name']}({tc['args']})")
                elif m.type == "ai":
                    print(f"💬 {m.content}")


if __name__ == "__main__":
    qs = sys.argv[1:] or [
        "단순 변심으로 반품하면 배송비는 누가 내?",
        "제주도로 3kg 상품 보내면 배송비 총 얼마야?",   # RAG + 계산 툴 조합
        "골드 회원 혜택 알려줘",
        "환불 계좌 변경은 어떻게 해?",                    # 문서에 없음 → 모른다고 해야 정상
    ]
    for q in qs:
        ask(q)