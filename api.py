from fastapi import FastAPI
from pydantic import BaseModel

from agent import answer
from search import search

app = FastAPI(title="card-rag")


class AskRequest(BaseModel):
    q: str


@app.get("/search")
def api_search(q: str, k: int = 3, category: str | None = None) -> list[dict]:
    return search(q, k=k, category=category)


@app.post("/ask")
def api_ask(body: AskRequest) -> dict[str, str]:
    return {"answer": answer(body.q)}


@app.get("/healthz")
def health() -> dict[str, bool]:
    return {"ok": True}
