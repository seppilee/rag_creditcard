import os
from dotenv import load_dotenv
load_dotenv()

from arize.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

tp = register(
    space_id=os.environ["ARIZE_SPACE_ID"],
    api_key=os.environ["ARIZE_API_KEY"],
    project_name="card-rag",
)
LangChainInstrumentor().instrument(tracer_provider=tp)

from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

docs = [
    "해외 이용 시 국제브랜드 수수료 1%가 부과됩니다.",
    "연회비는 국내전용 1만원, 해외겸용 1만2천원입니다.",
    "전월 실적 30만원 이상 시 통신비 10% 할인됩니다.",
]
vs = InMemoryVectorStore.from_texts(docs, OllamaEmbeddings(model="bge-m3"))   # 임베딩 = Ollama bge-m3
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

retriever = vs.as_retriever(search_kwargs={"k": 2})

prompt = ChatPromptTemplate.from_template(
    "다음 약관만 근거로 한국어로 답하세요.\n\n{context}\n\n질문: {question}"
)
fmt = lambda ds: "\n".join(d.page_content for d in ds)



chain = (
    {"context": retriever | fmt, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
print(chain.invoke("해외에서 쓰면 수수료가 얼마야?"))

tp.force_flush()   # 스크립트가 바로 끝나면 span이 안 올라갈 수 있어서