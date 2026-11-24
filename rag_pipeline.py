import os
import re

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from prompt import NOT_FOUND_MESSAGE, SYSTEM_PROMPT, build_user_message
from vector_store import TfidfStore, chunk_pages

load_dotenv()


def build_store(pages):
    """Pages -> chunks -> TF-IDF store. Returns (store, chunk_count)."""
    chunks = chunk_pages(pages)
    store = TfidfStore()
    store.build(chunks)
    return store, len(chunks)


def get_llm():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY was not found in the .env file.")
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    return ChatGroq(model=model, temperature=0, api_key=key)


def pick_cited_sources(answer, hits):
    """Return only the sources the model cited in its answer, e.g. (file.pdf, page 2).
    Falls back to the best-scoring chunk if no citation can be parsed."""
    cited, seen = [], set()
    for h in hits:
        key = (h["source"], h["page"])
        if key in seen:
            continue
        # Match "<file name> ... page <number>" inside the answer text
        pattern = rf"{re.escape(h['source'])}[^)\d]*?(?:pages?|p\.?)\s*{h['page']}\b"
        if re.search(pattern, answer, flags=re.IGNORECASE):
            seen.add(key)
            cited.append({"source": h["source"], "page": h["page"]})
    if not cited:
        top = hits[0]  # hits are sorted by score, so this is the best match
        cited.append({"source": top["source"], "page": top["page"]})
    return cited


def answer_question(question: str, store, k: int = 4):
    """Returns {"answer": str, "sources": [{"source", "page"}]}"""
    hits = store.search(question, k=k)

    # Nothing relevant was retrieved: refuse without calling the LLM
    if not hits:
        return {"answer": NOT_FOUND_MESSAGE, "sources": []}

    llm = get_llm()
    response = llm.invoke(
        [("system", SYSTEM_PROMPT), ("human", build_user_message(question, hits))]
    )
    answer = (response.content or "").strip()

    if not answer or NOT_FOUND_MESSAGE in answer:
        return {"answer": NOT_FOUND_MESSAGE, "sources": []}

    return {"answer": answer, "sources": pick_cited_sources(answer, hits)}


if __name__ == "__main__":
    from document_loader import load_pdf

    with open("documents/sample.pdf", "rb") as f:
        pages = load_pdf(f, "sample.pdf")

    store, n = build_store(pages)
    print(f"Chunks: {n}\n")

    for q in [
        "How many questions are in the question bank?",
        "What does Q6 ask about?",
        "Who is the company CEO?",
    ]:
        result = answer_question(q, store)
        print("Q:", q)
        print("A:", result["answer"])
        print("Sources:", result["sources"], "\n")