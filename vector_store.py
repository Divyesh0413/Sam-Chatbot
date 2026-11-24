import unicodedata

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def chunk_pages(pages):
    """Har page alag split hota hai, taaki page number sahi rahe."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = []
    chunk_id = 0
    for page in pages:
        for piece in splitter.split_text(page["text"]):
            chunks.append(
                {
                    "text": piece,
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_id": chunk_id,
                }
            )
            chunk_id += 1
    return chunks


def _clean(text: str) -> str:
    # NFKC se "ﬁ" jaise ligature normal "fi" ban jaate hain
    return unicodedata.normalize("NFKC", text).lower()


class TfidfStore:
    """FAISS ki jagah: TF-IDF vectors + cosine similarity (numpy)."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            preprocessor=_clean,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.matrix = None
        self.chunks = []

    def build(self, chunks):
        if not chunks:
            raise ValueError("Koi chunk nahi mila, index nahi ban sakta.")
        self.chunks = chunks
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in chunks])

    def search(self, query: str, k: int = 4):
        """Top-k chunks (score ke saath). Score 0 wale chunks nahi aate."""
        if self.matrix is None:
            return []
        q = self.vectorizer.transform([query])
        # TF-IDF rows already L2-normalized hain, to dot product = cosine similarity
        scores = (self.matrix @ q.T).toarray().ravel()
        top = np.argsort(scores)[::-1][:k]
        results = []
        for i in top:
            if scores[i] > 0:
                item = dict(self.chunks[i])
                item["score"] = float(scores[i])
                results.append(item)
        return results


if __name__ == "__main__":
    from document_loader import load_pdf

    with open("documents/sample.pdf", "rb") as f:
        pages = load_pdf(f, "sample.pdf")

    chunks = chunk_pages(pages)
    print(f"Pages: {len(pages)} | Chunks: {len(chunks)}")
    print("Chunk lengths:", [len(c["text"]) for c in chunks])

    store = TfidfStore()
    store.build(chunks)

    for q in ["What is Artificial Intelligence?", "Who is the company CEO?"]:
        print(f"\nQuestion: {q}")
        hits = store.search(q, k=3)
        if not hits:
            print("  (koi match nahi)")
        for h in hits:
            print(f"  score={h['score']:.3f} | {h['source']} page {h['page']} | {h['text'][:90]}...")