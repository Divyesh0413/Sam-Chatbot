import csv
import re
import sys
import time
from pathlib import Path

from document_loader import load_pdf
from prompt import NOT_FOUND_MESSAGE
from rag_pipeline import answer_question, build_store

QUESTIONS_FILE = Path("tests/test_questions.csv")
RESULTS_FILE = Path("tests/test_results.csv")
DOCS_DIR = Path("documents")
PAUSE_SECONDS = 1.0  # small pause between LLM calls to avoid rate limits


NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
# Models often output special dash characters instead of a normal "-"
DASHES = "\u2010\u2011\u2012\u2013\u2014\u2212"


def extract_tokens(text):
    """Pull numbers out of text, e.g. 'Rs. 4,500' -> {'4500'}, 'Three tanks' -> {'3'}."""
    for d in DASHES:
        text = text.replace(d, "-")
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)  # remove thousands separators
    tokens = set(re.findall(r"\d+(?:[.:\-]\d+)*", text))
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in NUMBER_WORDS:
            tokens.add(NUMBER_WORDS[word])
    return tokens


def is_answer_correct(row, answer):
    """Rough automatic check. Review any 'no' rows by hand."""
    if row["type"] == "unavailable":
        return NOT_FOUND_MESSAGE in answer
    if NOT_FOUND_MESSAGE in answer:
        return False
    return extract_tokens(row["expected_answer"]).issubset(extract_tokens(answer))


def is_source_correct(row, sources):
    """Answerable: expected file+page must be among the sources. Unavailable: no sources."""
    if row["type"] == "unavailable":
        return len(sources) == 0
    wanted = (row["expected_source"], int(row["expected_page"]))
    return wanted in {(s["source"], s["page"]) for s in sources}


def main():
    with open(QUESTIONS_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Load only the documents that the test sheet refers to
    doc_names = sorted({r["expected_source"] for r in rows if r["type"] == "answerable"})
    pages = []
    for name in doc_names:
        path = DOCS_DIR / name
        if not path.exists():
            sys.exit(f"Missing file: {path}. Put all 3 test PDFs in the documents folder.")
        with open(path, "rb") as f:
            pages.extend(load_pdf(f, name))

    store, n_chunks = build_store(pages)
    print(f"Loaded {len(doc_names)} documents, {n_chunks} chunks, {len(rows)} questions\n")

    results = []
    for i, row in enumerate(rows, start=1):
        start = time.perf_counter()
        try:
            result = answer_question(row["question"], store)
            answer, sources = result["answer"], result["sources"]
        except Exception as e:
            answer, sources = f"ERROR: {e}", []
            if i == 1:
                import traceback
                traceback.print_exc()  # show the full error once, for the first question
        elapsed = time.perf_counter() - start

        source_ok = is_source_correct(row, sources)
        answer_ok = is_answer_correct(row, answer)

        results.append(
            {
                "id": row["id"],
                "question": row["question"],
                "type": row["type"],
                "expected_source": row["expected_source"],
                "expected_page": row["expected_page"],
                "expected_answer": row["expected_answer"],
                "retrieved_source": "; ".join(f"{s['source']} p{s['page']}" for s in sources) or "none",
                "source_correct": "yes" if source_ok else "no",
                "answer_correct": "yes" if answer_ok else "no",
                "actual_answer": " ".join(answer.split()),
                "time_s": f"{elapsed:.1f}",
            }
        )

        status = "OK  " if (source_ok and answer_ok) else "FAIL"
        print(f"[{i:2d}/{len(rows)}] {status} {elapsed:4.1f}s  {row['question']}")
        if row["type"] == "answerable":
            time.sleep(PAUSE_SECONDS)

    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # ---------- Summary ----------
    def count(kind, field):
        subset = [r for r in results if r["type"] == kind]
        return sum(r[field] == "yes" for r in subset), len(subset)

    a_ok, a_n = count("answerable", "answer_correct")
    s_ok, _ = count("answerable", "source_correct")
    u_ok, u_n = count("unavailable", "answer_correct")
    avg_time = sum(float(r["time_s"]) for r in results) / len(results)

    print("\n========== SUMMARY ==========")
    print(f"Answerable - correct answer : {a_ok}/{a_n}")
    print(f"Answerable - correct source : {s_ok}/{a_n}")
    print(f"Unavailable - correct refusal: {u_ok}/{u_n}")
    print(f"Average response time        : {avg_time:.1f}s")
    print(f"Results saved to {RESULTS_FILE}")

    failed = [r for r in results if r["answer_correct"] == "no" or r["source_correct"] == "no"]
    if failed:
        print("\nRows to review:")
        for r in failed:
            print(f"  #{r['id']} {r['question']}")
            print(f"     expected: {r['expected_answer']} ({r['expected_source']} p{r['expected_page']})")
            print(f"     got     : {r['actual_answer'][:150]}")
            print(f"     sources : {r['retrieved_source']}")


if __name__ == "__main__":
    main()