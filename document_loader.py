from pypdf import PdfReader

MAX_FILE_SIZE_MB = 20


def validate_pdf(file_name: str, file_size_bytes: int):
    """File type aur size check karta hai. Returns (ok, message)."""
    if not file_name.lower().endswith(".pdf"):
        return False, f"{file_name}: sirf PDF files allowed hain."
    if file_size_bytes > MAX_FILE_SIZE_MB * 1024 * 1024:
        return False, f"{file_name}: file {MAX_FILE_SIZE_MB} MB se badi hai."
    return True, ""


def load_pdf(file, file_name: str):
    """
    PDF ke har page se text nikalta hai.
    Returns: list of {"text", "source", "page"}
    Khali pages skip ho jaate hain.
    """
    try:
        reader = PdfReader(file)
    except Exception as e:
        raise ValueError(f"{file_name} padhi nahi ja saki: {e}")

    if reader.is_encrypted:
        raise ValueError(f"{file_name} password-protected hai.")

    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = " ".join(text.split())  # extra spaces/newlines saaf
        if not text:
            continue
        pages.append({"text": text, "source": file_name, "page": page_number})
    return pages


def load_multiple_pdfs(files):
    """files = list of (file_object, file_name). Saare pages ek list mein."""
    all_pages = []
    for file, name in files:
        all_pages.extend(load_pdf(file, name))
    return all_pages


if __name__ == "__main__":
    path = "documents/sample.pdf"
    with open(path, "rb") as f:
        pages = load_pdf(f, "sample.pdf")
    print(f"Total pages with text: {len(pages)}")
    if pages:
        print("Pehla page (pehle 300 characters):")
        print(pages[0]["text"][:300])
        print("Metadata:", pages[0]["source"], "page", pages[0]["page"])