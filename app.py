from io import BytesIO

import streamlit as st

from document_loader import load_pdf, validate_pdf
from rag_pipeline import answer_question, build_store

APP_NAME = "Sam Chatbot"

st.set_page_config(page_title=APP_NAME, page_icon="📄")
st.title(f"📄 {APP_NAME}")
st.caption("Upload PDFs, process them, then ask questions. Answers come only from your documents.")

# ---------- Session state ----------
# Keeps data alive across Streamlit reruns
if "store" not in st.session_state:
    st.session_state.store = None  # search index built from the PDFs
if "messages" not in st.session_state:
    st.session_state.messages = []  # chat history
if "doc_names" not in st.session_state:
    st.session_state.doc_names = []  # names of processed files
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0  # changing this key resets the uploader


def format_sources(sources):
    """Turn a list of {"source", "page"} dicts into one display line."""
    return "Sources: " + ", ".join(f"{s['source']} (page {s['page']})" for s in sources)


# ---------- Sidebar: documents ----------
with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if st.button("Process Documents", type="primary", use_container_width=True):
        if not uploaded:
            st.warning("Please upload at least one PDF first.")
        else:
            all_pages, names = [], []
            with st.spinner("Reading PDFs..."):
                for f in uploaded:
                    # Check file type and size before reading
                    ok, msg = validate_pdf(f.name, f.size)
                    if not ok:
                        st.error(msg)
                        continue
                    try:
                        pages = load_pdf(BytesIO(f.getvalue()), f.name)
                    except ValueError as e:
                        st.error(str(e))
                        continue
                    if not pages:
                        st.warning(f"{f.name}: no text found (it may be a scanned PDF).")
                        continue
                    all_pages.extend(pages)
                    names.append(f.name)

            if all_pages:
                store, n_chunks = build_store(all_pages)
                st.session_state.store = store
                st.session_state.doc_names = names
                st.session_state.messages = []  # new documents -> fresh chat
                st.success(f"{len(names)} document(s), {n_chunks} chunks ready.")

    if st.session_state.doc_names:
        st.subheader("Loaded documents")
        for name in st.session_state.doc_names:
            st.write(f"- {name}")

    if st.button("Clear Documents", use_container_width=True):
        st.session_state.store = None
        st.session_state.doc_names = []
        st.session_state.messages = []
        st.session_state.uploader_key += 1  # forces a fresh, empty uploader
        st.rerun()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------- Main area: hint when nothing is loaded ----------
if st.session_state.store is None:
    st.info("Step 1: upload a PDF in the sidebar. Step 2: click 'Process Documents'. Step 3: ask your question below.")

# ---------- Chat history ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption(format_sources(msg["sources"]))

# ---------- New question ----------
question = st.chat_input("Ask a question about your documents")

if question:
    if st.session_state.store is None:
        st.warning("Please upload a PDF and click 'Process Documents' first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    result = answer_question(question, st.session_state.store)
            except Exception as e:
                result = {"answer": f"Something went wrong: {e}", "sources": []}

            st.markdown(result["answer"])
            if result["sources"]:
                st.caption(format_sources(result["sources"]))

        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
        )

st.divider()
st.caption("⚠️ AI answers can be wrong. Please verify important information in the original document.")