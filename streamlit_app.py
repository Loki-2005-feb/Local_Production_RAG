import asyncio
from pathlib import Path
import time
import os
import requests

import streamlit as st
import inngest
from dotenv import load_dotenv

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Local RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

/* Main App */
.stApp {
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #111827 30%,
        #1e293b 100%
    );
    color: white;
}

/* Remove Streamlit Header */
header {
    visibility: hidden;
}

/* Main Container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Titles */
.main-title {
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 0.5rem;
    background: linear-gradient(to right, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 3rem;
    font-size: 1.1rem;
}

/* Cards */
.custom-card {
    background: rgba(17, 24, 39, 0.8);
    padding: 2rem;
    border-radius: 20px;
    border: 1px solid rgba(148, 163, 184, 0.2);
    backdrop-filter: blur(12px);
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* Upload Box */
[data-testid="stFileUploader"] {
    background: rgba(30, 41, 59, 0.8);
    border: 2px dashed #38bdf8;
    border-radius: 18px;
    padding: 1rem;
}

/* Input Fields */
.stTextInput input,
.stNumberInput input {
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(
        90deg,
        #0ea5e9,
        #6366f1
    ) !important;

    color: white !important;
    border: none !important;
    border-radius: 12px !important;

    padding: 0.75rem 1.5rem !important;
    font-weight: 700 !important;

    transition: 0.3s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    transform: scale(1.03);
    box-shadow: 0 0 20px rgba(99,102,241,0.5);
}

/* Answer Box */
.answer-box {
    background: rgba(15, 23, 42, 0.9);
    border-left: 5px solid #38bdf8;
    padding: 1.5rem;
    border-radius: 15px;
    margin-top: 1rem;
    font-size: 1.05rem;
    line-height: 1.7;
}

/* Sources */
.source-box {
    background: rgba(30, 41, 59, 0.8);
    padding: 1rem;
    border-radius: 12px;
    margin-top: 1rem;
}

/* Divider */
hr {
    border-color: rgba(148,163,184,0.2);
}

/* Spinner */
.stSpinner > div {
    border-top-color: #38bdf8 !important;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD ENV
# =====================================================

load_dotenv()

# =====================================================
# INNGEST CLIENT
# =====================================================

@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(
        app_id="rag_app",
        is_production=False
    )

# =====================================================
# SAVE PDF
# =====================================================

def save_uploaded_pdf(file) -> Path:

    uploads_dir = Path("uploads")

    uploads_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path = uploads_dir / file.name

    file_bytes = file.getbuffer()

    file_path.write_bytes(file_bytes)

    return file_path

# =====================================================
# SEND INGEST EVENT
# =====================================================

async def send_rag_ingest_event(
    pdf_path: Path
) -> None:

    client = get_inngest_client()

    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )

# =====================================================
# INNGEST API
# =====================================================

def _inngest_api_base() -> str:

    return os.getenv(
        "INNGEST_API_BASE",
        "http://127.0.0.1:8288/v1"
    )

# =====================================================
# FETCH RUNS
# =====================================================

def fetch_runs(event_id: str) -> list[dict]:

    url = f"{_inngest_api_base()}/events/{event_id}/runs"

    resp = requests.get(url)

    resp.raise_for_status()

    data = resp.json()

    return data.get("data", [])

# =====================================================
# WAIT FOR OUTPUT
# =====================================================

def wait_for_run_output(
    event_id: str,
    timeout_s: float = 120.0,
    poll_interval_s: float = 0.5
) -> dict:

    start = time.time()

    last_status = None

    while True:

        runs = fetch_runs(event_id)

        if runs:

            run = runs[0]

            status = run.get("status")

            last_status = status or last_status

            if status in (
                "Completed",
                "Succeeded",
                "Success",
                "Finished"
            ):

                return run.get("output") or {}

            if status in (
                "Failed",
                "Cancelled"
            ):

                raise RuntimeError(
                    f"Function run {status}"
                )

        if time.time() - start > timeout_s:

            raise TimeoutError(
                f"Timed out waiting for run output "
                f"(last status: {last_status})"
            )

        time.sleep(poll_interval_s)

# =====================================================
# QUERY EVENT
# =====================================================

async def send_rag_query_event(
    question: str,
    top_k: int
) -> None:

    client = get_inngest_client()

    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )

    return result[0]

# =====================================================
# HERO SECTION
# =====================================================

st.markdown("""
<div class="main-title">
    Local RAG PDF Assistant
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
    Production-Style Local RAG using Ollama + Qdrant + FastAPI
</div>
""", unsafe_allow_html=True)

# =====================================================
# PDF UPLOAD SECTION
# =====================================================

st.markdown('<div class="custom-card">', unsafe_allow_html=True)

st.subheader("📄 Upload PDF")

uploaded = st.file_uploader(
    "Choose a PDF",
    type=["pdf"],
    accept_multiple_files=False
)

if uploaded is not None:

    with st.spinner(
        "Uploading and ingesting PDF..."
    ):

        path = save_uploaded_pdf(uploaded)

        asyncio.run(
            send_rag_ingest_event(path)
        )

        time.sleep(0.3)

    st.success(
        f"Successfully ingested: {path.name}"
    )

st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# QUESTION SECTION
# =====================================================

st.markdown('<div class="custom-card">', unsafe_allow_html=True)

st.subheader("💬 Ask Questions")

with st.form("rag_query_form"):

    question = st.text_input(
        "Ask anything from your PDFs"
    )

    top_k = st.slider(
        "Retrieved Chunks",
        min_value=1,
        max_value=15,
        value=5
    )

    submitted = st.form_submit_button(
        "Generate Answer"
    )

if submitted and question.strip():

    with st.spinner(
        "Searching documents and generating answer..."
    ):

        event_id = asyncio.run(
            send_rag_query_event(
                question.strip(),
                int(top_k)
            )
        )

        output = wait_for_run_output(
            event_id
        )

        answer = output.get(
            "answer",
            ""
        )

        sources = output.get(
            "sources",
            []
        )

    st.markdown(
        f"""
        <div class="answer-box">
        {answer}
        </div>
        """,
        unsafe_allow_html=True
    )

    if sources:

        st.markdown("### 📚 Sources")

        for s in sources:

            st.markdown(
                f"""
                <div class="source-box">
                    📄 {s}
                </div>
                """,
                unsafe_allow_html=True
            )

st.markdown("</div>", unsafe_allow_html=True)