from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from langchain_ollama import OllamaEmbeddings

# ---------------------------------------------------
# Local Ollama Embedding Model
# ---------------------------------------------------

EMBED_MODEL = "nomic-embed-text"

embeddings = OllamaEmbeddings(
    model=EMBED_MODEL
)

# ---------------------------------------------------
# Get embedding dimension dynamically
# ---------------------------------------------------

sample_embedding = embeddings.embed_query("test")
EMBED_DIM = len(sample_embedding)

# ---------------------------------------------------
# Text Splitter
# ---------------------------------------------------

splitter = SentenceSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# ---------------------------------------------------
# Load + Chunk PDF
# ---------------------------------------------------

def load_and_chunk_pdf(path: str):

    docs = PDFReader().load_data(file=path)

    texts = [
        d.text
        for d in docs
        if getattr(d, "text", None)
    ]

    chunks = []

    for text in texts:
        chunks.extend(
            splitter.split_text(text)
        )

    return chunks

# ---------------------------------------------------
# Generate Local Embeddings
# ---------------------------------------------------

def embed_texts(texts: list[str]) -> list[list[float]]:

    return embeddings.embed_documents(texts)