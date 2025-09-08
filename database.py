import os
import sys
from typing import List
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

PDF_PATH = os.environ.get("RAG_PDF_PATH", "solarcell-basic-knowledge-SolarHub.pdf")
CHROMA_DIR = os.environ.get("CHROMA_DIR", "./chroma_db")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")

def extract_lines_from_pdf(path: str) -> List[str]:
    reader = PdfReader(path)
    lines: List[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text()
        except Exception:
            text = None
        if not text:
            continue
        for line in text.split("\n"):
            if line and line.strip():
                lines.append(line.strip())
    return lines

def chunk_lines(lines: List[str], chunk_size: int = 5, overlap: int = 2) -> List[str]:
    chunks: List[str] = []
    if chunk_size <= 0:
        return chunks
    step = max(1, chunk_size - overlap)
    for i in range(0, len(lines), step):
        chunk = "\n".join(lines[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

if __name__ == "__main__":
    os.makedirs(CHROMA_DIR, exist_ok=True)

    if not os.path.isfile(PDF_PATH):
        print(f"[ERROR] PDF not found: {PDF_PATH}")
        sys.exit(1)

    print(f"[RAG] Extracting from {PDF_PATH}…")
    lines = extract_lines_from_pdf(PDF_PATH)
    print(f"[RAG] Extracted {len(lines)} lines.")

    chunks = chunk_lines(lines, chunk_size=5, overlap=2)
    print(f"[RAG] Created {len(chunks)} chunks.")

    embedding = SentenceTransformerEmbeddings(model_name=EMBED_MODEL_NAME)

    documents = [
        Document(page_content=c, metadata={"source": os.path.basename(PDF_PATH), "id": str(i)})
        for i, c in enumerate(chunks)
    ]

    print("[RAG] Building Chroma DB…")
    vectorstore = Chroma.from_documents(documents, embedding, persist_directory=CHROMA_DIR)
    vectorstore.persist()
    print(f"[RAG] Vector DB built at: {CHROMA_DIR}")
