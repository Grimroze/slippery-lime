import os
from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader
from src.config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Reads all pages of a PDF file and returns concatenated text."""
    reader = PdfReader(str(pdf_path))
    text_parts = []
    for page_idx, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if page_text.strip():
            text_parts.append(page_text)
    return "\n\n".join(text_parts)

def extract_text_from_file(file_path: Path) -> str:
    """Extracts text from .txt, .md, or .pdf files."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    elif suffix in [".txt", ".md"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return ""

def chunk_text(text: str, source_name: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """
    Splits text into overlapping word chunks.
    Each chunk gets a unique ID, text content, and document source.
    """
    words = text.split()
    chunks = []
    chunk_idx = 0

    if not words:
        return []

    # Sliding window chunking
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words)

        chunks.append({
            "id": f"{source_name}_chunk_{chunk_idx}",
            "content": chunk_str,
            "source": source_name,
            "chunk_index": chunk_idx
        })

        chunk_idx += 1
        # Advance by (chunk_size - overlap)
        step = chunk_size - overlap
        if step <= 0:
            step = chunk_size
        start += step

    return chunks

def load_all_documents(data_dir: Path = DATA_DIR) -> List[Dict[str, Any]]:
    """
    Scans the data directory, reads all PDFs and text files,
    and returns a combined list of parsed chunks.
    """
    supported_extensions = [".pdf", ".txt", ".md"]
    all_chunks = []

    files = [f for f in data_dir.iterdir() if f.is_file() and f.suffix.lower() in supported_extensions]

    if not files:
        print(f"[Warning] No documents found in {data_dir}. Please place .pdf or .txt files there.")
        return []

    print(f"[DocLoader] Found {len(files)} document(s) in {data_dir}:")
    for file_path in files:
        print(f"  - Reading: {file_path.name}")
        raw_text = extract_text_from_file(file_path)
        if raw_text.strip():
            file_chunks = chunk_text(raw_text, source_name=file_path.name)
            all_chunks.extend(file_chunks)
            print(f"    -> Extracted {len(file_chunks)} chunks.")

    print(f"[DocLoader] Total chunks loaded: {len(all_chunks)}\n")
    return all_chunks

if __name__ == "__main__":
    chunks = load_all_documents()
    if chunks:
        print(f"Sample chunk 0: {chunks[0]['content'][:120]}...")
