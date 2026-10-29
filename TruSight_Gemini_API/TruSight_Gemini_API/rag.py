"""PDF extraction, chunking and retrieval using hosted Gemini embeddings."""
import fitz
import numpy as np
from google.genai import types
from gemini_client import get_client, EMBEDDING_MODEL


def extract_pdf(pdf_path):
    pages = []
    with fitz.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:
                pages.append({"page": page_number, "text": text})
    return pages


def chunk_text(pages, chunk_size=200, overlap=40):
    """
    Sentence-aware paragraph chunking.
    Preserves sentence boundaries and negative qualifiers (e.g., 'not', 'never', 'denies')
    across window overlaps while strictly preserving the 1-indexed document page reference.
    """
    import re
    chunks = []
    for page_data in pages:
        text = page_data["text"]
        # Split on sentence terminators while preserving punctuation
        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        if not sentences:
            continue

        current_chunk_words = []
        for sentence in sentences:
            words = sentence.split()
            if len(current_chunk_words) + len(words) > chunk_size and current_chunk_words:
                chunks.append({
                    "text": " ".join(current_chunk_words),
                    "page": page_data["page"],
                })
                # Retain overlap words from the end of previous chunk to prevent boundary truncation
                overlap_words = current_chunk_words[-overlap:] if len(current_chunk_words) > overlap else current_chunk_words
                current_chunk_words = list(overlap_words) + words
            else:
                current_chunk_words.extend(words)

        if current_chunk_words:
            chunks.append({
                "text": " ".join(current_chunk_words),
                "page": page_data["page"],
            })
    return chunks


def _embed(texts, task_type):
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    client = get_client()
    vectors = []
    # Gemini accepts multiple text contents per request. Batch to keep request sizes manageable.
    for offset in range(0, len(texts), 64):
        batch = texts[offset:offset + 64]
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        if not response.embeddings or len(response.embeddings) != len(batch):
            raise RuntimeError("Gemini returned an unexpected number of embeddings.")
        vectors.extend(item.values for item in response.embeddings)
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def build_index(chunks):
    if not chunks:
        raise ValueError("No text chunks were created from the PDF.")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = _embed(texts, "RETRIEVAL_DOCUMENT")
    # Index is a normalized NumPy matrix; no local embedding model or FAISS is needed.
    return embeddings, embeddings


def retrieve(query, chunks, index, k=3):
    if not query or not chunks or index is None:
        return []
    query_vector = _embed([query], "RETRIEVAL_QUERY")[0]
    scores = np.asarray(index) @ query_vector
    selected = np.argsort(scores)[::-1][:min(k, len(chunks))]
    return [
        {
            "text": chunks[int(i)]["text"],
            "page": chunks[int(i)]["page"],
            "distance": round(float(1.0 - scores[int(i)]), 4),
            "similarity": round(float(scores[int(i)]), 4),
        }
        for i in selected
    ]
