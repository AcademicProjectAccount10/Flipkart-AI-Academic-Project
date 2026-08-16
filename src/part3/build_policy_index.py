"""Part 3 Tasks 1--2: sentence chunking, local embeddings, and FAISS index build."""

from __future__ import annotations

import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_DOCUMENTS_PATH = PROJECT_ROOT / "kb" / "part3_policy_documents.json"
EVALUATION_QUERIES_PATH = PROJECT_ROOT / "kb" / "part3_retrieval_evaluation_queries.json"
INDEX_DIR = PROJECT_ROOT / "data" / "part3_policy_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def sentence_chunk(text: str) -> list[str]:
    """Split a short policy document sentence-wise without losing parent context."""
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def build_chunks(documents: list[dict[str, str]]) -> list[dict[str, object]]:
    """Attach a stable parent document ID to every sentence chunk."""
    chunks: list[dict[str, object]] = []
    for document in documents:
        for sentence_number, sentence in enumerate(sentence_chunk(document["text"]), start=1):
            chunks.append(
                {
                    "chunk_id": f"{document['document_id']}-S{sentence_number}",
                    "parent_document_id": document["document_id"],
                    "document_title": document["title"],
                    "sentence_number": sentence_number,
                    "text": sentence,
                }
            )
    return chunks


def main() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    documents = json.loads(POLICY_DOCUMENTS_PATH.read_text(encoding="utf-8"))
    evaluation_queries = json.loads(EVALUATION_QUERIES_PATH.read_text(encoding="utf-8"))
    if len(documents) < 12:
        raise ValueError("Part 3 requires at least 12 policy documents.")
    if len(evaluation_queries) < 5:
        raise ValueError("Part 3 requires at least five retrieval-evaluation queries.")

    chunks = build_chunks(documents)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = embedding_model.encode(
        [chunk["text"] for chunk in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "policy_sentences.faiss"))
    (INDEX_DIR / "policy_chunks.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    (INDEX_DIR / "index_metadata.json").write_text(
        json.dumps(
            {
                "embedding_model": EMBEDDING_MODEL_NAME,
                "index_type": "FAISS IndexFlatIP over L2-normalized sentence embeddings",
                "document_count": len(documents),
                "chunk_count": len(chunks),
                "embedding_dimension": int(embeddings.shape[1]),
                "evaluation_query_count": len(evaluation_queries),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Documents: {len(documents)} | Chunks: {len(chunks)} | Dimension: {embeddings.shape[1]}")


if __name__ == "__main__":
    main()
