from abc import ABC
from typing import Optional, List, Dict

import numpy as np
from rank_bm25 import BM25Okapi


class BaseRetrieval(ABC):
    def ingest(self, documents: List[str], metadata: Optional[List[dict]] = None) -> bool:
        raise NotImplementedError

    def retrieve(self, query: str, top_k: int = 10) -> List[dict]:
        raise NotImplementedError

    def rerank(self, query: str, documents: List[dict], top_k: int = 10) -> List[dict]:
        raise NotImplementedError("This retrieval model does not support reranking.")


class BM25Retrieval(BaseRetrieval):
    def __init__(self, *args, **kwargs):
        super().__init__()

        documents = kwargs.get("documents")
        if not documents:
            raise ValueError("Please provide documents")

        # 🔥 NORMALIZE INPUT (string → chunk dict)
        if isinstance(documents[0], str):
            documents = [
                {
                    "chunk_id": f"chunk_{i}",
                    "chunk_text": doc,
                    "source_document": "unknown",
                }
                for i, doc in enumerate(documents)
            ]

        self.__ingest__(documents)

    def __ingest__(self, chunks: List[Dict]):
        assert len(chunks) > 0, "Chunk list is empty."

        self.chunks = chunks
        self.documents = [c["chunk_text"] for c in chunks]
        self.metadata = chunks

        # Tokenize
        tokenized_docs = [doc.split() for doc in self.documents]

        # Init BM25
        self.bm25 = BM25Okapi(tokenized_docs)

        # 🔥 LOGGING
        print("\n===== BƯỚC 2: CHUNK DỮ LIỆU =====")
        for c in chunks:
            print(f'{c["chunk_id"]} → {c["chunk_text"][:80]}... → {c["source_document"]}')
        print("=================================\n")

        return True

    def retrieve(self, query: str, top_k: int = 10):
        if not hasattr(self, "bm25"):
            raise RuntimeError("BM25 not initialized")

        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_n = np.argsort(scores)[::-1][:top_k]

        results = []

        print("\n===== RETRIEVE LOG =====")
        for rank, idx in enumerate(top_n):
            chunk = self.metadata[idx]
            score = scores[idx]

            print(f"[{rank+1}] {chunk['chunk_id']}")
            print(f"→ Score: {score:.4f}")
            print(f"→ Source: {chunk['source_document']}")
            print(f"→ Text: {chunk['chunk_text'][:100]}...\n")

            results.append({
                "chunk_id": chunk["chunk_id"],
                "chunk_text": chunk["chunk_text"],
                "source_document": chunk["source_document"],
                "score": float(score),
            })

        print("========================\n")

        return results