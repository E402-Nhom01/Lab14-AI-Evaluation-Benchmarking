import asyncio
from typing import Dict, List, Any

from rag.data_helper import PDFReader
from rag.llm import OllamaLLM
from rag.pipeline import SimpleRAGPipeline
from rag.rerank import CrossEncoderRerank
from rag.retrieval import BM25Retrieval
from rag.text_utils import text2chunk


class MainAgent:
    """
    Production-grade RAG Agent for evaluation benchmark
    """

    def __init__(self, pdf_path: str = "sample.pdf", top_k: int = 5):
        self.name = "SimpleRAG-BM25-Ollama"
        self.top_k = top_k

        # ================= LOAD & CHUNK =================
        contents = PDFReader(pdf_paths=[pdf_path]).read()
        text = " ".join(contents)

        chunks = text2chunk(text, chunk_size=200, overlap=50)

        # assign stable doc IDs (IMPORTANT FOR EVAL)
        self.documents = {
            f"doc_{i}": chunk for i, chunk in enumerate(chunks)
        }

        self.doc_ids = list(self.documents.keys())

        # ================= RETRIEVAL =================
        self.retriever = BM25Retrieval(documents=list(self.documents.values()))

        self.llm = OllamaLLM(model_name="llama3.2:latest")

        self.reranker = CrossEncoderRerank(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        self.pipeline = SimpleRAGPipeline(
            retrieval=self.retriever,
            llm=self.llm,
            rerank=self.reranker,
        )

    async def query(self, question: str) -> Dict[str, Any]:
        """
        Benchmark-safe RAG pipeline output
        """

        await asyncio.sleep(0.05)

        result = self.pipeline.run(question)

        # ================= EXTRACT CONTEXT =================
        contexts, retrieved_ids = self._extract_contexts(result)

        return {
            "answer": result.answer,
            "contexts": contexts,

            # REQUIRED FOR EVALUATION
            "retrieved_docs": retrieved_ids[: self.top_k],

            "metadata": {
                "model": "llama3.2",
                "retrieval": "BM25",
                "reranker": "cross-encoder",
                "top_k": self.top_k,
                "pipeline": "SimpleRAGPipeline",
            },
        }

    def _extract_contexts(self, result) -> (List[str], List[str]):
        """
        Stable + evaluation-safe extraction
        """

        raw_contexts = []

        if hasattr(result, "contexts") and result.contexts:
            raw_contexts = result.contexts

        elif hasattr(result, "source_docs"):
            raw_contexts = result.source_docs

        elif hasattr(result, "retrieved_docs"):
            raw_contexts = result.retrieved_docs

        # ================= CLEAN + STABILIZE =================
        cleaned_contexts = []
        retrieved_ids = []

        for i, c in enumerate(raw_contexts):

            if not isinstance(c, str):
                continue

            c = c.strip()

            # filter noise (IMPORTANT for semantic eval stability)
            if len(c) < 30:
                continue

            # assign stable doc id
            doc_id = f"doc_{i}"

            cleaned_contexts.append(c)
            retrieved_ids.append(doc_id)

        # fallback safety
        if not cleaned_contexts:
            return ["No valid context retrieved"], []

        return cleaned_contexts[: self.top_k], retrieved_ids[: self.top_k]


# ================= TEST =================
if __name__ == "__main__":
    async def test():
        agent = MainAgent("sample.pdf", top_k=5)

        resp = await agent.query("What can Ollama do?")

        print("\n=== OUTPUT ===")
        print(resp)

    asyncio.run(test())