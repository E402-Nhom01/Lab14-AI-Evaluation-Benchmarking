from abc import ABC

from .llm import BaseLLM
from .prompt import ANSWER_PROMPT
from .rerank import BaseRerank
from .retrieval import BaseRetrieval


class Answer:
    def __init__(self, answer: str, contexts: list[str]):
        self.answer = answer
        self.contexts = contexts


class Pipeline(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def run(self, query: str) -> Answer:
        raise NotImplementedError


class SimpleRAGPipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not kwargs.get("retrieval"):
            raise ValueError("Please provide a `retrieval` model.")
        if not kwargs.get("llm"):
            raise ValueError("Please provide a `llm` model.")
        self.retrieval = kwargs.get("retrieval")
        # assert retrieval must be BaseRetrieval class or it inherits from BaseRetrieval
        assert issubclass(self.retrieval.__class__, BaseRetrieval)
        self.rerank = kwargs.get("rerank")
        if self.rerank:
            # assert rerank must be BaseRerank class or it inherits from BaseRerank
            assert issubclass(self.rerank.__class__, BaseRerank)
        self.llm = kwargs.get("llm")
        # assert llm must be BaseLLM class or it inherits from BaseLLM
        assert issubclass(self.llm.__class__, BaseLLM)
        self.retrieval_top_k = kwargs.get("retrieval_top_k", 100)
        self.rerank_top_k = kwargs.get("rerank_top_k", 3)

    def run(self, query: str) -> Answer:
        print("\n================ PIPELINE START ================\n")
        print(f"QUERY: {query}\n")

        # 🔍 STEP 1: RETRIEVE
        relevant_docs, relevant_meta = self.retrieval.retrieve(
            query, top_k=self.retrieval_top_k
        )

        print("\n----- AFTER RETRIEVAL -----")
        for i, doc in enumerate(relevant_docs[:5]):  # log top 5 only
            if relevant_meta:
                meta = relevant_meta[i]
                chunk_id = meta.get("chunk_id", f"chunk_{i}")
                source = meta.get("source_document", "unknown")
            else:
                chunk_id = f"chunk_{i}"
                source = "unknown"

            print(f"[{i+1}] {chunk_id} | {source}")
            print(f"→ {doc[:100]}...\n")

        # 🔄 STEP 2: RERANK
        if self.rerank:
            reranked_docs, scores = self.rerank.rerank(
                query,
                relevant_docs,
                top_k=self.rerank_top_k,
                metadata=relevant_meta  # 👈 FIX HERE
            )

            print("\n----- AFTER RERANK -----")
            for i, doc in enumerate(reranked_docs):
                if relevant_meta:
                    # ⚠️ need mapping (important)
                    meta = relevant_meta[relevant_docs.index(doc)]
                    chunk_id = meta.get("chunk_id", f"chunk_{i}")
                    source = meta.get("source_document", "unknown")
                else:
                    chunk_id = f"chunk_{i}"
                    source = "unknown"

                print(f"[{i+1}] {chunk_id} | {source}")
                print(f"→ Score: {scores[i]:.4f}")
                print(f"→ {doc[:100]}...\n")

        else:
            reranked_docs = relevant_docs

        # 🧠 STEP 3: LLM
        print("\n----- FINAL CONTEXT TO LLM -----")
        for i, doc in enumerate(reranked_docs):
            print(f"[{i+1}] {doc[:100]}...\n")

        prompt = ANSWER_PROMPT.format(
            query=query,
            context="\n".join(reranked_docs)
        )

        answer = self.llm.generate(prompt)

        print("\n================ PIPELINE END =================\n")

        return Answer(answer=answer, contexts=reranked_docs)
