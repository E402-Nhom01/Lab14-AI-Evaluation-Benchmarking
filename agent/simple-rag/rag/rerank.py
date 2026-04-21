from abc import ABC

from sentence_transformers import CrossEncoder


class BaseRerank(ABC):

    def __init__(self, *args, **kwargs):
        pass

    def rerank(
        self, query: str, documents: list[str], top_k: int = 10, metadata: list[dict] = None,
    ) -> tuple[list, list]:
        raise NotImplementedError


class CrossEncoderRerank(BaseRerank):
    def __init__(self, *args, **kwargs):
        """Initialize the Cross-Encoder reranker.
        - model_name: str: The name of the model to use."""
        super().__init__(*args, **kwargs)
        if "model_name" not in kwargs:
            raise ValueError("Please provide a model_name.")
        self.model = CrossEncoder(kwargs["model_name"])

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10,
        metadata: list[dict] = None,
    ) -> tuple[list, list]:

        cross_inp = [[query, passage] for passage in documents]
        cross_scores = self.model.predict(cross_inp)

        passage_scores = {}
        for idx, score in enumerate(cross_scores):
            passage_scores[idx] = score

        sorted_passages = sorted(
            passage_scores.items(), key=lambda x: x[1], reverse=True
        )

        relevants = []
        scores = []
        selected_meta = [] if metadata else None

        for idx, score in sorted_passages[:top_k]:
            if score > 0:
                relevants.append(documents[idx])
                scores.append(score)
                if metadata:
                    selected_meta.append(metadata[idx])

        if len(relevants) == 0:
            idx, score = sorted_passages[0]
            relevants.append(documents[idx])
            scores.append(score)
            if metadata:
                selected_meta = [metadata[idx]]

        return relevants, scores
