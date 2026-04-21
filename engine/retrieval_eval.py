from typing import List, Dict

class RetrievalEvaluator:
    def __init__(self):
        pass

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        """
        TODO: Tính toán xem ít nhất 1 trong expected_ids có nằm trong top_k của retrieved_ids không.
        """
        top_retrieved = retrieved_ids[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        TODO: Tính Mean Reciprocal Rank.
        Tìm vị trí đầu tiên của một expected_id trong retrieved_ids.
        MRR = 1 / position (vị trí 1-indexed). Nếu không thấy thì là 0.
        """
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    async def evaluate_batch(self, dataset: List[Dict]) -> Dict:
        """
        Chạy eval cho toàn bộ bộ dữ liệu.
        Dataset cần có trường 'expected_retrieval_ids' và Agent trả về 'retrieved_ids'.
        """
        if not dataset:
            return {
                "total_cases": 0,
                "avg_hit_rate": 0.0,
                "avg_mrr": 0.0,
                "per_case": []
            }

        total_hit_rate = 0.0
        total_mrr = 0.0
        per_case = []

        for idx, case in enumerate(dataset):
            expected_ids = case.get("expected_retrieval_ids", [])
            retrieved_ids = case.get("retrieved_ids", [])
            top_k = case.get("top_k", 3)

            case_hit_rate = self.calculate_hit_rate(expected_ids, retrieved_ids, top_k=top_k)
            case_mrr = self.calculate_mrr(expected_ids, retrieved_ids)

            total_hit_rate += case_hit_rate
            total_mrr += case_mrr

            per_case.append(
                {
                    "case_index": idx,
                    "question": case.get("question", ""),
                    "expected_retrieval_ids": expected_ids,
                    "retrieved_ids": retrieved_ids,
                    "top_k": top_k,
                    "hit_rate": case_hit_rate,
                    "mrr": case_mrr
                }
            )

        total_cases = len(dataset)
        return {
            "total_cases": total_cases,
            "avg_hit_rate": total_hit_rate / total_cases,
            "avg_mrr": total_mrr / total_cases,
            "per_case": per_case
        }
