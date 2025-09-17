import math
from typing import List, Dict

def precision_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    retrieved_k = retrieved[:k]
    return sum(1 for r in retrieved_k if r in relevant) / k

def recall_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    retrieved_k = retrieved[:k]
    if not relevant:
        return 0.0
    return sum(1 for r in retrieved_k if r in relevant) / len(relevant)

def dcg_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    dcg = 0.0
    for i, doc in enumerate(retrieved[:k]):
        if doc in relevant:
            dcg += 1.0 / math.log2(i + 2)
    return dcg

def ndcg_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(retrieved, relevant, k) / ideal_dcg
