from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from logger import get_logger
from config import RERANK_MODEL, DEVICE
from typing import List, Dict
import numpy as np

logger = get_logger("reranker")

class Reranker:
    def __init__(self, model_name=RERANK_MODEL, device=DEVICE):
        logger.info(f"Loading reranker model {model_name} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 10, batch_size: int = 16) -> List[str]:
        """
        candidates: list of dicts with keys {"id":..., "text":...}
        Returns top_k candidate ids after reranking.
        """
        pairs = [(query, c["text"]) for c in candidates]
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            texts_q = [q for q, d in batch]
            texts_d = [d for q, d in batch]
            inputs = self.tokenizer(texts_q, texts_d, padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits.squeeze(-1)
                batch_scores = logits.cpu().numpy().tolist()
            scores.extend(batch_scores)
        idx_sorted = np.argsort(scores)[::-1][:top_k]
        return [candidates[i]["id"] for i in idx_sorted]
