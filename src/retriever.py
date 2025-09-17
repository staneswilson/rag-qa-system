from logger import get_logger
from config import EMBED_MODEL, METRIC, RETRIEVE_CANDIDATES, NPROBE
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

logger = get_logger("retriever")

class Retriever:
    def __init__(self, indexer):
        self.indexer = indexer
        # Use same underlying sentence-transformers model for query encoding
        self.model = indexer.model if hasattr(indexer, "model") else SentenceTransformer(EMBED_MODEL)
        self.metric = METRIC

    def retrieve(self, query: str, k: int = RETRIEVE_CANDIDATES):
        q_emb = self.model.encode([query], convert_to_numpy=True).astype('float32')
        if self.metric == "cosine":
            faiss.normalize_L2(q_emb)
        # If IVF, set nprobe at query time
        try:
            if hasattr(self.indexer.index, "nprobe"):
                self.indexer.index.nprobe = NPROBE
        except Exception:
            pass
        scores, idxs = self.indexer.search(q_emb, k)
        hits = []
        for idx in idxs[0]:
            if idx == -1:
                continue
            # id_map keys are ints in indexer
            val = self.indexer.id_map.get(str(idx)) or self.indexer.id_map.get(int(idx))
            hits.append(val)
        return hits, scores[0].tolist()
