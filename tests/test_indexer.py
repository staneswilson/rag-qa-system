from src.indexer import EmbeddingIndexer

def test_build_and_search():
    docs = [{"id": f"doc{i}", "text": f"This document talks about cats {i}"} for i in range(10)]
    idx = EmbeddingIndexer()
    idx.build(docs, use_ivf=False)
    from src.retriever import Retriever
    retr = Retriever(idx)
    hits, _ = retr.retrieve("cats", k=3)
    assert len(hits) == 3
