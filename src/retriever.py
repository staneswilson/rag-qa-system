from logger import get_logger
from config import RETRIEVE_CANDIDATES, NPROBE

logger = get_logger("retriever")

class Retriever:
    def __init__(self, indexer):
        """
        Initializes the Retriever.
        Args:
            indexer: An already initialized EmbeddingIndexer instance.
        """
        self.indexer = indexer

    def retrieve(self, query: str, k: int = RETRIEVE_CANDIDATES):
        """
        Retrieves document IDs and scores for a given query.
        This method is a simple wrapper around the indexer's search function.

        Args:
            query (str): The user's query text.
            k (int): The number of candidates to retrieve.

        Returns:
            A tuple of (list of document IDs, list of scores).
        """
        # The indexer's search method handles everything:
        # encoding, normalizing, setting nprobe, and searching.
        scores, ids = self.indexer.search(
            query_text=query,
            k=k,
            nprobe=NPROBE  # Use the nprobe value from config
        )

        # The 'ids' are the final, correct document IDs. No mapping is needed.
        return ids, scores