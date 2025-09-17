import os
import torch
import numpy as np
import faiss
from itertools import islice
from pathlib import Path
from sentence_transformers import SentenceTransformer
from logger import get_logger

## UPDATED: Import the specific config values we need ##
from config import (
    EMBED_MODEL,
    EMBED_BATCH,
    OUTPUT_DIR,
    INDEX_PATH,
    METRIC,
    DEVICE,
    NLIST,
    NPROBE,
    RETRIEVE_CANDIDATES,
    PQ_M,
    PQ_BITS,
    NUM_TRAIN_SAMPLES
)


logger = get_logger("indexer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def document_generator(data_directory: Path):
    """
    A generator that yields documents one by one from a directory.
    This prevents loading the entire dataset into memory.
    """
    file_paths = list(data_directory.glob('*.*'))
    total_files = len(file_paths)
    logger.info(f"Found {total_files} files to process.")
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            doc_id = int(file_path.stem)
            yield {"id": doc_id, "text": text}
        except (IOError, ValueError) as e:
            logger.warning(f"Skipping file {file_path} due to error: {e}")
            continue

class EmbeddingIndexer:
    def __init__(self, model_name=EMBED_MODEL):
        ## UPDATED: Respect the DEVICE setting from config.py ##
        logger.info(f"Initializing SentenceTransformer on device: {DEVICE}")
        self.model = SentenceTransformer(model_name, device=DEVICE)
        self.index: faiss.Index = None

    def build_from_generator(
        self,
        doc_generator,
        total_docs,
        metric=METRIC,
        nlist=NLIST,
        pq_m=PQ_M,
        pq_bits=PQ_BITS
    ):
        """
        Builds a memory-efficient IndexIVFPQ index from a document generator.
        """
        dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Vector dimension is {dim}")

        # NEW: Automatically adjust nlist if there aren't enough documents.
        if total_docs < nlist:
            logger.warning(f"Number of documents ({total_docs}) is less than nlist ({nlist}).")
            nlist = max(1, int(total_docs / 8)) # Ensure at least 8 docs per cluster, with a min of 1.
            logger.warning(f"Adjusting nlist to {nlist} to be smaller than the number of documents.")

        if metric == "cosine":
            quantizer = faiss.IndexFlatIP(dim)
            metric_faiss = faiss.METRIC_INNER_PRODUCT
            logger.info("Using cosine similarity (inner product on L2-normalized vectors)")
        else:
            quantizer = faiss.IndexFlatL2(dim)
            metric_faiss = faiss.METRIC_L2
            logger.info("Using L2 distance")
        
        # Initialize the IndexIVFPQ index
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, pq_m, pq_bits, metric_faiss)
        
        # --- Train the Index ---
        # MODIFIED: Use total_docs for training if it's less than NUM_TRAIN_SAMPLES
        num_train = min(total_docs, NUM_TRAIN_SAMPLES)
        
        # We also need to make sure we have enough data for the chosen nlist. Faiss recommends at least 39*nlist
        if num_train < 39 * nlist:
            logger.warning(f"Number of training samples ({num_train}) is less than the Faiss recommendation of 39 * nlist ({39*nlist})."
                        " Index quality may be suboptimal.")

        logger.info(f"Collecting {num_train} samples for training the index...")
        train_samples = list(islice(doc_generator, num_train))
        
        if not train_samples:
            raise ValueError("Document generator yielded no data for training.")
            
        train_texts = [d['text'] for d in train_samples]
        
        logger.info("Encoding training samples...")
        train_embeddings = self.model.encode(
            train_texts, batch_size=EMBED_BATCH, convert_to_numpy=True, show_progress_bar=True
        ).astype('float32')

        if metric == "cosine":
            faiss.normalize_L2(train_embeddings)
            
        logger.info("Training index...")
        index.train(train_embeddings)
        logger.info("Index training complete.")

        # --- Wrap index for custom IDs and add all documents ---
        self.index = faiss.IndexIDMap2(index)
        
        logger.info("Adding training samples to the index...")
        train_ids = np.array([d['id'] for d in train_samples], dtype=np.int64)
        self.index.add_with_ids(train_embeddings, train_ids)
        
        logger.info("Adding remaining documents to the index...")
        docs_processed = len(train_samples)
        
        while True:
            batch_docs = list(islice(doc_generator, EMBED_BATCH * 20))
            if not batch_docs:
                break
            
            batch_texts = [d['text'] for d in batch_docs]
            batch_ids = np.array([d['id'] for d in batch_docs], dtype=np.int64)
            
            batch_embeddings = self.model.encode(
                batch_texts, batch_size=EMBED_BATCH, show_progress_bar=True
            ).astype('float32')
            if metric == "cosine":
                faiss.normalize_L2(batch_embeddings)
            
            self.index.add_with_ids(batch_embeddings, batch_ids)
            docs_processed += len(batch_docs)
            logger.info(f"Added {docs_processed}/{total_docs} vectors to index.")

        logger.info(f"Index built successfully. Total vectors: {self.index.ntotal}")

    def save(self, index_path=INDEX_PATH):
        logger.info(f"Saving FAISS index to {index_path}")
        faiss.write_index(self.index, str(index_path))

    def load(self, index_path=INDEX_PATH):
        logger.info(f"Loading FAISS index from {index_path}")
        self.index = faiss.read_index(str(index_path))
        logger.info(f"Index loaded. Total vectors: {self.index.ntotal}")

    def search(self, query_text: str, k: int = RETRIEVE_CANDIDATES, nprobe: int = NPROBE):
        """
        Searches the index for the top-k most similar documents.
        
        Args:
            query_text (str): The text to search for.
            k (int): The number of candidate documents to retrieve. From config.
            nprobe (int): The number of clusters to probe. From config.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing scores and document IDs.
        """
        if self.index is None:
            raise ValueError("Index not built or loaded. Call build() or load() first.")

        query_emb = self.model.encode(query_text, convert_to_numpy=True).astype('float32').reshape(1, -1)
        
        # Normalize query vector if the index uses cosine similarity
        if self.index.index.metric_type == faiss.METRIC_INNER_PRODUCT:
            faiss.normalize_L2(query_emb)

        # Set the number of clusters to search
        if hasattr(self.index.index, "nprobe"):
             self.index.index.nprobe = nprobe
        
        scores, ids = self.index.search(query_emb, k)
        return scores[0], ids[0]