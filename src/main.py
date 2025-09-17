import argparse
import json
import numpy as np
import pickle
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import (DATA_DIR, OUTPUT_DIR, RETRIEVE_CANDIDATES, STARTUP_NAME,
                    TOP_K)
from indexer import EmbeddingIndexer
# This is the new generator function from indexer.py
from indexer import document_generator as chunks_generator
from logger import get_logger
from preprocess import preprocess_file, preprocess_folder_parallel
from eval_metrics import precision_at_k, recall_at_k, ndcg_at_k
from reranker import Reranker
from retriever import Retriever
from utils import save_json, zip_dir

logger = get_logger("main")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

## PASTE THIS UPDATED FUNCTION INTO YOUR main.py ##

def cmd_preprocess(args):
    folder = Path(args.folder or DATA_DIR)
    chunks_out_path = OUTPUT_DIR / "chunks.jsonl"
    doc_map_out_path = OUTPUT_DIR / "chunks_map.pkl"
    id_map_out_path = OUTPUT_DIR / "id_mapping.json"
    
    logger.info("Starting preprocessing...")

    if args.parallel:
        # For parallel processing, we first let the workers create a temporary file
        # with the original string IDs. Then, the main process will read this file
        # and assign the final, sequential integer IDs. This is the safest way
        # to handle counters with multiprocessing.
        
        # Step 1: Parallel chunking to a temporary file
        temp_chunks_path = OUTPUT_DIR / "chunks_temp.jsonl"
        temp_chunks_path.unlink(missing_ok=True)
        workers = getattr(args, "workers", 4)
        preprocess_folder_parallel(folder, out_path=temp_chunks_path, num_workers=workers)
        
        # Step 2: Remap to sequential integer IDs
        logger.info("Remapping chunk IDs to sequential integers...")
        chunk_counter = 0
        id_to_source_map = {}
        chunks_out_path.unlink(missing_ok=True) # Ensure final output file is clean

        with open(temp_chunks_path, "r", encoding="utf-8") as infile, \
             open(chunks_out_path, "a", encoding="utf-8") as outfile:
            for line in infile:
                ch = json.loads(line.strip())
                original_id = ch["id"]
                ch["id"] = chunk_counter
                id_to_source_map[chunk_counter] = original_id
                outfile.write(json.dumps(ch, ensure_ascii=False) + "\n")
                chunk_counter += 1
        
        temp_chunks_path.unlink() # Clean up temporary file
        total_chunks = chunk_counter

    else:
        # Sequential processing (already correct from last time)
        chunk_counter = 0
        id_to_source_map = {}
        files = [p for p in sorted(folder.iterdir()) if p.is_file()]
        chunks_out_path.unlink(missing_ok=True)

        for path in tqdm(files, desc="Preprocessing files"):
            chunks_from_file = preprocess_file(path)
            if chunks_from_file:
                with open(chunks_out_path, "a", encoding="utf-8") as f:
                    for ch in chunks_from_file:
                        original_id = ch["id"]
                        ch["id"] = chunk_counter
                        id_to_source_map[chunk_counter] = original_id
                        f.write(json.dumps(ch, ensure_ascii=False) + "\n")
                        chunk_counter += 1
        total_chunks = chunk_counter
    
    # Save manifest with the total count
    (OUTPUT_DIR / "preprocessed_meta.json").write_text(json.dumps({"chunks": total_chunks}))
    logger.info(f"Successfully preprocessed {total_chunks} chunks.")

    # Create and save the chunk_id -> text map for fast retrieval
    logger.info("Creating fast lookup map for document chunks...")
    chunks_map = {}
    with open(chunks_out_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line.strip())
            chunks_map[d["id"]] = d["text"]
    
    with open(doc_map_out_path, "wb") as f:
        pickle.dump(chunks_map, f)
    logger.info(f"Saved document map to {doc_map_out_path}")

    # Save the integer ID to original source ID mapping
    with open(id_map_out_path, "w", encoding="utf-8") as f:
        json.dump(id_to_source_map, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved integer ID to source mapping at {id_map_out_path}")


def cmd_index(args):
    chunks_path = OUTPUT_DIR / "chunks.jsonl"
    meta_path = OUTPUT_DIR / "preprocessed_meta.json"

    if not chunks_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "Saved chunks or metadata not found. Please run the 'preprocess' command first."
        )

    # Load total chunk count from metadata
    total_docs = json.loads(meta_path.read_text())["chunks"]
    
    # This generator streams documents from the JSONL file one by one
    def read_chunks_generator(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line.strip())

    logger.info("Initializing embedding indexer for a large-scale build...")
    idx = EmbeddingIndexer()
    doc_gen = read_chunks_generator(chunks_path)
    
    # Call the scalable, generator-based build method
    idx.build_from_generator(doc_gen, total_docs)
    idx.save()
    logger.info("Scalable index built and saved successfully.")

def cmd_query(args):
    # Load the index and retriever
    idx = EmbeddingIndexer()
    idx.load()
    retriever = Retriever(idx)
    
    # Load the fast document map for reranking
    doc_map_path = OUTPUT_DIR / "chunks_map.pkl"
    if not doc_map_path.exists():
        raise FileNotFoundError("Document map not found. Please run 'preprocess' first.")
    
    logger.info("Loading document map for reranking...")
    with open(doc_map_path, "rb") as f:
        chunks_map = pickle.load(f)

    reranker = Reranker() if args.rerank else None
    
    # Load queries from Queries.json
    queries_path = Path(args.queries_file or DATA_DIR / "Queries.json")
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found at: {queries_path}")
    
    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    results_dir = OUTPUT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing {len(queries)} queries...")
    # MODIFIED: Change the loop to correctly handle a list of dictionaries
    for item in tqdm(queries, total=len(queries), desc="Running queries"):
        qid = item.get("id") or item.get("query_id")
        query = item.get("query") or item.get("text")

        if qid is None or query is None:
            logger.warning(f"Skipping invalid query item: {item}")
            continue

        # 1. Retrieve candidate IDs from FAISS
        cand_ids, _ = retriever.retrieve(query, k=args.candidates)
        
        # 2. Get text for candidates from our fast map
        candidates = [{"id": cid, "text": chunks_map.get(cid, "")} for cid in cand_ids]
        
        # 3. Rerank or take top K
        if reranker:
            top_ids = reranker.rerank(query, candidates, top_k=args.k)
        else:
            top_ids = [c["id"] for c in candidates[:args.k]]
            
        out = {"query": query, "response": top_ids}
        save_json(out, results_dir / f"{qid}.json")
        
    zip_dir(results_dir, OUTPUT_DIR / f"{STARTUP_NAME}_PS4.zip")
    logger.info("Query run finished and submission package created.")

    
def cmd_eval(args):
    logger.info("Starting evaluation...")
    ground_truth_path = Path(args.ground_truth_file or DATA_DIR / "Ground_truth.csv")
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground truth file not found at: {ground_truth_path}")

    # Load ground truth
    gt_df = pd.read_csv(ground_truth_path)
    ground_truth = {}
    for _, row in gt_df.iterrows():
        # The ground truth relevant_document_ids might be a string like '["id1", "id2"]'
        # We need to convert it to a set of integers
        relevant_ids_str = row["relevant_document_ids"]
        # Assuming the IDs in the CSV are the integer IDs we generated
        relevant_ids = set(json.loads(relevant_ids_str.replace("'", '"')))
        ground_truth[str(row["query_id"])] = relevant_ids
    
    # Load the results your system generated
    results_dir = OUTPUT_DIR / "results"
    if not results_dir.exists():
        raise FileNotFoundError("Results directory not found. Run the 'query' command first.")
        
    all_scores = {"precision@k": [], "recall@k": [], "ndcg@k": []}
    
    for qid, relevant_ids in ground_truth.items():
        result_file = results_dir / f"{qid}.json"
        if not result_file.exists():
            logger.warning(f"Result file for query ID {qid} not found. Skipping.")
            continue
            
        with open(result_file, "r", encoding="utf-8") as f:
            result_data = json.load(f)
        
        retrieved_ids = result_data["response"]
        k = args.k
        
        # Calculate metrics
        all_scores["precision@k"].append(precision_at_k(retrieved_ids, relevant_ids, k))
        all_scores["recall@k"].append(recall_at_k(retrieved_ids, relevant_ids, k))
        all_scores["ndcg@k"].append(ndcg_at_k(retrieved_ids, relevant_ids, k))

    # Calculate final average scores
    avg_precision = np.mean(all_scores["precision@k"])
    avg_recall = np.mean(all_scores["recall@k"])
    avg_ndcg = np.mean(all_scores["ndcg@k"])

    # Calculate the final combined score as per the PDF's Stage 1 rules
    combined_score = (avg_precision * 0.20) + (avg_recall * 0.50) + (avg_ndcg * 0.30)
    
    logger.info("--- EVALUATION RESULTS ---")
    logger.info(f"Average Precision@{args.k}: {avg_precision:.4f}")
    logger.info(f"Average Recall@{args.k}:    {avg_recall:.4f}")
    logger.info(f"Average NDCG@{args.k}:       {avg_ndcg:.4f}")
    logger.info("--------------------------")
    logger.info(f"Final Combined Score: {combined_score:.4f}")
    logger.info("--------------------------")



def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_proc = sub.add_parser("preprocess", help="Preprocess raw documents into chunks.")
    p_proc.add_argument("--folder", default=None, help="Folder with raw data files.")
    p_proc.add_argument("--parallel", action="store_true", help="Use parallel processing.")
    p_proc.add_argument("--workers", type=int, default=4, help="Number of workers for parallel processing.")

    # Index command no longer needs arguments, it relies on preprocess output
    p_idx = sub.add_parser("index", help="Build a FAISS index from preprocessed chunks.")

    p_q = sub.add_parser("query", help="Run queries against the index.")
    p_q.add_argument("--queries-file", default=None, help="Path to a JSON file with queries.") 
    p_q.add_argument("--k", type=int, default=TOP_K, help="Final number of results to return.")
    p_q.add_argument("--rerank", action="store_true", help="Enable the reranking step.")
    p_q.add_argument("--candidates", type=int, default=RETRIEVE_CANDIDATES, help="Number of candidates to fetch for reranking.")

    p_eval = sub.add_parser("eval", help="Evaluate retrieval results against ground truth.")
    p_eval.add_argument("--ground-truth-file", default=None, help="Path to the ground truth CSV file.")
    p_eval.add_argument("--k", type=int, default=TOP_K, help="K value for metrics (e.g., Precision@k).")
    

    args = parser.parse_args()
    if args.cmd == "preprocess":
        cmd_preprocess(args)
    elif args.cmd == "index":
        cmd_index(args)
    elif args.cmd == "query":
        cmd_query(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()