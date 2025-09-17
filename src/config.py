from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]

# Data + output
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
STARTUP_NAME = "Saforia"

# Preprocessing
CHUNK_SIZE_WORDS = 400
CHUNK_OVERLAP = 50

# Embeddings (quality vs speed tradeoff)
EMBED_MODEL = "sentence-transformers/paraphrase-mpnet-base-v2"  # high quality
EMBED_BATCH = 64

# Reranker (cross-encoder)
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # fast and effective
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# FAISS / Indexing
USE_IVF = True         # IVF recommended for large corpora
NLIST = 1024           # number of clusters (tune: 256-4096)
NPROBE = 32            # number of probes at search (tune for recall)
METRIC = "cosine"      # "cosine" or "l2"
TOP_K = 10             # default top-K to return (final output)
RETRIEVE_CANDIDATES = 100  # retrieve many, then rerank (e.g., 100)
PQ_M = 96
PQ_BITS = 8
NUM_TRAIN_SAMPLES = 40000

# Persistence paths
INDEX_PATH = OUTPUT_DIR / "faiss.index"
IDMAP_PATH = OUTPUT_DIR / "id_map.json"
EMB_PATH = OUTPUT_DIR / "embeddings.npy"

# Misc
SEED = 42
NUM_WORKERS = 4

