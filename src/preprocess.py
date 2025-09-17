import os
import re
import json
from pathlib import Path
from multiprocessing import Pool, Lock, Manager
from typing import List, Dict

from tqdm import tqdm

from config import CHUNK_SIZE_WORDS, CHUNK_OVERLAP, DATA_DIR
from logger import get_logger

logger = get_logger("preprocess")

# NEW: Add a file size limit in MB to avoid loading gigantic files
MAX_FILE_SIZE_MB = 50

# Optional PDF support
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except Exception:
    PDFPLUMBER_AVAILABLE = False


def clean_text(t: str) -> str:
    t = t.replace("\r\n", " ").replace("\n", " ")
    # MODIFIED: Replace memory-intensive regex with a more efficient method
    t = " ".join(t.split())
    return t.strip()


def read_file(path: Path) -> str:
    # NEW: First, check the file size. Skip if it's too large.
    try:
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            logger.warning(f"Skipping file {path.name} as it is too large ({file_size_mb:.2f} MB).")
            return ""
    except OSError as e:
        logger.warning(f"Could not get size of file {path.name}: {e}")
        return ""

    ext = path.suffix.lower()

    # Text / Markdown
    if ext in {".txt", ".md"}:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.warning(f"Failed to read {path}: {e}")
            return ""

    # JSON
    if ext == ".json":
        try:
            content = path.read_text(encoding="utf-8", errors="ignore").strip()
            if not content:
                # This warning is already in your code, which is good.
                # logger.warning(f"Empty JSON file: {path}")
                return ""
            data = json.loads(content)
            if isinstance(data, dict):
                for key in ("text", "content", "body"):
                    if key in data:
                        return str(data[key])
            return json.dumps(data)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in {path}: {e}")
            return ""
        except Exception as e:
            logger.warning(f"Failed to read {path}: {e}")
            return ""

    # PDF (if pdfplumber is available)
    if ext == ".pdf" and PDFPLUMBER_AVAILABLE:
        try:
            texts = []
            with pdfplumber.open(path) as pdf:
                for p in pdf.pages:
                    texts.append(p.extract_text() or "")
            return "\n".join(texts)
        except Exception as e:
            logger.warning(f"Failed to read PDF {path}: {e}")
            return ""

    # Fallback for unknown extensions
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.warning(f"Failed to read {path}: {e}")
        return ""


def chunk_text(text: str, chunk_words: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_words - overlap)
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_words])
        if chunk:
            chunks.append(chunk)
        if i + chunk_words >= len(words):
            break
    return chunks


def preprocess_folder(folder: str = None) -> List[Dict]:
    # This function remains for potential single-threaded use, but we add the safety net.
    folder = Path(folder or DATA_DIR)
    docs = []
    files = [p for p in sorted(folder.iterdir()) if p.is_file()]
    logger.info(f"Preprocessing folder: {folder} ({len(files)} files)")
    for path in tqdm(files, desc="Preprocessing files"):
        chunks = preprocess_file(path)
        if chunks:
            docs.extend(chunks)
    logger.info(f"Created {len(docs)} chunks")
    return docs


def preprocess_file(path: Path) -> List[Dict]:
    # NEW: Add a try/except block to catch MemoryError for a specific file
    try:
        raw = read_file(path)
        if not raw:
            return []
        text = clean_text(raw)
        if not text:
            return []
        chunks = chunk_text(text)
        return [{"id": f"{path.name}::chunk{idx}", "text": ch} for idx, ch in enumerate(chunks)]
    except MemoryError:
        logger.error(f"MemoryError while processing file: {path.name}. Skipping this file.")
        # Return empty list to allow other processes to continue
        return []
    except Exception as e:
        logger.warning(f"An unexpected error occurred with file {path.name}: {e}")
        return []


def worker_process_file(path_out_tuple):
    path, out_path, lock = path_out_tuple
    chunks = preprocess_file(path)
    if chunks:
        with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                for ch in chunks:
                    f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    return len(chunks)


def preprocess_folder_parallel(folder: str = None, out_path: Path = None, num_workers: int = 4) -> int:
    folder = Path(folder or DATA_DIR)
    out_path = Path(out_path)
    files = [p for p in sorted(folder.iterdir()) if p.is_file()]
    total_files = len(files)
    total_chunks = 0

    logger.info(f"Preprocessing folder: {folder} ({total_files} files) with {num_workers} workers")
    manager = Manager()
    lock = manager.Lock()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # The main script already handles cleaning this file, so we don't need to do it here.
    # out_path.touch(exist_ok=True) 

    tasks = [(p, out_path, lock) for p in files]

    with Pool(processes=num_workers) as pool:
        for processed in tqdm(pool.imap_unordered(worker_process_file, tasks), total=total_files, desc="Preprocessing"):
            total_chunks += processed
    logger.info(f"Created {total_chunks} chunks")
    return total_chunks