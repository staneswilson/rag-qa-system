# RAG System - Quickstart Guide

This guide provides the essential commands to set up and run the RAG pipeline.

---
## Setup

1.  **Create Environment** (Recommended):
    ```bash
    conda create -n rag python=3.10
    conda activate rag
    ```

2.  **Install Dependencies**:
    Create a `requirements.txt` file with the content below and run the install command.
    ```
    # requirements.txt
    torch
    torchvision
    torchaudio
    sentence-transformers
    faiss-gpu
    transformers
    pandas
    tqdm
    numpy
    pdfplumber
    ```
    ```bash
    pip install -r requirements.txt
    ```

---
## Running the Pipeline

Before running, place your `train_set` folder, `Queries.json`, and `Ground_truth.csv` inside the `data/` directory.

### Step 1: Preprocess Data

Generate document chunks from the raw files in your `train_set`.
```bash
python src/main.py preprocess --folder data/train_set --parallel --workers 9
```

### Step 2: Build the Index
Create the Faiss vector index from the preprocessed chunks.
```bash
python src/main.py index
```

### Step 3: Generate Submission File
Run the official queries against the index to create the final results and the submission zip file.
```bash
python src/main.py query --rerank
```

### Step 4: (Optional) Evaluate Performance
Score your retrieval results locally using the ground truth file.
```bash
python src/main.py eval
```