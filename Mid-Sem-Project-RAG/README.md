# Offline Wikipedia Retrieval-Augmented Generation (RAG)

## IIT Bombay — Deep Learning / Generative AI Project

### Project Overview

This project implements an offline Retrieval-Augmented Generation (RAG) system using a locally stored Wikipedia corpus.

The application retrieves relevant Wikipedia passages using TF-IDF and cosine similarity, then generates grounded answers using Google's Flan-T5-small model.

The system supports local inference without requiring a hosted language-model API.

## 1. Project Objectives

The objectives are to:

1. Construct a small Wikipedia-based knowledge corpus.
2. Split documents into overlapping passages.
3. Build a TF-IDF retrieval index.
4. Retrieve relevant passages for user questions.
5. Generate answers using Flan-T5-small.
6. Compare retrieval sizes K=1, 3 and 5.
7. Compare concatenation and generate-from-evidence fusion.
8. Evaluate answer quality, retrieval performance and cost.
9. Provide a reproducible standalone application.

## 2. Dataset

**Source:** Wikimedia Wikipedia

**Dataset:** `wikimedia/wikipedia`

**Configuration:** `20231101.en`

A subset of 1,000 Wikipedia articles was selected and stored locally.

The articles were divided into overlapping passages.

| Parameter | Value |
|---|---|
| Wikipedia articles | 1,000 |
| Indexed passages | 8,191 |
| Chunk size | 100 words |
| Chunk overlap | 20 words |
| TF-IDF vocabulary size | 30,000 |

The source corpus and retrieval index are stored in the project's `data/` directory.

## 3. RAG Architecture

The system follows these steps:

1. Accept a user question.
2. Transform the question using the fitted TF-IDF vectorizer.
3. Calculate cosine similarity against indexed passages.
4. Retrieve the highest-ranked passages.
5. Prepare evidence within the generator's token budget.
6. Generate an answer using Flan-T5-small.
7. Return the answer with source attribution.

### Core Technologies

| Component | Technology |
|---|---|
| Programming language | Python 3.11 |
| Dataset processing | Hugging Face Datasets |
| Indexing | scikit-learn TF-IDF |
| Similarity | Cosine similarity |
| Generator | google/flan-t5-small |
| Deep learning framework | PyTorch |
| GPU acceleration | NVIDIA CUDA |
| Evaluation | Exact Match, Token F1 and retrieval metrics |

## 4. Final RAG Configuration

The selected configuration is stored in:

`config/rag_config.json`

| Setting | Value |
|---|---|
| Retrieval method | TF-IDF |
| Retrieval size | K=1 |
| Evidence token limit | 80 |
| Fusion strategy | Concatenation |
| Generator | Flan-T5-small |
| Maximum input tokens | 512 |
| Maximum generated tokens | 64 |

This configuration was selected based on the evaluation experiments described below.

## 5. Project Structure

```text
Mid-Sem-Project-RAG/
├── config/
│   └── rag_config.json
├── data/
│   ├── wiki_slice_1000/
│   ├── wiki_chunks_100w_20overlap/
│   ├── tfidf_index/
│   └── evaluation/
├── notebooks/
├── results/
│   └── evaluation/
├── src/
│   ├── rag/
│   │   ├── data_loader.py
│   │   ├── chunking.py
│   │   ├── indexer.py
│   │   ├── retriever.py
│   │   ├── evidence.py
│   │   ├── fusion.py
│   │   ├── generator.py
│   │   ├── evaluation.py
│   │   └── pipeline.py
│   └── run_rag.py
├── tests/
├── requirements.txt
└── README.md
```

## 6. Environment

The project was developed and tested using:

- Windows 11-compatible environment
- Python 3.11.14
- PyTorch 2.10.0+cu128
- Transformers 5.21.0
- scikit-learn 1.8.0
- CUDA 12.8
- NVIDIA GeForce RTX 5060 Laptop GPU

See `requirements.txt` for additional dependencies.

### Installation

Create a compatible Python 3.11 environment.

For the tested NVIDIA CUDA 12.8 setup, install the corresponding PyTorch build:

```powershell
python -m pip install torch==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

Then install the remaining project dependencies:

```powershell
python -m pip install -r requirements.txt
```

For CPU-only systems, a compatible CPU PyTorch build and adjusted dependency specification may be required.

## 7. Running the Application

Open a terminal in the project root and activate the project's Python environment.

Execute:

```powershell
python src/run_rag.py --question "Who is Diana Richburg?"
```

Example output:

```text
Question: Who is Diana Richburg?
Answer: middle-distance runner

Article: Diana Richburg
Similarity: 0.3932
Device: cuda
```

The application also reports retrieval time, generation time, input tokens and generated tokens.

## 8. Offline Execution

- The Wikipedia passage dataset and TF-IDF index are saved locally.
- Flan-T5-small must be downloaded and cached before offline use.
- To test offline execution in PowerShell:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

python src/run_rag.py --question "Who is Diana Richburg?"
```

The application was verified to run successfully with these offline-mode settings enabled.

The application does not require a hosted language-model API.

## 9. Evaluation Methodology

An evaluation dataset containing 20 manually prepared question-answer pairs was constructed.

The questions covered 14 Wikipedia articles.

Two fusion strategies were evaluated:

1. Concatenation
2. Generate-from-evidence

Three retrieval sizes were tested:

- K=1
- K=3
- K=5

This produced 120 evaluations and 300 model-generation calls.

### Evaluation Metrics

- Exact Match
- Token F1
- Retrieval Hit@K
- Evidence Support@K
- Abstention rate
- Generation time
- Token consumption
- Model-call count

## 10. Experimental Results

### Accuracy Comparison

| Method | K | Exact Match | Token F1 |
|---|---:|---:|---:|
| Concatenation | 1 | 75% | 85.62% |
| Concatenation | 3 | 70% | 78.88% |
| Concatenation | 5 | 65% | 73.88% |
| Generate-from-evidence | 1 | 50% | 57.54% |
| Generate-from-evidence | 3 | 35% | 43.66% |
| Generate-from-evidence | 5 | 20% | 24.38% |

### Main Findings

1. Concatenation achieved higher accuracy than generate-from-evidence for every tested K.
2. Increasing K did not improve Exact Match in this evaluation.
3. Irrelevant passages sometimes caused answer contamination.
4. Prefix-based evidence shortening occasionally removed answer-bearing text.
5. Generate-from-evidence required more model calls and achieved lower accuracy.

### Selected Configuration

**Concatenation Fusion with K=1**

This achieved:

- Exact Match: 75%
- Mean Token F1: 85.62%

A preliminary manual review identified two additional factually correct answers that did not exactly match their references.

The resulting preliminary manual factual-correctness estimate was 85%, separate from automated Exact Match.

## 11. Evaluation Artifacts

Evaluation outputs are available under:

`results/evaluation/`

Important files include:

- `rag_full_evaluation.jsonl`
- `rag_full_evaluation.csv`
- `rag_evaluation_summary.csv`
- `rag_fusion_comparison.csv`
- `rag_manual_assessment.csv`

The 20-question evaluation dataset is available under:

`data/evaluation/`

## 12. Limitations

1. The evaluation dataset contains only 20 questions.
2. The questions were constructed from the indexed corpus.
3. Some examples informed development decisions.
4. The results may not generalize to arbitrary Wikipedia questions.
5. TF-IDF relies on lexical similarity rather than semantic embeddings.
6. Evidence shortening can discard relevant information.
7. Flan-T5-small can generate incomplete or incorrect answers.
8. Increasing K can introduce irrelevant evidence.
9. Exact Match may penalize factually correct paraphrases.
10. Evidence Support@K uses lexical matching rather than semantic entailment.

## 13. Future Improvements

Possible extensions include:

- Dense embedding retrieval
- Hybrid lexical and semantic retrieval
- Reranking retrieved passages
- Query-aware evidence compression
- Improved source-aware fusion
- Groundedness evaluation
- Larger and independently prepared evaluation datasets
- A separate user interface for querying and visualizing results

These are potential enhancements rather than requirements for the current baseline.

## 14. Conclusion

This project demonstrates a functional offline Wikipedia RAG system integrating TF-IDF retrieval and Flan-T5-small generation.

A systematic evaluation of retrieval size and fusion strategy identified Concatenation Fusion with K=1 as the best-performing configuration on the selected evaluation dataset.

The experiments also demonstrated that retrieving additional passages does not necessarily improve generated-answer quality.

The final application supports configuration-driven execution, local artifact loading, source attribution, and offline inference.

### Platform Compatibility

The application uses platform-independent Python paths and common cross-platform libraries.

It has been tested on Windows with Python 3.11 and an NVIDIA CUDA-enabled GPU.

Linux, macOS and CPU-only execution are expected to be possible with compatible PyTorch installations, but these platforms have not been independently tested.

The pretrained Flan-T5-small model must be downloaded or cached before offline inference.

PyTorch installation instructions may differ depending on the operating system and available accelerator hardware.


## Cross-Platform Installation

This project supports CPU inference and can use CUDA when a compatible NVIDIA GPU and PyTorch installation are available.

### Prerequisites

- Python 3.11 recommended
- A supported version of PyTorch
- Dependencies listed in requirements-portable.txt
- Saved Wikipedia passages and TF-IDF index
- Bundled Flan-T5-small model files

### Step 1: Create a Python environment

Windows PowerShell:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

Linux/macOS:

    python3 -m venv .venv
    source .venv/bin/activate

### Step 2: Install PyTorch

Select the appropriate operating system, package manager, and compute platform at:

https://pytorch.org/get-started/locally/

- For CPU-only computers, select the CPU installation option.
- For supported NVIDIA GPUs, select the appropriate CUDA build.
- Do not assume the CUDA build used during development is compatible with every computer.

### Step 3: Install remaining dependencies

    python -m pip install -r requirements-portable.txt

### Step 4: Verify the installation

    python -c "import torch; print(torch.__version__)"
    python -c "import transformers; print(transformers.__version__)"

### Step 5: Run the application

Windows PowerShell:

    $env:HF_HUB_OFFLINE = "1"
    $env:HF_DATASETS_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"

    python src/run_rag.py --question "Who is Diana Richburg?"

Linux/macOS:

    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1

    python src/run_rag.py --question "Who is Diana Richburg?"

### Local Model Loading

The application first checks the configured local model directory:

    models/flan-t5-small

If a complete local model exists, it loads the model without downloading weights.

If the directory does not exist, the application falls back to the configured Hugging Face model identifier.

The fallback requires the model to be available in the Hugging Face cache or downloadable when network access is permitted.

### Reproducibility Note

The project was developed and evaluated with Python 3.11.14, PyTorch 2.10.0+cu128, and an NVIDIA RTX 5060 Laptop GPU.

The original reported results were obtained using this environment.

Other supported systems may execute the application successfully, but identical performance and latency are not guaranteed.