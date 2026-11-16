# Offline Wikipedia RAG — Streamlit Visualization

## IIT Bombay | Deep Learning / Generative AI Project

An interactive Streamlit application for exploring an offline
Retrieval-Augmented Generation (RAG) system built using Wikipedia,
TF-IDF retrieval, cosine similarity, and Flan-T5-small.

The visualization application is maintained separately from the
main RAG implementation.

We've built and verified:
- Interactive Wikipedia question answering.
- Configurable retrieval K (1, 3, 5).
- Retrieved versus prepared evidence inspection.
- Token-budget visualization.
- Accuracy and computational-cost dashboards.
- Detailed failure analysis with manual assessments.
- Separate UI and RAG projects connected through Python.
- UI dependencies and installation documentation.

---

## 1. Project Overview

The application provides two primary interfaces.

### 1.1 Wikipedia Question Answering

Users can:

- Enter a Wikipedia-related question.
- Select the number of retrieved passages (K = 1, 3, or 5).
- Generate an answer using Flan-T5-small.
- Inspect retrieved Wikipedia passages.
- Compare original and prepared evidence.
- Examine token counts and evidence-token budgets.
- View retrieval and generation timings.
- Identify whether inference runs on CPU or GPU.

### 1.2 Evaluation Dashboard

Users can:

- Compare Exact Match accuracy.
- Compare Token F1 scores.
- Examine the effect of retrieval K.
- Compare two evidence-fusion methods.
- Analyze generation time and input tokens.
- Compare computational cost and model calls.
- Inspect failed questions.
- Review manually assessed answers.
- Examine whether retrieved and prepared evidence
  supports a reference answer.

The dashboard reads saved evaluation results.
It does not rerun the original experiments.

---

## 2. Academic Project Results

The underlying RAG project evaluated:

- 20 questions.
- 3 retrieval configurations: K = 1, 3, and 5.
- 2 evidence-fusion methods.
- 120 individual evaluations.

The best evaluated configuration was:

| Parameter | Value |
|---|---|
| Retriever | TF-IDF |
| Similarity | Cosine similarity |
| Retrieval K | 1 |
| Evidence preparation | Prefix shortening |
| Evidence token limit | 80 tokens per passage |
| Fusion | Concatenation |
| Generator | google/flan-t5-small |
| Exact Match | 75% |
| Token F1 | Approximately 85.62% |

These findings apply to the project's selected Wikipedia corpus,
evaluation questions, and model configuration.

They should not be interpreted as universal RAG performance results.

---

## 3. Architecture

The application uses two separate project directories.

```text
code/
|
+-- Mid-Sem-Project-RAG/
|   |
|   +-- config/
|   +-- data/
|   +-- models/
|   +-- results/
|   +-- src/
|       |
|       +-- run_rag.py
|       +-- rag/
|
+-- Mid-Sem-Project-RAG-UI/
    |
    +-- app.py
    +-- pages/
    |   |
    |   +-- 1_Evaluation_Dashboard.py
    |
    +-- services/
    |   |
    |   +-- rag_service.py
    |   +-- evaluation_service.py
    |
    +-- components/
    +-- config/
    +-- assets/
    +-- requirements.txt
    +-- README.md
```

### 3.1 Question-Answering Flow

1. A user enters a question in Streamlit.
2. Streamlit invokes the RAG service.
3. The service initializes or reuses the existing RAG pipeline.
4. TF-IDF and cosine similarity retrieve relevant passages.
5. Retrieved passages are shortened to the configured token budget.
6. Prepared passages are concatenated.
7. Flan-T5-small generates an answer.
8. Streamlit displays the answer, evidence, and performance metrics.

### 3.2 Evaluation Dashboard Flow

1. Streamlit invokes the evaluation service.
2. The service reads saved CSV files from the main RAG project.
3. Pandas loads the evaluation data.
4. The service validates required columns and experiment coverage.
5. Streamlit displays accuracy, cost, and failure-analysis charts.

### 3.3 Backend Integration

The UI imports the existing Python RAG application directly.

A separate HTTP API server is not required.

This is an in-process Python integration, not a REST API connection.

A FastAPI backend may be introduced later as an optional
production-engineering improvement.

---

## 4. System Requirements

Recommended environment:

- Python 3.11
- Windows, Linux, or macOS with compatible Python dependencies
- Sufficient RAM for the model, index, and application
- CPU inference supported
- NVIDIA GPU acceleration available where PyTorch CUDA is supported

The original development environment used:

- Python 3.11.14
- Streamlit 1.66.1
- Pandas 3.0.1
- NVIDIA RTX 5060 Laptop GPU
- CUDA-enabled PyTorch

A GPU is helpful but not mandatory.

Actual compatibility depends on the platform and installed
PyTorch build.

---

## 5. Installation

### 5.1 Arrange Project Directories

Keep the UI and main RAG projects as sibling directories.

```text
code/
    Mid-Sem-Project-RAG/
    Mid-Sem-Project-RAG-UI/
```

The UI currently resolves the main RAG project using this
relative directory structure.

Changing the folder names or arrangement requires updating
the corresponding path-resolution code.

### 5.2 Create a Python Environment

Example using Conda:

```bash
conda create -n wikipedia-rag python=3.11
conda activate wikipedia-rag
```

Alternatively, use an existing compatible Python environment.

### 5.3 Install Main RAG Dependencies

First, install a PyTorch build appropriate for your operating
system and hardware.

Follow the official PyTorch installation instructions:

https://pytorch.org/get-started/locally/

Then navigate to the main project:

```bash
cd Mid-Sem-Project-RAG
```

Install its portable dependencies:

```bash
python -m pip install -r requirements-portable.txt
```

The portable requirements file should not force a Windows-specific
CUDA build onto Linux or macOS.

### 5.4 Install UI Dependencies

Navigate to the UI project:

```bash
cd ../Mid-Sem-Project-RAG-UI
```

Install:

```bash
python -m pip install -r requirements.txt
```

Verify dependency consistency:

```bash
python -m pip check
```

Expected:

```text
No broken requirements found.
```

---

## 6. Running the Application

From the UI project directory, execute:

```bash
python -m streamlit run app.py
```

Streamlit normally displays a local address such as:

http://localhost:8501

Open the address shown in the terminal.

Keep the terminal running while using the application.

The Streamlit navigation provides:

- Offline Wikipedia RAG: interactive question answering.
- Evaluation Dashboard: saved experimental results.

---

## 7. Offline Execution

The main project supports loading a locally exported
Flan-T5-small model.

Expected local model directory:

```text
Mid-Sem-Project-RAG/
    models/
        flan-t5-small/
            config.json
            generation_config.json
            model.safetensors
            tokenizer.json
            tokenizer_config.json
```

The main RAG configuration references the local model directory.

When the local model is available, the generator can load it
without downloading model weights.

If the local directory is unavailable, the generator preserves
its Hugging Face model-loading fallback behavior.

That fallback may require internet access unless the model is
already present in the target machine's cache.

### 7.1 Offline Environment Variables

For Windows PowerShell:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
```

For Linux or macOS shells:

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Then start Streamlit.

These settings prevent supported Hugging Face components
from attempting online access.

### Important

The local model, Wikipedia passages, and TF-IDF index must
already be available on disk.

Python packages must also be installed.

A self-contained model bundle does not automatically provide
offline installation of Python dependencies on a new machine.

---

## 8. Evaluation Metrics

### Exact Match

Checks whether the normalized generated answer matches the
normalized reference answer.

### Token F1

Measures token overlap between the generated and reference answers.

It considers both precision and recall.

### Retrieval Hit

Indicates whether retrieval found the relevant source article
according to the project's evaluation criteria.

### Evidence Support

Indicates whether the original or prepared passages contain
sufficient information to support the reference answer.

### Generation Time

Measures time spent generating an answer.

### Model Calls

Counts the number of generation-model invocations.

---

## 9. Understanding RAG Failures

### Example: Diana Richburg

Question:

Who is Diana Richburg?

For K = 1, the system generates:

middle-distance runner

For larger K values, an irrelevant passage about Diana Baciu
may influence the generator.

This demonstrates that retrieving additional passages can
introduce distracting information.

### Example: John Hubley

Question:

Who was John Hubley's second wife?

Reference answer:

Faith Hubley

The original retrieved evidence supported the answer.

However, the configured 80-token prefix shortening removed
the relevant information.

The model therefore answered:

Insufficient evidence

This demonstrates an evidence-preparation failure.

---

## 10. Troubleshooting

### Problem: localhost refused to connect

Start the Streamlit server:

```bash
python -m streamlit run app.py
```

Keep its terminal open.

Use the local URL printed by Streamlit.

### Problem: Main RAG project not found

Verify the sibling-directory structure.

Check that the expected configuration exists:

```text
Mid-Sem-Project-RAG/config/rag_config.json
```

### Problem: Evaluation CSV not found

Verify:

```text
Mid-Sem-Project-RAG/results/evaluation/
```

### Problem: Model cannot be loaded

Verify the local model directory and required model files.

Ensure that the configured path is correct.

### Problem: GPU unavailable

Verify the installed PyTorch build and device compatibility.

The application can use CPU inference when the generator
is configured for automatic device selection.

---

## 11. Project Boundaries

This UI project is an academic visualization and demonstration
application.

It does not currently include:

- A separate FastAPI backend
- Authentication or user accounts
- Production-scale concurrent request management
- Persistent user conversation history
- Hosted deployment
- Production security hardening

These are possible future engineering improvements.

---

## 12. Project Context

IIT Bombay — ePGD in Artificial Intelligence and Data Science

Deep Learning / Generative AI Academic Project

Application: Offline Wikipedia Retrieval-Augmented Generation

UI Framework: Streamlit

Retrieval: TF-IDF with cosine similarity

Generation: Flan-T5-small