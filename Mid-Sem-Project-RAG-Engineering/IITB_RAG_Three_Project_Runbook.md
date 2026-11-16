# IIT Bombay Offline Wikipedia RAG — Three-Project Runbook

> Operational reference, 11 November 2026. Use Windows PowerShell inside VS Code. This document records verified project facts and distinguishes them from commands requiring first-run confirmation. It does not replace the academic reports or the detailed learning handbook.

## 1. The three folders

All three are siblings inside `...\DeepLearning-Prof-Biplab\code`.

| Folder | Role | Identified entrypoint | Key artifacts |
| --- | --- | --- | --- |
| `Mid-Sem-Project-RAG` | Original academic RAG; TF-IDF retrieval and Flan-T5 generation | `src\run_rag.py` | 8,191 Wikipedia passages; TF-IDF index; local Flan-T5 model; evaluation files |
| `Mid-Sem-Project-RAG-UI` | Academic Streamlit visualization | `app.py` | UI services and evaluation displays |
| `Mid-Sem-Project-RAG-Engineering` | Independent FastAPI, MiniLM/ChromaDB, evaluation, logging, tests and security | `api\main.py` | `data\chroma_minilm`; `results\evaluation`; test suite |

The Engineering adapter reads/reuses the academic RAG code and original data. The original academic source and UI remain separate; do not casually edit them to configure Engineering features. All three currently use the same Conda environment, `.cvenv`.

## 2. Prerequisites and verification

Verified on the development laptop: Windows; Python 3.11.14; NVIDIA GeForce RTX 5060 Laptop GPU; PyTorch `2.10.0+cu128` (CUDA runtime 12.8); `transformers` 5.21.0; `sentence-transformers` 6.2.0; `chromadb` 1.5.9; `scikit-learn` 1.8.0; NumPy 2.1.3.

Open a **new PowerShell terminal** in VS Code:

```powershell
conda activate .cvenv
python --version
python -m pip check
python -c "import torch; print('Torch:',torch.__version__); print('CUDA:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected: Python `3.11.14`, no broken requirements, CUDA `True`, and the RTX GPU name on the verified machine. For another computer, CPU mode may be supported, but benchmark times will differ.

Each folder has its own `requirements.txt`. **Do not merge all requirements or blindly run `pip freeze` into every file.** PyTorch's CUDA wheel may need its specific installation index on a fresh machine. Review version constraints and installation instructions before changing the shared `.cvenv`.

```powershell
python -m pip show torch transformers sentence-transformers chromadb fastapi streamlit
```

If `streamlit` is not installed, check the UI's `requirements.txt` before installing. Keep dependency snapshots when making changes.

## 3. Open each project reliably

From the Engineering folder:

```powershell
$codeRoot = (Get-Item ..).FullName
$ragRoot = Join-Path $codeRoot "Mid-Sem-Project-RAG"
$uiRoot = Join-Path $codeRoot "Mid-Sem-Project-RAG-UI"
$engineeringRoot = Join-Path $codeRoot "Mid-Sem-Project-RAG-Engineering"
@($ragRoot, $uiRoot, $engineeringRoot) | ForEach-Object { Write-Host $_ (Test-Path $_ -PathType Container) }
```

All three should report `True`. Use a **separate terminal per running app**. Activate `.cvenv` in each.

## 4. Original academic RAG (`Mid-Sem-Project-RAG`)

```powershell
Set-Location $ragRoot
python --version
python -m py_compile .\src\run_rag.py
```

Verified code entrypoint: `src\run_rag.py`. It contains `load_config` and `initialize_rag`; initialization loads the dataset, saved TF-IDF index and local Flan-T5 model.

**Launch command requires confirmation from the original README or the script's `main()` guard and argument parser.** Do not assume an arbitrary CLI flag is supported. Inspect its documented invocation with:

```powershell
Select-String -Path .\README.md -Pattern 'python|run_rag|config|jupyter' -Context 1,1
Get-Content .\src\run_rag.py -Tail 65
```

Then use the command documented by that project. Its academic notebooks are also at the project root (`Project-Offline-RAG-Wikipedia.ipynb` and the submission variant). Opening a notebook is not the same as executing a CLI entrypoint.

**Expected initialization evidence (previously observed):** `Passages loaded: 8191`; `TF-IDF matrix: (8191, 30000)`; local `models\flan-t5-small`; CUDA device where available. Example question: *Who is Diana Richburg?* → `middle-distance runner` at retrieval K=1.

The original dataset is at `data\wiki_chunks_100w_20overlap`, and saved TF-IDF artifacts are under `data\tfidf_index`. Neither should be rebuilt as part of a routine launch.

## 5. Streamlit UI (`Mid-Sem-Project-RAG-UI`)

```powershell
Set-Location $uiRoot
python -m pip show streamlit
python -m streamlit run .\app.py
```

This is the conventional launch form for the discovered `app.py` Streamlit entrypoint; **verify any project-specific environment variables or working-directory assumptions from `README.md` / `README_RAG_UI_Consolidated.md`**. Streamlit normally prints a local browser URL (often `http://localhost:8501`). Use the URL actually printed in your terminal. Stop with **Ctrl+C**.

The UI is the academic visualization interface. Do not assume it automatically calls the Engineering FastAPI server: that integration has not been verified. Its `services\evaluation_service.py` reads original evaluation data.

## 6. Engineering FastAPI (`Mid-Sem-Project-RAG-Engineering`)

In a dedicated terminal:

```powershell
Set-Location $engineeringRoot
conda activate .cvenv
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Wait for the server startup message. In a **second terminal**, test:

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/health"
$body = @{ question = "Who is Diana Richburg?"; top_k = 1 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/answer" -ContentType "application/json" -Body $body
```

Expected: health `status=healthy`; `/answer` responds with the question, non-empty answer, `top_k=1`, and device/timing data. The first answer request may take several seconds to load the pipeline; later requests reuse its cached instance. To stop Uvicorn, press **Ctrl+C** in its own terminal.

**Authentication caveat:** the C15 middleware only requires `X-API-Key` when `RAG_API_KEY` was configured before server startup. If configured, include `-Headers @{ 'X-API-Key' = '<your-secret>' }` when calling `/answer` (replace the placeholder locally; do not record the secret in README, source or screenshots). If not configured, the endpoint is unauthenticated local-development mode. Keep Uvicorn bound to `127.0.0.1`.

Example negative validation test:

```powershell
@'
from fastapi.testclient import TestClient
from api.main import app
with TestClient(app) as client:
    for body in [{"question":""},{"top_k":1},{"question":"Hello","top_k":True}]:
        response = client.post('/answer',json=body)
        print(response.status_code, body)
'@ | python -
```

Expected: all three HTTP `422`. The test client may display a Starlette `httpx` deprecation warning in this environment; it did not prevent the verified suite from passing.

## 7. Semantic retrieval and ChromaDB

The Engineeering project uses `sentence-transformers/all-MiniLM-L6-v2`, 384-dimensional normalized passage embeddings, and a persistent ChromaDB collection named `wikipedia_minilm_384`, stored under `data\chroma_minilm`.

**Routine use:** read the existing ChromaDB index; do not rebuild it every time. It has already been fully indexed and audited: 8,191/8,191 records; IDs, text, metadata, finite values, dimensions and norms verified.

Only if the index must be built for the first time, after verifying the dataset and available disk space:

```powershell
Set-Location $engineeringRoot
python -m retrieval.build_chroma_index --limit 10
python -m retrieval.build_chroma_index --batch-size 32
```

`--limit 10` is a partial smoke run; the second command indexes the full corpus, skipping already present IDs. Building may load the embedding model and use CUDA, download cached weights if missing, and take time. **Never delete or overwrite the existing persistent index as a routine test.**

Known corpus limitation: 32 of 8,191 chunks (0.39%) exceed MiniLM's 256-token maximum and may be truncated during embedding.

## 8. Essential tests and expected outputs

Run from the **Engineering** root with `.cvenv` active:

| Purpose | Command | Verified expectation |
| --- | --- | --- |
| Python syntax | `python -m py_compile .\api\main.py .\api\observability.py .\api\security.py` | No traceback or output |
| API regression | `python -m unittest discover -s tests -p "test_*.py" -v` | `Ran 12 tests`, `OK` |
| Dependencies | `python -m pip check` | `No broken requirements found.` |
| MiniLM smoke test | `python .\scripts\embedding_smoke_test.py` | 384-dimensional vectors, normalized, all checks pass |
| Token coverage | `python .\scripts\embedding_token_inspection.py` | 8,191 passages; 32 over 256-token limit |
| Retrieval comparison | `python -m evaluation.retrieval_comparison --limit 3` | Both retrievers and K=1,3,5 evaluated |
| Full retrieval comparison | `python -m evaluation.retrieval_comparison` | 20 questions; read-only |
| Answer comparison smoke test | `python -m evaluation.answer_comparison --limit 3` | 3 questions × 2 methods × 3 K = 18 records |
| Full answer comparison | `python -m evaluation.answer_comparison` | 120 evaluations, potentially expensive; this module prints results and is not assumed to rewrite saved CSVs |

The archived full evaluation artifacts are:

```text
results/evaluation/c11_answer_comparison_full.csv       (120 records)
results/evaluation/c11_answer_comparison_summary.csv    (6 aggregate rows)
```

Use them for later analysis without rerunning the generator:

```powershell
@'
import pandas as pd
from pathlib import Path
folder = Path('results/evaluation')
full = pd.read_csv(folder / 'c11_answer_comparison_full.csv')
summary = pd.read_csv(folder / 'c11_answer_comparison_summary.csv')
print('Full evaluations:',len(full))
print('Summary configurations:',len(summary))
print(summary.to_string(index=False))
'@ | python -
```

## 9. Verified experimental outcomes

| Retriever | K | Exact Match | Token F1 | Designated gold-chunk hit |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF | 1 | 75% | 85.625% | 95% |
| MiniLM | 1 | 75% | 85.625% | 90% |
| TF-IDF | 3 | 70% | 78.875% | 95% |
| MiniLM | 3 | 70% | 82.625% | 90% |
| TF-IDF | 5 | 65% | 73.875% | 95% |
| MiniLM | 5 | 70% | 83.958% | 90% |

The 20 questions are a small, partially correlated benchmark. Gold-chunk hit assesses a single annotated chunk and is not exhaustive recall. Exact Match and token F1 are lexical measures, not proof of factual accuracy.

The Q009 investigation found a concrete **evidence truncation** failure: a 145-token answer-bearing retrieved passage was shortened to 80 Flan-T5 tokens and lost the album title *Now Is Not Forever*. The retained fraction is `80 / 145 ≈ 55.17%`. This does not mean that the semantic retriever failed to find the answer.

## 10. Observability, performance and security

Observability returns `X-Request-ID` and `X-Process-Time` response headers and emits JSON logs without question or answer bodies. The local benchmark measured: cold `/answer` 6.046 s; warm mean 103.526 ms (5 requests); health mean 2.157 ms (20 requests); warm P95 123.272 ms; 242.84 health requests/s in a tiny 2-worker test. These are descriptive **in-process** figures, not a concurrent GPU inference capacity estimate.

The security baseline includes optional shared API-key authentication for `/answer`, `X-Content-Type-Options`, `X-Frame-Options` and `Cache-Control: no-store`. Passing C15 does **not** imply public-deployment readiness. HTTPS, rate limits, quotas, authentication/authorization design, secret management, backups and a security review remain outside the verified local scope.

## 11. Troubleshooting

| Symptom | Check | Next action |
| --- | --- | --- |
| `ModuleNotFoundError: rag` | Original `src` is not on `sys.path` | Invoke through Engineering adapter; do not import `rag` before adapter initialization |
| CUDA unavailable | `python -c "import torch; print(torch.cuda.is_available())"` | Check `.cvenv`, CUDA PyTorch build and NVIDIA driver |
| HTTP connection refused | Uvicorn not listening on expected port | Start server in separate terminal, check port 8000 |
| HTTP 422 | Request schema invalid | Supply a nonempty question and allowed integer `top_k` 1,3,5 (not `true`) |
| HTTP 401 | API key required | Supply configured `X-API-Key`; never print or commit secret |
| First request slow | Model/index lazily initialized | Compare warm requests; avoid repeated process restarts |
| Poor answer with higher K | Irrelevant passages or evidence truncation | Inspect sources, prepared sources and K=1 results |
| HF Hub warning | Cached model missing / unauthenticated requests | Check local model cache and network policy; do not expose tokens |
| Windows symlink warning | Hugging Face cache uses fallback copy behavior | Cache can still work with greater disk usage |
| TestClient deprecation warning | Installed Starlette/httpx interface | Record warning, plan deliberate compatible dependency migration; avoid ad-hoc upgrades in shared `.cvenv` |

## 12. Quick daily sequence

1. Open the relevant folder in VS Code; activate `.cvenv`.
2. Run `python --version` and `python -m pip check`.
3. Choose **one** app: academic CLI/notebook, Streamlit UI, or Engineering API.
4. For Engineering, run Uvicorn in one terminal and `/health` from another.
5. Run `python -m unittest discover -s tests -p "test_*.py" -v` before changing code.
6. Reuse existing model/index/evaluation artifacts; rebuild only for a deliberate experiment.
7. Stop the foreground server with **Ctrl+C**.

## 13. Still to confirm from repository source

The original academic `run_rag.py` exact CLI arguments, the UI's startup/environment configuration, and any UI-to-Engineering API integration were **not established by the file inventory alone**. Consult their existing READMEs and entrypoint source before converting this runbook into a final authoritative repository README. Do not guess their interfaces.

This document is a cross-project operational companion. Keep existing academic report and assignment documentation intact.



# Engineering Roadmap - Final Completion Report

## Project

**IIT Bombay - Deep Learning / Generative AI**

### Offline Wikipedia Retrieval-Augmented Generation (RAG)

---

## Architecture

Three separate projects are maintained within the overall solution:

### 1. Mid-Sem-Project-RAG
Original academic RAG implementation.

### 2. Mid-Sem-Project-RAG-UI
Academic visualization and evaluation interface.

### 3. Mid-Sem-Project-RAG-Engineering
Independent engineering extension containing production-oriented enhancements.

> The academic projects were preserved unchanged while the Engineering project introduced additional capabilities and verification workflows.

---

## Completed Engineering Stages

| Stage | Description | Status |
|---------|-------------|----------|
| C10 | FastAPI Architecture | ✅ Complete |
| C11 | Semantic Retrieval & ChromaDB | ✅ Complete |
| C12 | Observability | ✅ Complete |
| C13 | Performance Benchmarking | ✅ Complete |
| C14 | Regression Testing | ✅ Complete |
| C15 | Security Baseline | ✅ Complete |
| C16 | Consolidated Audit | ✅ Passed |

---

## Key Technical Achievements

### Retrieval Layer
- Preserved original TF-IDF retrieval pipeline
- Implemented MiniLM semantic retrieval
- Built persistent ChromaDB vector index
- Indexed **8,191 Wikipedia passages**
- Generated **384-dimensional embeddings**

### Generation Layer
- Shared Flan-T5 answer generation pipeline

### Evaluation & Validation
- Conducted **120 end-to-end answer evaluations**
- Implemented automated regression testing
- Established latency benchmarking framework

### Observability
- Structured request logging
- Request correlation identifiers
- Execution traceability support

### Security
- Optional API-key authentication
- Security-related HTTP headers
- Baseline security hardening controls

---

## Final Audit Results

### Audit Summary

| Check | Result |
|---------|---------|
| Audit Checks | ✅ 18 / 18 Passed |
| Regression Tests | ✅ 12 / 12 Passed |
| Dependency Verification | ✅ No broken requirements found |
| Original Academic Projects | ✅ Present |
| Engineering Components | ✅ Present |
| Evaluation Artifacts | ✅ Verified |

### Final Status