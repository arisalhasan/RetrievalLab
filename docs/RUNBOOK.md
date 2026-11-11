# Run RetrievalLab locally

Use Python 3.12. Run commands from the folder containing `app.py` and `pyproject.toml`. On this reference computer the environment and model assets are already prepared. No accounts, paid API, database service or GPU are required.

## Already prepared project

Start the dashboard:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open <http://127.0.0.1:8501>. In another terminal start the API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn retrieval_lab.api:app --host 127.0.0.1 --port 8000 --workers 1
```

Open <http://127.0.0.1:8000/docs>. If either server is already running, use its existing page. Stop a server with Ctrl+C in its terminal. Keep one API worker because each worker loads its own models. BM25 works without preparing the neural models.

Streamlit reruns the page after some selections and downloads. If it returns to **Search evidence**, reopen the analysis tab; the selected filter is retained. After changing imported source modules during development, restart Streamlit so its Python process uses the updated verifier. The shipped release code and saved experiments are already aligned.

## Fresh Windows CPU setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-release.lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m retrieval_lab.cli download
.\.venv\Scripts\python.exe -m retrieval_lab.cli verify-split
.\.venv\Scripts\python.exe -m retrieval_lab.cli model-download
.\.venv\Scripts\python.exe -m retrieval_lab.cli build-dense
.\.venv\Scripts\python.exe -m retrieval_lab.cli reranker-download
```

The first commands need internet for free public packages/data/weights. Downloaded archives and model files are checked against pinned identities. This is a Windows x86-64/Python 3.12 reference lock with a CPU PyTorch wheel. Other operating systems need their corresponding CPU wheel and have not been verified here. If the Python launcher is unavailable, use an installed Python 3.12 executable to create the environment; no activation or PowerShell policy change is needed.

The full release bundle includes both trained safetensors checkpoints and their manifests. A source-only copy needs to restore these artifacts or run the documented training experiment before choosing **Fine-tuned**. The dense index encodes all corpus documents once; allow several minutes for this initial preparation. Normal requests never download models or build embeddings. Allow several GB for the environment, model assets, experiment evidence and caches.

## Tests and checks

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pytest --basetemp runs/pytest-local
```

The ordinary suite is offline and uses small synthetic fixtures. After official data and pretrained assets are prepared, opt into real integration checks:

```powershell
$env:RUN_SCIFACT_TESTS='1'
$env:RUN_DENSE_MODEL_TESTS='1'
$env:RUN_RERANKER_MODEL_TESTS='1'
.\.venv\Scripts\python.exe -m pytest --basetemp runs/pytest-local-real
```

Real-model tests include a tiny synthetic training/reload check. That is separate from the recorded full SciFact training experiment. Run `scripts/smoke_api.py --output runs/my-api-smoke.json` through the same Python executable to check a running API. Use new output names; saved experiment and release evidence is not overwritten.

## Experiments and final results

For baseline/dense/hybrid/pretrained recipes, use the earlier milestone guides and `scripts/run_milestone3.py`, `run_milestone4.py`, and `run_milestone5.py`. A fresh train-and-validation study is:

```powershell
.\.venv\Scripts\python.exe -u scripts/run_milestone7.py --output runs/my-training-study
```

Read the compute plan and leakage rules in [milestone 7](MILESTONE_7.md) first. The original full experiment took about 16 minutes plus a separate full validation benchmark. A new run does not replace the frozen release configuration automatically.

For held-out results, follow [milestone 8](MILESTONE_8.md): verify all development evidence, freeze settings **before** test-label parsing, then explicitly evaluate test. The shipped original freeze verifies exact code/config/model/data/runtime/hardware fingerprints. Different machines intentionally cannot impersonate the original environment; document a new environment-specific reproduction. Since the original test is now reported, subsequent work must disclose that it has been seen.

`python -m retrieval_lab.final_report` exports a completed frozen test study. Use a new `--output` directory if the original export exists. `python scripts/package_release.py` builds the full local evidence bundle after the final test and wheel exist; it preserves an existing release ZIP. Neither command publishes files externally.

## Understand and demonstrate

Read [the architecture](ARCHITECTURE.md), [model/data card](MODEL_CARD.md), [execution record](VERIFICATION_RELEASE.md), [training results](results/TRAINING_V1.md), and [official test report](results/OFFICIAL_TEST_V1.md). Use [the three-minute demo](DEMO.md). Every milestone guide has a student verification checkpoint and three interview questions. Verify the saved numbers and the chosen model before quoting them in an application.
