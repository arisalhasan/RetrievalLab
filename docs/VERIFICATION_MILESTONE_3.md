# Milestone 3 execution record

Reference environment: **7 November 2026, Windows x86-64, Python 3.12.14, RetrievalLab 0.3.0, CPU only**. Exact installed packages are recorded in `requirements-dense.lock.txt` and each experiment manifest. Milestone 1 and 2 records/runs are preserved.

## Checks actually executed

- Existing milestone 2 baseline: 81 tests passed after a temporary-directory retry.
- Incremental dense config, vector/cache, model-download, safe directory installation, shared evaluator and dashboard checks ran as implementation progressed. The first full milestone 3 suite passed 105 tests, with two explicit official-data skips.
- Full suite with official training-data checks enabled: **111 passed, one explicit real-model skip** (`runs/m3-tests-data.log`). The separate real pinned-model check then passed (`runs/m3-tests-real-model.log`). It exercised real tokenization, PyTorch embeddings, unit norms, semantic toy search, FAISS/NumPy agreement and verified cache reuse. Synthetic encoders used in unit/integration tests are clearly named and never reported as benchmark measurements.
- Ruff lint and formatting passed for 39 Python files. Editable package 0.3.0 installed, `pip check` reported no broken requirements, and the distributable 0.3.0 wheel built successfully.
- Pinned public MiniLM files downloaded and verified against exact upstream revision/blob hashes and the declared weights SHA-256. Actual architecture: Transformer → mean Pooling → Normalize. CPU PyTorch 2.9.0+cpu, Sentence Transformers 5.1.2 and FAISS 1.12.0 imported successfully.
- Integration checks deliberately replaced official-test-shaped fixture labels with invalid text. BM25 evaluation, dense evaluation and their paired comparison still completed through the real shared runner. No official test labels were parsed for model selection/evaluation.

## Operational issues and corrections

The restricted execution profile initially failed a native FAISS DLL import. Normal Windows-profile imports and native tests passed, so model jobs run in that profile. No authentication, certificate validation, isolation or endpoint protections were disabled.

An initial whole-file model transfer was interrupted after stalling, and a native curl attempt timed out. Bounded TLS-verified byte-range downloads completed; each range checks its exact response offset/length, followed by upstream checksums for the entire weights file. Unit tests reject a server that ignores the Range request. Interrupted staging files are ignored, never loaded as models and excluded from the source artifact.

One test run overlapped an in-progress package installation and encountered incomplete distribution metadata. Environment mutations and tests were subsequently serialized and checks passed. Temporary Windows directory locks were addressed with narrowly bounded retries on atomic installation, with a regression test. There are no permission changes or silent overwrites.

The old long-running Streamlit process retained an imported milestone 2 module, producing an import error after a page reload. Restarting the local server loaded milestone 3 correctly; BM25 live search worked. Third-party UI initialization notices and a Transformers `torch_dtype` deprecation message were observed; neither substitutes for a test result.

## Boundaries and checks not executed

- Official SciFact test evaluation, model tuning on test labels, reranker training, GPU inference, hybrid fusion and FastAPI are outside this milestone and were not executed.
- macOS/Linux, fresh-machine installation from the full dense lock, a million-document scale test and a pretraining-data overlap audit were not executed. The lock records this working Windows environment; a wheel build and compatibility check do not prove a clean-room install.
- Live narrow-screen/mobile layout was not retested in this milestone. Desktop browser checks used a 1280 × 720 viewport; no viewport override was applied.
- The full BEIR neural API was not run. Recall/nDCG were independently checked with BEIR's `pytrec_eval` backend, and vector search with NumPy/scikit-learn.
- Quality estimates concern the fixed train-derived validation partition, not an untouched official test. Grouping is a leakage-reduction heuristic. Workstation load and CPU power state are uncontrolled; RSS is sampled whole-process RAM and can miss brief peaks. No per-model isolated memory, GPU usage, energy or causal speed claim is made.
- Git metadata is unavailable in this project mirror; data/config/model/source hashes and package snapshots provide provenance instead.

## One-time document preparation

The actual pinned encoder prepared all **5,183 × 384** document vectors. Corpus encoding took **465.59 seconds**; the complete preparation stage took **481.31 seconds** with **480.69 process CPU seconds**. Sampled whole-process peak RSS was **1,188.15 MiB**. The vector array occupies **7,961,088 bytes** (7.59 MiB), excluding the model, Python, temporary tensors, FAISS's vector copy and other process allocations.

The longest combined title/abstract had **1,939 model tokens**; **3,681 of 5,183 documents** exceeded the declared 256-token input limit and were right-truncated. This is a substantial limitation, recorded rather than silently changing the predeclared config after evaluation. Complete data and model files remain unchanged. Original preparation metadata is preserved in `runs/m3-dense-preparation.json` and the verified cache manifest.

## Measured validation and final checks

Four official-data validation runs completed: BM25 and dense first runs plus a separate-process reproduction of each. Each scored the unchanged 162-query validation partition and measured 810 warm retrieval calls. Recall/nDCG were **0.801337 / 0.667302 for BM25** and **0.811831 / 0.648212 for dense**. Each method's complete query artifact reproduced byte for byte. Both paired differences' 95% group-bootstrap intervals cross zero; no clear quality winner is established.

All four reports passed data/split/artifact verification and matched final retrieval/evaluation source hashes. Every per-query metric passed the independent reference backend check. Both dense runs agreed with NumPy on all 162 top-10 rankings, with maximum score error 1.1921e-7 and zero near-tie reorderings. No query exceeded the 256-token limit. First/reproduction timings, CPU/RSS, complementary errors and concrete success/failure examples are in [the measured report](results/DENSE_COMPARISON_V1.md).

The final complete suite, with **both official-data and real-model opt-ins enabled, passed 112 tests in 18.73 seconds with no skips**. Ruff lint and formatting passed; `pip check` found no broken requirements. The final 0.3.0 wheel built with SHA-256 `52d975b8c5fac6044c3989473b1525b275925523cc40162f031e6bc2cf043d01`. Logs: `runs/pytest-milestone3.txt`. All local Markdown links checked at that stage existed. The repeated-experiment script's command help ran; the equivalent individual benchmark commands were actually executed rather than claiming a separate complete script run.

## Live desktop browser verification

- BM25 search returned real abstracts in the updated application after the old server was restarted.
- Selecting Dense loaded the already prepared local model/cache. Searching validation claim `0` as free text returned judged document `31715818` at rank 4, with cosine similarity 0.2879, matching the saved experiment. The full abstract expanded and was visibly verified. This UI call's individual cold timing is not substituted for the warm benchmark distribution.
- The paired comparison displayed the correct two first-run names, 162-query heading, quality/speed/memory table, candidate-minus-baseline differences and 95% intervals. The JSON export downloaded successfully and its parsed deltas matched the saved comparison.
- The dense validation report displayed Recall 0.8118, nDCG 0.6482, p50 18.49 ms, p95 23.26 ms, and 29 queries missing all judged evidence. A missed-evidence abstract (`13639330`) expanded and its complete text was visibly verified.
- Native screenshots saved as `docs/assets/m3-dense-search.png` and `docs/assets/m3-comparison.png`. The local comparison tab was left open as a deliverable. Browser radio `check()` initially timed out without changing state; clicking its visible label succeeded and the selected state was verified.
