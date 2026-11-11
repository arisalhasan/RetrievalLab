# Milestone 4 execution record

Reference environment: **9 November 2026, Windows x86-64, Python 3.12.14, RetrievalLab 0.4.0, CPU only**. The model, dense runtime packages, document encodings and BM25 scoring implementation are unchanged from milestone 3. Exact settings/package/model/source/data fingerprints accompany each measured run. No synced `sources/` files were changed.

## Tests and implementation checks actually executed

- Pre-change baseline: **109 passed, three explicit skips** (official-data and real-model opt-ins disabled).
- Incremental RRF/shared-evaluation checks: **28 passed**. Integration/UI checks then passed **35**, followed by **37** after adding candidate-stage edge cases and live hybrid explanations.
- RRF hand calculation covered one-based ranks, repeated appearances across sources, zero contribution when absent, consensus, empty/short lists and string-ID tie ordering. An independent `fractions.Fraction` rank-table oracle checked results for seeded randomly ordered lists; raw-score rescaling and source-order changes did not alter fusion.
- Invalid configs, duplicate source IDs, conflicting document content, candidate-depth limits, config snapshots and immutable model revision checks passed. A component test verified both children are called at the configured depth on every query, without a query-result cache.
- Candidate diagnostics distinguished absent evidence from relevant candidates dropped below rank 10, excluded self IDs, retained no-positive queries at zero, and rejected an altered aggregate. Loading a hybrid report rebuilds scores/contributions from saved source ranks before presenting it.
- CLI integration exercised actual BM25/dense/RRF composition with an explicitly synthetic external encoder. Deliberately invalid test-shaped fixture labels were never read. The synthetic integration artifacts are not benchmark results.
- Streamlit AppTest covered hybrid search, both source contributions, method switching without stale RRF results, model/cache setup recovery and saved hybrid failure/diagnostic displays.
- Updated package 0.4.0 installed editable; `pip check` reported no broken requirements. No packages or model weights were newly installed/downloaded for milestone 4. Ruff lint and formatting passed (42 Python files).
- Complete suite with both official-data and real-model opt-ins enabled: **131 passed in 28.23 seconds**, no skips (`runs/m4-tests-full-corrected.log`). A focused pre-experiment rerun passed **37** tests.

## Test-isolation issue found and corrected

The first full run had **130 passes and one failure**: a synthetic UI test patched the model-store verifier before importing the encoder module. That module retained an alias to the temporary stub after the patch was restored, so the later real-model check received an incomplete synthetic manifest. Importing the real encoder before replacing the UI boundary corrected the test isolation. The fresh full-suite rerun passed all 131 tests. Production model verification and encoding source files were not changed to accommodate a test stub.

An installed Transformers `torch_dtype` deprecation message was observed during real-model startup. It is retained in logs and is not an execution failure. The older milestone 3 native-DLL/profile and interrupted-download issues remain documented in that milestone's record; no protection, certificate or permission settings were weakened here.

## Experimental design and boundaries

Main RRF constant 60 and candidate depth 100 were declared before milestone 4 measurement. A single depth-10 ablation was also declared before running, with all other settings fixed. Neither is a test-selected model; no automatic hyperparameter winner was chosen. Validation had already been used for development and inspection during earlier milestones. First and repeated outcomes are retained even when the ablation's point score exceeds the main setting.

The unchanged split has 647 derived training and 162 validation queries across 98 validation groups, seed 42, with zero positive-evidence document overlap. Only official training judgments are parsed. Test bytes remain in the verified dataset archive/cache, but no official test evaluation or test-label model selection was performed.

Performance is measured in sequential fresh processes: one full warm-up pass and five shuffled measured passes, 810 complete live search calls per method. Hybrid includes both searches at configured candidate depth, fresh query encoding, fusion, sorting and trace creation. Post-measurement candidate diagnostics are untimed and never replace the scored final top 10. Preparation is separate and reuses verified document vectors. Workstation load, OS caches and power state are uncontrolled; CPU/RSS cover the whole process and sampled peaks can miss brief allocations.

## Not executed

- Official test evaluation, full BEIR neural API execution, reranking, fine-tuning, gradient updates and GPU inference.
- A rank-constant sweep, automatic best-config selection, pretraining-overlap audit or strict zero-shot novelty validation.
- macOS/Linux, fresh-machine lockfile installation, million-document load testing, energy measurement or isolated per-model memory profiling.
- Narrow-screen/mobile browser layouts; live checks used the default 1280 × 720 desktop viewport without an override.

## Measured runs and reproducibility

The actual command `python scripts/run_milestone4.py --name m4 --repeat --depth-ablation 10` completed successfully. It verified the split/model, reused the existing document cache, snapshotted four matching configs, and produced **eight validation runs and six paired comparisons**. Every run measured 810 live calls on the unchanged 162 validation queries.

Primary hybrid Recall/nDCG were **0.838066 / 0.713230**; the declared depth-10 ablation scored **0.866358 / 0.729131**. Baseline BM25/dense quality matched milestone 3 exactly. Primary-versus-BM25 paired intervals cross zero; the primary-versus-dense nDCG interval is above zero. These remain exploratory development comparisons. The [measured report](results/HYBRID_COMPARISON_V1.md) retains the complete quality/coverage/resource tables, first/repeat timings, uncertainty and a fusion regression.

All eight saved reports passed dataset/split/artifact checks and matched current retrieval/evaluation source hashes. All eight environment snapshots were identical. Each system's complete first/repeated query artifact was **byte-identical**; paired differences and bootstrap intervals also matched exactly. Repeating the same experiment name was explicitly rejected before work started, with saved manifest/comparison hashes unchanged.

Every per-query metric passed the independent pytrec_eval backend, with maximum error at most 2.22e-16. All dense/actual-depth hybrid vector checks agreed with NumPy on all 162 rankings, zero near-tie reorderings, maximum score error 1.1921e-7. Primary hybrid checked top-100 dense candidates; the depth-10 ablation checked top 10. BM25, dense, encoder and vector source hashes were explicitly verified unchanged from milestone 3. Git metadata remains unavailable in this mirror.

All dense/hybrid runs recorded verified document-cache hits; the corpus was not re-encoded. The original truncation/preparation metadata remains part of that cache identity and provenance. Query timing retained real fresh encodings and both live child calls. Script stdout is in `runs/m4-experiments.log`.

## Live browser verification

The previous preview tabs were unavailable while the local server was stopped. Starting the local server and opening a new allowed localhost tab restored the preview; no network/security settings were changed.

- Selected Hybrid and observed the declared constant 60 / depth 100 in the sidebar.
- Searched the recorded claim `165` as free text. Judged document `5824985` appeared first, matching the saved experiment. **Why this rank?** displayed BM25 rank 3 and dense rank 3, each contributing 0.015873, total 0.031746. The UI's individual cold query timing was not substituted for the CLI benchmark.
- Downloaded/parsed the search JSON: system hybrid, correct first document, source ranks, contribution values and total score were present.
- Compared `m4-bm25-validation-v1` with `m4-hybrid-validation-v1`: the live table, 162-query count, deltas +0.0367 / +0.0459 and their intervals matched saved evidence. Downloaded/parsed comparison JSON reproduced the exact deltas.
- Opened the saved hybrid validation report. It displayed Recall 0.8381, nDCG 0.7132, p50/p95 31.35 / 44.25 ms, 24 complete evidence misses, and candidate recall 0.9877. For query `0`, the diagnostic view correctly showed no evidence outside the union and document `31715818` lost below fused top 10.
- Native screenshots saved as `docs/assets/m4-fusion-explanation.png` and `docs/assets/m4-comparison.png`; both are 1280 × 720. The comparison tab was left open as a deliverable. No mobile-layout claim is made.

## Final package checks

The final complete suite, with real-data and real-model opt-ins enabled, passed **131 tests in 32.34 seconds with no skips** (`runs/pytest-milestone4.txt`). Ruff passed and all 42 Python files were formatted. `pip check` found no broken requirements. The 0.4.0 wheel built successfully with SHA-256 `6f7cfdd313cc6e1b3df6b53ed070436b9d944e43fd817a23bae6ade48ce2368a`. A wheel/compatibility check is not a fresh-machine installation test.

The portable source ZIP includes the guides, tests, settings, screenshots and actual measured evidence. Dataset/model files, embedding caches, virtual environments, temporary test directories and synced references are excluded. Historical milestone artifacts/runs remain preserved.
