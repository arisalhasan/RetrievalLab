# Milestone 5 execution record

Reference environment: **9 November 2026, Windows x86-64, Python 3.12.14, RetrievalLab 0.5.0, CPU only**. No Python dependencies were added. The new pretrained reranker is separate from the unchanged document encoder and hybrid candidate generator. Historical experiments and synced `sources/` files remain preserved.

## Checks actually executed

- Pre-change suite: **128 passed, three explicit data/model skips**, before implementing reranking.
- Reranking core: **20 passed**, including the real pinned TinyBERT test. It checked direct PyTorch logits against the wrapper, semantic toy ordering, exact repeated scores, and pair truncation, alongside candidate membership, ties, bad scores/configs and live repeated scorer calls.
- Shared evaluation/RRF/rerank regression checks: **49 passed, one explicit real-reranker skip** in the default focused run.
- Download/evaluation/UI checks: **45 passed, one explicit real-reranker skip**; after adding live reranked-search coverage, **46 passed, one explicit skip**. Synthetic external scorer/encoder fixtures are clearly named and never used as SciFact benchmark scores.
- Complete suite with official SciFact, real dense encoder and real reranker opt-ins enabled: **160 passed in 21.81 seconds with no skips** (`runs/m5-tests-full.log`). Editable 0.5.0 installation, `pip check`, Ruff lint and formatting passed; 49 Python files were formatted.
- The small timing-sample test evaluated every query in an expanded synthetic validation partition while timing only one ID. Saved report validation checked the full declared ID/repetition coverage. Invalid timing limits were rejected.
- Reranking integration succeeded with deliberately invalid test-shaped fixture labels, showing the pipeline/evaluator did not read them. It exercised real BM25/dense/RRF composition with explicitly synthetic external model boundaries, trace integrity, candidate ceilings, summary tamper rejection and Streamlit saved reports.
- UI AppTest checked the new method, original/final rank messages, scalar-score labels, missing-model instructions, and switching back to BM25 without stale reranker scores. Neural modules were imported before temporary file-verification stubs to avoid the import-alias isolation issue found in milestone 4.

## Verified pretrained assets and inference scope

The public Hugging Face model API was inspected before selection/measurement. Model `cross-encoder/ms-marco-TinyBERT-L2-v2`, commit `81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc`, has a two-layer, 128-hidden-unit BERT sequence-classification architecture with one scalar output. The safetensors file is **17,552,980 bytes**, SHA-256 `a0e7364ddf91ff7028f1102e1b91ac7a72e3db4061241bd84efe45c72c9af03a`.

All seven allowed model/tokenizer/card assets downloaded successfully via the existing TLS-verified bounded transport and passed upstream Git/LFS and local checksums. No remote Python or pickle weights were loaded; runtime uses local files only with remote code disabled. The pinned installed Sentence Transformers 5.1.2 source/signatures were inspected to match its actual tokenizer/identity-activation API rather than assuming the latest documentation interface. Existing dense runtime packages and retrieval weights were reused.

The small pretrained model was selected for CPU practicality before milestone 5 validation scores. The pipeline uses the top 50 from the existing primary hybrid, 512 joint tokens, longest-first truncation, batch 16, float32 CPU eager inference, one thread, seed 42, raw logits and no training. A real-model test independently matched those logits with a direct forward pass. Model-card GPU throughput was not substituted for local CPU measurements.

## Declared evaluation profile

Full quality remains all **162** frozen validation queries. The new explicit `evaluation-m5.json` profile selects **32 IDs** uniformly with seed 42 without using relevance labels to choose timing cases. A separate untimed full-quality pass precedes one selected-query warm-up and five shuffled measured passes: **160 live calls per system**. Every timed reranker call recomputes retrieval, query encoding and pair scores; retained quality traces serve only post-measurement diagnostics.

Full-quality input/ordering/truncation diagnostics are untimed. Exact sample IDs, repetition rows, configs, model/data/source hashes and package snapshots are persisted; comparisons reject different configs or timing IDs. These timings are a small descriptive sample and are not interchangeable with historical all-query 810-call measurements. Workstation power/load/OS caches are uncontrolled. RSS is sampled whole-process memory, not isolated model allocations or a continuous guaranteed peak.

## Not executed

- Official SciFact test evaluation, test-label tuning, gradient updates, fine-tuning, checkpoint training, negative mining, GPU inference and pretraining-overlap audit.
- The full BEIR neural API, FastAPI/platform work reserved for the next milestone, a model/depth sweep, or automatic best-model selection.
- macOS/Linux, clean-machine lockfile installation, million-document load testing, GPU/energy/isolated-model memory profiling, or a claim of causal architecture-only improvement.

## Measured experiment execution

The actual command `python scripts/run_milestone5.py --name m5 --repeat` completed successfully: **eight fresh-process runs and six paired comparisons**. Every run evaluated all 162 validation queries and measured the same 32 query IDs across 160 timed calls. All selected IDs and environment snapshots matched; all reports passed artifact/data/split validation and current retrieval/evaluation source hashes. Reusing the existing experiment name was explicitly rejected before work, with saved manifest/comparison hashes unchanged.

Every system's complete first/repeated `queries.json` was **byte-identical**, including all neural logits, input/fusion traces and token diagnostics where applicable. Paired differences and bootstrap intervals also matched. BM25, hybrid, dense, encoder and vector source hashes were verified unchanged from milestone 4. Model loading reused the existing verified document vectors; no corpus re-encoding occurred.

TinyBERT's Recall/nDCG were **0.819033 / 0.703273**, versus hybrid **0.838066 / 0.713230**. All paired quality intervals crossed zero. The fixed small model did not improve the hybrid's point scores and added query cost; the result was retained. Its first/repeated p50 timing was **1,185.57 / 732.47 ms**, with p95 **1,303.43 / 1,277.38 ms**. Full first/repeated resource/quality evidence and limitations are in [the measured report](results/RERANKING_COMPARISON_V1.md).

The bounded top-50 input had recall 0.946296. Of the 8,100 full-quality pairs, 1,069 exceeded the 512-token joint budget; maximum input length was 1,968 tokens. Candidate omissions and final ordering losses are separated. All reference checks and independent dense-search checks passed; the real-reranker unit check independently validated raw logits rather than assuming that ordering improvement proves inference correctness.

## Live browser verification

The prior local preview server was restarted and the existing localhost tab reloaded to obtain current milestone 5 modules. The available preview was a **493 × 438** compact viewport. Its sidebar was initially collapsed; expanding the menu allowed method selection. No viewport override or network/security setting change was applied. A full desktop/mobile/device audit was not executed.

- Selected Reranked and observed the 50-candidate / 512-joint-token setting. Both local models/cache loaded successfully.
- Searched the exact recorded claim `618` as free text. Document `6836086` ranked first and displayed raw logit -4.501628, matching the saved experiment. **How the rank changed** displayed hybrid rank 16 → reranked rank 1, with an explicit uncalibrated-score explanation. The UI's individual timing was not substituted for the CLI percentiles.
- Downloaded/parsed search JSON: system rerank, 50 input candidates, correct first document, original/new rank and exact raw logit.
- Compared `m5-hybrid-validation-v1` and `m5-rerank-validation-v1`. The 162-query heading, table, deltas -0.0190 / -0.0100 and paired intervals matched saved evidence. Downloaded/parsed comparison JSON matched the exact deltas. One selector click timed out after opening its menu; fresh state inspection showed the open menu, and selecting the visible option succeeded.
- The saved neural report explicitly displayed 160 timed calls on 32 of 162 queries while quality covered every query. Candidate diagnostics showed input recall 0.9463, 8,100 pairs, 1,069 truncated and maximum input length 1,968. Query `0` correctly showed document `31715818` present in the input but lost below the final cutoff.
- Native screenshots saved in `docs/assets/m5-rank-movement.png` and `docs/assets/m5-comparison.png`. The comparison tab was left open as a deliverable. No generated/fabricated UI results were used.

## Final package verification

The source remained unchanged after the complete 160-test, real-model/data-enabled run. Final Ruff lint/format checks passed for all 49 Python files; `pip check` found no broken requirements. The 0.5.0 wheel built successfully, SHA-256 `ce5a74629353f4ab2068f6076bbf8e9923a6d45abc456e59e0c32c8e8401ebec`. Both native proof images were verified as 493 × 438. A wheel build/compatibility check is not a clean-room installation test.

The portable milestone 5 source artifact includes configuration, guides, tests, screenshots, the eight actual new runs, six comparisons and preserved earlier evidence. Virtual environments, downloaded weights/datasets, document caches, temporary test directories and synced references are excluded. The full suite was not needlessly repeated after documentation-only changes; its actual output remains in `runs/m5-tests-full.log`.
