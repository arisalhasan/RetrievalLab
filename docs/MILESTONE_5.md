# Milestone 5: pretrained cross-encoder reranking

Retrieval proposes documents quickly. A reranker reads each query and candidate abstract **together**, assigns a relevance score, and changes their order. It does not search outside the proposed candidates or decide whether a scientific claim is true.

The dense retriever encodes documents once and compares vectors. A cross-encoder instead permits interactions between query and document tokens inside the neural network. Those interactions can help with specific relationships, but must be recomputed for every query/document pair. That makes bounded candidate sets important for speed.

## Start locally

Use the existing Python 3.12 environment and the dense/hybrid setup. No new Python dependencies, paid APIs, database or GPU are required. Download the new public weights explicitly, then start Streamlit:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e ".[dev,dense]"
.\.venv\Scripts\python.exe -m retrieval_lab.cli reranker-download
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Select **Reranked**, enter a query, and open **How the rank changed** below a result. You can inspect its original hybrid rank, final rank and raw cross-encoder score. The search export contains all reranked candidate scores and their original retrieval order. It provides rank movement, not an invented explanation of the network's reasoning.

Fresh checkouts still need the README's dependency/data setup and the existing dense model/index:

```powershell
.\.venv\Scripts\python.exe -m retrieval_lab.cli download
.\.venv\Scripts\python.exe -m retrieval_lab.cli model-download
.\.venv\Scripts\python.exe -m retrieval_lab.cli build-dense
.\.venv\Scripts\python.exe -m retrieval_lab.cli reranker-download
```

The new reranker weights occupy **17,552,980 bytes**, approximately 18 MB. They are separate from the existing MiniLM retrieval weights and generated document vectors. The first model load takes additional time; later queries reuse loaded models. Every query still computes fresh pair scores. The UI never downloads weights or builds corpus vectors during search.

## Fixed experiment choices

These choices were declared before milestone 5 validation results:

- Model: [`cross-encoder/ms-marco-TinyBERT-L2-v2`](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L2-v2), immutable revision `81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc`. Its upstream task is MS MARCO passage ranking; the small two-layer, 128-hidden-unit architecture keeps CPU inference practical. This is a compact pretrained baseline, not a claim that it is the best reranker for scientific text.
- Retrieve with the unchanged milestone 4 primary hybrid: RRF constant 60, up to 100 candidates from each child. Pass only the **top 50 fused candidates** to the reranker. There is no claim that the full union is reranked.
- Query is the first sequence; title + newline + full abstract is the second. A **512-token joint budget**, including special tokens, uses the tokenizer's longest-first truncation. Both sides share this budget, unlike the retrieval encoder's separate 256-token inputs. A token is not necessarily a whole word.
- Float32 CPU, eager attention, batch size 16, one PyTorch/FAISS CPU thread, seed 42, evaluation mode, no gradient updates. CPU threads agree across both neural components.
- Identity activation: keep raw scalar logits, with no sigmoid, softmax or score threshold. Higher logits rank first, with string document ID ascending for exact ties. These are uncalibrated pair relevance scores, not probabilities of scientific truth.
- Keep the frozen 647-training / 162-validation partition from official train labels. No official test labels enter retrieval, scoring, timing selection or model selection.

There is no model sweep, automatic best-checkpoint selection, chunking experiment or fine-tuning here. The TinyBERT choice was driven by a small local CPU baseline rather than validation-score selection. Upstream pretraining/training overlap with SciFact is not audited; do not claim strict zero-shot novelty. The larger pair budget and different model are both part of this comparison, so any improvement cannot be attributed solely to joint attention.

## Follow one query through the code

1. `reranking.py` stores a complete `RerankerConfig`, including the nested hybrid/child configs. `RerankedIndex` requests the top 50 hybrid documents and creates one query/title/abstract pair for each. It never receives relevance labels.
2. `cross_encoder.py` loads verified local files through Sentence Transformers' `CrossEncoder` and PyTorch. Its `score()` returns one finite raw logit per pair. It uses a lock for shared UI model access, offline loading, safetensors and no remote Python code.
3. `rerank_candidates()` sorts only these candidates by their neural scores. It rejects duplicate IDs, non-finite scores and a wrong score count. It returns shared `SearchResult` objects plus a trace of original ranks/scores, all input logits and final ranks. Empty candidate lists remain empty; there is no silent fallback to an unrelated method.
4. `reranker_store.py` downloads only seven known assets at the pinned commit. It reuses bounded TLS-verified downloads, checks upstream Git/LFS hashes and the declared weights SHA-256, verifies the expected architecture, and installs atomically. The old retrieval encoder/vector files remain unchanged, allowing document-cache reuse.
5. `evaluation.py` and `rerank_diagnostics.py` score full validation quality and analyze the bounded rerank input after retrieval. They distinguish evidence that never reached the reranker from input evidence that fell below the final cutoff. Saved traces are rebuilt and aggregates checked when reports load.
6. `app.py` / `dashboard.py` show live rank movements and saved candidate/truncation diagnostics. The experiment CLI runs separately from the UI.

The raw logit is a learned scoring function `s(query, document)`. Sorting it produces a ranking; no probability threshold is needed for Recall@10 or nDCG@10. Applying a monotonic sigmoid would not add scientific-truth calibration, and can compress score differences. We preserve the raw score and record its interpretation explicitly.

## Reproduce full quality with an explicit timing sample

```powershell
.\.venv\Scripts\python.exe scripts/run_milestone5.py --name student-m5 --repeat
```

Use a new name for every experiment. The script verifies prerequisites, snapshots matching BM25/dense/hybrid/reranker settings and the timing profile, runs four fresh systems sequentially, repeats each in fresh processes, and writes reranker-versus-each-baseline comparisons. `runs/student-m5-configs/` holds the settings; run names include method and `v1` or `reproduction`. Existing evidence cannot be overwritten.

All four systems use `configs/evaluation-m5.json`:

- **Quality:** every one of the 162 validation queries. Save every ranking, score and failure case; check Recall/nDCG with pytrec_eval as before.
- **Latency:** the same **32 validation IDs** sampled uniformly with seed 42, without inspecting relevance labels to choose fast/easy cases. Their exact IDs are recorded. First run a separate full-quality pass, then one warm-up pass on the selected IDs and five shuffled measured passes: **160 timed calls per system**.
- Each timed reranked call runs real hybrid retrieval, fresh dense query encoding, fresh pair inference over every selected candidate, sorting and trace construction. There is no query/result/pair-score cache. Diagnostics/token counts are untimed.

This declared timing sample keeps a CPU experiment manageable while preserving complete quality evaluation. It is a small descriptive timing sample, not the latency distribution over every validation query or a production service guarantee. It differs from milestone 4's 810-call profile; preserve historical timings separately. The comparison rejects different evaluation configs or timing query IDs. Tests and report loading verify that the declared IDs/repetitions are all present.

The evaluator still supports the older full-query timing protocol: omit `latency_query_limit` or set it to `null` in a separate profile. That makes every validation query timed again and substantially increases reranking work. When changing the profile or candidate depth, use a new experiment name and rerun all baselines under the same profile. Example:

```bash
python -m retrieval_lab.cli evaluate --system rerank --retriever-config configs/reranker.json --evaluation-config configs/evaluation-m5.json --output runs/my-rerank-validation-v1
```

For custom settings, pass `--reranker-config` and/or `--evaluation-config` to the script. The model/settings are not tuned on test labels. A depth change also changes candidate coverage and the number of pair computations; keep those effects visible.

## Interpret candidate ceilings, failures and resources

The reranker can recover a document that hybrid ranked 17 if the input depth is 50. It cannot recover one ranked 70 or absent from the source union. **Reranker input recall@50** is different from the larger source-union recall and final Recall@10. The report shows all three stages where available, plus each query's hybrid top-10 quality before reranking and quality afterward.

Relevant evidence may still be ranked lower by a pretrained model trained in a different domain. Raw neural scores cannot establish support/refutation or label missing judgments as negatives. The framework preserves improvements and regressions rather than assuming a neural stage must win.

The report records actual pair counts and pre-truncation token lengths, including how many candidate pairs exceed 512 tokens. The earlier dense encoder still truncates many documents at 256 tokens during retrieval. Longest-first pair truncation can also shorten a long query. Full abstracts remain available in the UI; displaying them does not mean the network read every token.

Resource measurements are whole-process CPU time and sampled resident RAM, including both models, Python, corpus/indexes and observer overhead. They are not isolated reranker allocation counts, GPU usage or energy. One-time loading/preparation is separate from warm query timing. Background workstation load, power state and OS caches remain uncontrolled; retain both first/repeated timings. Upstream model-card GPU throughput is not a laptop CPU benchmark.

No model training is implemented. Optional fine-tuning remains a later milestone and needs its own compute/data plan before implementation, with training-only negatives/updates and validation-only checkpoint selection.

## Tests and your checkpoint

Default tests are offline and use explicitly synthetic scores only at external neural-model boundaries. They check bounded membership, raw-score ordering and ties, invalid outputs/configs, fresh inference calls, source/trace reconstruction, sampled timing versus full quality, malformed test-label isolation, model-file integrity and Streamlit rank movements. The real pretrained reranker test independently compares wrapper scores with direct PyTorch logits and checks semantic toy ordering, repeatability and truncation. Synthetic scores are never SciFact benchmark results.

```powershell
$env:RUN_SCIFACT_TESTS = '1'
$env:RUN_DENSE_MODEL_TESTS = '1'
$env:RUN_RERANKER_MODEL_TESTS = '1'
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```

Please verify locally: compare Hybrid and Reranked on the same claim; open a rank-movement explanation; compare saved quality and timing; inspect one improvement and one regression; and confirm that a missed document was or was not in the actual 50-document input.

Three internship interview questions:

1. How does a cross-encoder differ from a bi-encoder, and why is it usually more expensive per query?
2. Why is the reranker's candidate recall a ceiling, and how do candidate depth and joint token length affect quality and cost?
3. How can every validation query contribute to quality while latency uses a fixed sample, and what must be disclosed for a fair comparison?

The next milestone is the FastAPI service and broader platform integration. Fine-tuning is not required to demonstrate this pretrained retrieve-and-rerank pipeline.

Read [the measured tradeoffs](results/RERANKING_COMPARISON_V1.md) and [execution record](VERIFICATION_MILESTONE_5.md). The compact model did not improve hybrid point scores, all paired quality intervals crossed zero, and its extra CPU cost was retained rather than hidden. This is a complete measured baseline, not an assumed neural improvement.
