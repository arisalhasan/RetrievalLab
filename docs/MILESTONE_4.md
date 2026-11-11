# Milestone 4: combine complementary searches with reciprocal rank fusion

BM25 finds shared words. Dense search finds similar learned text representations. Hybrid search runs both, then combines their ranked lists. It does not train another model or decide whether a scientific claim is true.

Milestone 3 showed complementary failures: dense alone retrieved judged evidence on nine validation queries, and BM25 alone on ten. That motivates an experiment, not a promise that combining them will win. These same validation queries are development data already inspected during earlier milestones; they are not a fresh holdout.

## Start locally

Use the existing Python 3.12 environment and SciFact/model preparation from [milestone 3](MILESTONE_3.md). Milestone 4 adds **no dependencies**, model downloads, GPU requirement or paid service beyond that stack.

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e ".[dev,dense]"
.\.venv\Scripts\python.exe -m retrieval_lab.cli verify-split
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Select **Hybrid** in the sidebar and search a claim. Open **Why this rank?** below an abstract to see each source's rank and contribution. The search export includes the complete source candidate IDs and fusion trace. Switching methods clears displayed stale results. The app loads the existing dense document cache; it does not encode the corpus during an interactive search. If prerequisites are missing, it gives `model-download` / `build-dense` instructions.

Fresh setup still needs the README's dataset download and optional CPU dependency installation, followed by:

```powershell
.\.venv\Scripts\python.exe -m retrieval_lab.cli model-download
.\.venv\Scripts\python.exe -m retrieval_lab.cli build-dense
```

Unchanged document/model/encoder settings reuse the milestone 3 embeddings. That cache identifies the actual corpus, model files, configuration, embedding-source hashes and library versions. A new embedding identity requires a separate corpus build; the old model's 256-token truncation and unaudited pretraining-exposure limitations still apply. The original document encoding was about eight minutes on the recorded Windows CPU machine; reusing it avoids repeating that work. Other platforms and hardware may differ.

## The fusion calculation

For a document `d`:

```text
RRF(d) = sum over methods containing d: 1 / (c + rank_of_d)
```

Ranks start at **1**. A document outside a source's candidate list contributes **0** from that source. The constant `c` reduces how sharply rank 1 dominates later ranks; it is different from the requested result count `top_k`.

Example with `c = 60`: BM25 ranks A first and B second; dense ranks C first and B second. A and C each get `1/61 ≈ 0.016393`. B gets `1/62 + 1/62 ≈ 0.032258`, so B comes first. The two remaining scores tie and string document ID decides their order. One shared document is returned once with both contributions.

Raw BM25 and cosine scores use different scales. RRF uses rank positions rather than adding those raw scores or assuming calibration between methods. It discards score-gap information, so it can also promote a consensus mistake or push a relevant single-source result down. The method follows the formula in [Cormack, Clarke and Büttcher's original RRF paper](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf); its reported success on other collections does not establish success on SciFact.

The first milestone 4 settings were fixed **before this milestone's measured runs**:

- Equal contribution formula for BM25 and dense; **constant 60**.
- Up to **100 candidates from each method**; their union contains at most 200 documents.
- Final **top 10** for validation. At most one copy of each document, score descending then string document ID ascending for ties.
- The unchanged BM25 and pinned MiniLM settings from milestone 3, including normalized float32 CPU embeddings, 256-token inputs, seed 42 and one PyTorch/FAISS CPU thread.
- The unchanged grouped 647-train / 162-validation partition derived from official training labels. Official test labels remain reserved.

This is a fixed baseline experiment. A single planned **depth-10 ablation** holds the constant and both components fixed while changing only candidate depth from 100 to 10. Both first/repeated outcomes are retained; the main depth-100 configuration is not selected from the resulting scores. Candidate depth and the rank constant are explicit in `configs/hybrid.json` and can be changed for separately named validation experiments. No test labels may select them. Bigger candidate lists can increase coverage and cost, and they may change final rankings; they do not guarantee better Recall@10.

## Read the code in this order

1. `hybrid.py`: `HybridConfig` stores complete settings for both components. `fuse_rankings()` is a pure function over ordered results. It validates duplicates/conflicting IDs, uses accurate summation, and creates a label-free trace. `HybridIndex` runs both children serially on every call, then fuses their candidate lists. There is no query-result cache or silent fallback if one method fails.
2. `retrievers.py`: selects BM25, dense or hybrid behind the same `search(query, top_k)` interface. The underlying encoder, vector search and BM25 implementation are unchanged.
3. `evaluation.py`: checks the deeper dense candidate search against NumPy, warms up, times actual full hybrid searches, evaluates Recall/nDCG, and performs a separate untimed pass for candidate analysis. Labels enter here, after retrieval, never the search model.
4. `dashboard.py` and `app.py`: display live fusion contributions, candidate coverage, missed evidence and verified saved comparisons. Experiment timing runs in a separate CLI process.
5. `scripts/run_milestone4.py`: snapshots component/config settings, prepares/verifies prerequisites, runs all three methods sequentially and writes paired comparisons. Each run/output has a new name; earlier evidence cannot be overwritten.

Work for fusion is proportional to the combined candidate-list lengths, followed by sorting their union. This implementation builds the trace inside each timed search. Overall hybrid time includes both retrievers, and the dense component still scans all 5,183 document vectors exactly. It remains a small-corpus reference design, not an approximate million-document index.

## Reproduce the experiment

```powershell
.\.venv\Scripts\python.exe scripts/run_milestone4.py --name student-m4 --repeat
```

Choose a new name each time. The script saves complete settings in `runs/student-m4-configs/`, first/repeated BM25, dense and hybrid experiments, and hybrid-versus-each-baseline JSON comparisons in `comparisons/`. It derives each standalone baseline's configuration from the same nested hybrid config, so component settings match. Fresh processes and sequential execution avoid model work from other experiment processes competing during the measured stage.

For a separate custom experiment, copy `configs/hybrid.json`, change the declared rank constant/depth, then pass `--hybrid-config path/to/your.json` with a new `--name`. The final evaluation cutoff remains 10, and depth must be at least 10. Keep all trial outcomes; do not describe a selected validation winner as an untouched test result.

To reproduce the planned depth-10 comparison as well, add `--depth-ablation 10`. The script snapshots and runs that additional hybrid configuration, compares it with the main hybrid, and repeats it when `--repeat` is set. This is a diagnostic ablation, not an automatic hyperparameter-selection command.

Individual commands also work:

```bash
python -m retrieval_lab.cli evaluate --system hybrid --output runs/my-hybrid-validation-v1
python -m retrieval_lab.cli compare --baseline runs/my-bm25-validation-v1 --candidate runs/my-hybrid-validation-v1 --output comparisons/my-bm25-vs-hybrid.json
python -m retrieval_lab.cli compare --baseline runs/my-dense-validation-v1 --candidate runs/my-hybrid-validation-v1 --output comparisons/my-dense-vs-hybrid.json
```

The two baseline folders must already exist and use the same frozen data/split/evaluation settings and matching component configurations. Reload Streamlit, open **Compare systems**, select a baseline and hybrid run, and compare. **Validation & failures** also exposes the hybrid trace and candidate diagnostics for each query.

## Understand candidate coverage and performance

Candidate recall asks what fraction of known relevant documents occurs anywhere in the two candidate lists before choosing the final ten. This is an **upper bound on attainable recall from that pool**, not a model's measured Recall@10. The evaluator reports each source's recall at the configured depth and their union's recall separately.

A missing relevant abstract can fail at either stage:

- **Candidate miss:** it is absent from both source lists. Fusion cannot recover it.
- **Fusion loss:** it occurs in the candidate union but falls outside the final top 10. Better candidate recall alone has not solved ordering.

Only judged positives define these diagnostics. Unjudged documents are not proven negatives. Self document IDs are excluded consistently: final evaluation removes them after the retrieval cutoff without backfill; candidate diagnostics remove self IDs from the already truncated source lists. Related validation claims stay together during paired 95% group-bootstrap intervals.

Query latency includes BM25 at the configured depth, fresh dense query encoding, exact vector search, fusion, sorting and trace creation. It excludes model/index preparation, labels/metrics, diagnostics, UI and reporting. There is one warm-up pass plus five seeded shuffled passes: 810 calls per method. Model/cache loading and building both indexes are recorded separately. Process CPU and sampled RSS include framework/interpreter/data/observer allocations and can miss brief peaks; they are not isolated model memory or GPU/energy measurements.

Runs occur on an uncontrolled workstation. Retain first and reproduction timings rather than selecting the fastest. Historical milestone 3 timings remain in their own reports. Do not claim that RRF is cheaper than one child: this implementation executes both methods for every search.

## Tests and your checkpoint

Default tests never download models or data. New tests check hand-calculated one-based ranks, consensus, empty/short lists, missing sources, raw-score scale independence, duplicate rejection, deterministic ties, an independent exact-rational oracle, config round trips, live child calls, candidate/fusion failures, report integrity, CLI integration and Streamlit explanations. A clearly named synthetic encoder substitutes only at the external neural-model boundary.

```powershell
$env:RUN_SCIFACT_TESTS = '1'
$env:RUN_DENSE_MODEL_TESTS = '1'
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```

Please verify locally: search the same claim with all three methods; inspect one hybrid **Why this rank?** explanation; compare the saved hybrid report against BM25 and dense; and inspect one candidate miss or fusion loss. Be prepared to explain any regression as well as improvements.

Three internship interview questions:

1. Why combine ranks rather than raw BM25 and cosine scores, and how is an RRF score calculated?
2. What is the difference between candidate recall and final Recall@10, and why can fusion still lose relevant evidence?
3. How do candidate depth and the fusion constant affect rankings and cost, and how would you choose them without leaking test labels?

The next milestone is pretrained cross-encoder reranking over a bounded candidate set. No reranker training or fine-tuning is implemented in milestone 4.

Read [the actual hybrid results](results/HYBRID_COMPARISON_V1.md) and [execution record](VERIFICATION_MILESTONE_4.md). The deeper pool improved candidate coverage but had lower final point scores than the depth-10 ablation; the paired differences and uncertainty are preserved rather than selecting a single flattering result.
