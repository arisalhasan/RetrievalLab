# Milestone 8 — Frozen, held-out evaluation and a portfolio release

The official SciFact test is a final measurement, not another source of tuning feedback. All model choices and settings are recorded before the first explicit parse of its relevance labels.

## Freeze and evaluate

`configs/release.json` names every candidate configuration and its verified validation report. `release.freeze` reads official **train** judgments and verifies these reports. It chooses the candidate with highest validation nDCG@10, then Recall@10, then the lexicographically first name. This is an exploratory choice because development validation has been inspected repeatedly; it is not a significance claim. Both the primary hybrid and previously declared depth-10 ablation remain visible.

`releases/v1/freeze.json` seals configurations, validation evidence, data hashes, model/checkpoint asset hashes, runtime versions, source-module hashes, the selection rule, and the final measurement protocol. Hashing the raw test-label file verifies its identity without parsing relevance judgments. The freeze records that test labels have not yet been parsed.

```powershell
.\.venv\Scripts\python.exe -m retrieval_lab.release freeze --spec configs/release.json --output releases/v1/freeze.json
.\.venv\Scripts\python.exe -u -m retrieval_lab.release evaluate-test --freeze releases/v1/freeze.json --output runs/m8-official-test-v1
```

The second command verifies the freeze before opening test labels. Any source, dependency, data, model, config, or validation-artifact drift causes refusal. It writes an explicit access event before parsing judgments. Completed per-system results are never overwritten. Repeating the command resumes missing systems while re-verifying completed results; it does not tune or reselect the model. The pre-test selected name remains unchanged even when another system has a better test score.

For a fresh reproduction, preserve the original release and choose new paths. Use the recorded dependency snapshot and model assets. A deliberately changed system belongs in a new development study; the already-inspected test cannot honestly become “untouched” again. On a different OS/runtime the strict original freeze will intentionally fail verification. Generate and document a new environment-specific freeze before any new evaluation and disclose that the test has already been seen.

## Measurement protocol

Quality covers every official test query with official relevance labels. Recall@10 is the fraction of known relevant documents found among the ten retrieved results. nDCG@10 rewards placing relevant evidence nearer the top, normalized against the ideal order. Metrics use linear gains, zero gain for unjudged documents, query macro averages, and the same self-ID filtering as previous milestones. pytrec_eval independently checks the calculations.

Every system uses the same label-independent sample of 16 query IDs, seed 42, one warmup pass, three shuffled measured passes, and fresh complete inference for each timing call. Thus each system has 48 timed calls; quality is **not** reduced to 16 queries. Model/index preparation is measured separately. Process CPU time and sampled RSS are recorded; RSS includes the Python process and libraries, and sampling can miss short peaks. These are workstation measurements, not production load tests. Timings and training memory are not claims about GPU behavior or million-document scale.

The final CLI launches a fresh Python process for each system, preventing earlier models and allocator state from carrying into the next system's preparation/memory measurement. The coordinating process and each worker record explicit test-label access. The lower-level library function also supports in-process synthetic checks; the actual release uses isolated workers. Windows native libraries require the normal local process environment used by the documented commands.

Test confidence intervals use 1,000 seeded query bootstrap samples. Unlike development's grouped bootstrap, test queries are singleton groups: correlated or related test claims can make these intervals too narrow. Paired intervals are descriptive, exploratory comparisons; no adjustment for multiple comparisons is claimed. All systems use exactly the same labels, IDs, and protocol.

A descriptive audit after opening the test found 157 test queries sharing positive evidence with the 647 used training claims, and 61 test queries with a lexically similar training claim at the already-used token-Jaccard threshold of 0.8. These are properties of the official benchmark split, which was preserved. No test labels trained the model and no settings changed after this audit. Report the overlap when discussing generalization; do not claim a fully independent topic/evidence holdout. See the model card and `releases/v1/post-test-data-audit.json`.

## Artifacts and honest presentation

Each final run contains `manifest.json`, `split.json`, `queries.json`, `timings.csv`, and a readable report, with checksums. The legacy internal key `validation_query_ids` stores the common evaluator's query-ID list; official partitions explicitly retain `source_split: test`, `partition: official_test`, and a freeze identity. No official test is disguised as training data.

The Streamlit **Final test** tab verifies reports, displays quality/latency, exports scores, and shows missed relevant evidence. Error analysis after testing is descriptive; it must not become hidden tuning followed by another “first” test result. The final report includes model regressions and limitations, not just the best score. See `docs/results/OFFICIAL_TEST_V1.md` and `docs/MODEL_CARD.md` for measured evidence and appropriate claims.

## Verify this milestone

Compare the freeze timestamp with the first `test-access.jsonl` event. Check that the selected name in the freeze equals `selected_before_test` in final `result.json`. Match table values to per-system manifests and raw rankings. Try editing a *copy* of a frozen configuration in a disposable copy of the project; the guard must refuse it. Do not edit original release evidence. Check the tests for tampered freezes, changed configs, forbidden unapproved test evaluation, complete official-test results, and resume behavior.

Interview questions:

1. Why must the model selected before test stay selected if another model wins on test?
2. What do bootstrap uncertainty intervals say, and why can related queries weaken them?
3. Which artifacts would another engineer need to reproduce a result and investigate a regression?
