# Milestone 1: explain and verify

The working path is official dataset → validated documents → BM25 index → query → ranked abstracts → Streamlit/JSON. A baseline makes later ML gains interpretable.

## What the code teaches

The downloader checks the published archive and creates a manifest. This is **data provenance**: someone else can identify the exact files behind a run.

The loader converts raw files to predictable objects and refuses malformed references. An explicit split prevents accidental merging. One test deliberately breaks test judgments and proves train loading does not depend on them.

The index counts tokens and stores postings. BM25 rewards rare shared terms, limits repetition benefits, and normalizes document length. Work through `test_score_matches_hand_calculated_bm25` with a calculator and the formula in `ARCHITECTURE.md`.

The UI calls the same core as the CLI. It shows actual scores and one query time. Scores order results for a query; they are not calibrated across queries or scientific truth assessments.

Hand calculations catch math mistakes; synthetic BEIR files catch parsing/split mistakes; AppTest exercises controls and results. The optional real-data test catches official-file compatibility issues. Synthetic fixture outputs are never benchmark evidence.

## Student verification checkpoint

Please run the README setup and verify these before moving to milestone 2:

1. Download twice. The second run should verify/reuse the cache. Inspect `data/scifact/manifest.json`.
2. Search `cell division and mitosis`. Open an abstract and find a matching query token. Poor matches are still possible with lexical search.
3. Search a training claim and inspect annotations. Unjudged does not necessarily mean irrelevant.
4. Search `qzxqzxunknown` for the no-match state. Submit a blank query for input guidance.
5. Save two CLI runs with different filenames. IDs/scores should match; timestamps/timings can vary. Confirm `split: train`, settings and data hashes.
6. Run default and opt-in dataset tests. Share the summary and one search observation with your mentor, without calling search scores benchmark results.

The agent's execution record is in `VERIFICATION.md`. Your local result is the checkpoint for proceeding.

## Three interview questions

1. **Why can BM25 beat a neural retriever on some scientific queries?** Explain rare exact terms/identifiers, lexical overlap, and limitations with synonyms.
2. **What do k1, b, and IDF do?** Explain term-frequency saturation and length normalization, then describe when each changes a ranking.
3. **Why is indexing the full corpus valid while tuning on test judgments is not?** Separate searchable inputs from held-out supervision and describe creating validation data from official train.

Answer in your own words before reading advanced retrieval code.
