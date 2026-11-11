# A focused three-minute RetrievalLab demo

Use the local Streamlit dashboard and the API documentation at <http://127.0.0.1:8000/docs>. Prepare the data, pinned models, dense index and included trained checkpoint before presenting. Do not start a benchmark or train a model during the demo.

1. **Explain the problem.** “Given a scientific claim, find abstracts containing relevant evidence. I built a platform to compare how different search methods rank that evidence and what they cost to run.” Relevance does not tell us whether the claim is true; a paper can support or refute it.
2. **Show a search.** Enter `cell division and mitosis`. Start with BM25, inspect a title/abstract and its recorded configuration. Switch to Dense, Hybrid, Reranked or Fine-tuned. Explain that rankings change because word matching, semantic similarity and joint claim/document scoring use different signals. Raw scores have different scales.
3. **Show an interpretable mechanism.** Open a hybrid contribution trace: each appearance adds `1 / (60 + rank)`. Open a reranked result's rank movement. Explain that this shows the input/output ranking change, not an invented explanation of the neural model's internal reasoning.
4. **Show experiments, not anecdotes.** In **Compare systems**, compare matching M5 runs. Read Recall@10, nDCG@10, paired intervals, latency and memory. Download the JSON, CSV and figure. A selected example explains behavior; the complete query population supplies the benchmark score.
5. **Show scientific discipline.** In **Final test**, point out the system selected *before* test. Explain the 647 training / 162 validation split, two-epoch training experiment, safe checkpoint hashes, explicit pre-test freeze, and complete official-test evaluation. If a more complicated method loses, say so. Show an automatically listed missed relevant abstract.
6. **Show engineering.** In the API docs, send a BM25 search, then a blank query to show validation. Mention the shared retrieval core, lazy local assets, serialized CPU inference, repeatable commands and actual passing tests. Separate query time from first-request setup and HTTP time.

End with the measured result and one concrete limitation from the final report. Avoid claims about web-scale production deployment, clinical safety, statistical significance, GPU performance, or independently repeated training that this project has not demonstrated.

## Interview preparation

Each milestone guide includes three questions. For the final stages, be able to explain the API/core boundary, benchmark timing scope, negative sampling, train/eval modes, checkpoint selection, held-out evaluation, uncertainty and artifact provenance. Explain one measured regression without making excuses or hiding it.

## Portfolio checklist

- Link the source/release artifact, setup guide, architecture, model card and measured results.
- Show a real screenshot; keep validation and official test scores clearly labeled.
- Cite exact benchmark conditions when quoting a score or latency.
- Include limitations and what actually ran. The authored GitHub workflow has not been executed remotely.
- Use the truthful portfolio paragraph in the final report; update numerical claims only from saved evidence.
