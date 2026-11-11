# Milestone 7 — A small, auditable fine-tuning experiment

Fine-tuning starts with the pinned two-layer TinyBERT cross-encoder and adjusts its weights using SciFact training evidence. This is retrieval relevance training: a label says that an abstract is relevant to a claim. It does not classify whether the claim is scientifically true.

## Compute plan, explained before implementation

The reference computer has 8 physical CPU cores, 16 logical cores, about 15.25 GiB usable RAM, and no CUDA GPU. The approximately 4.4-million-parameter model uses float32 CPU inference/training, one PyTorch thread, batch size 8, a 512-token joint input limit, and two epochs. No GPU or paid service is needed. Planning allowed several minutes to tens of minutes; actual wall time and sampled process memory are in `runs/m7-training-v1/result.json`. This includes full validation checks before training and after each epoch. CPU backpropagation is slower than inference; there is no claim about measured GPU memory or GPU training speed.

## Data boundary

The already-frozen official-train-derived split contains 647 training and 162 validation claims. Shared positive evidence and near-duplicate claims were grouped before this split. This heuristic reduces leakage; it cannot guarantee topic independence. All 5,183 corpus documents remain searchable, as allowed in retrieval evaluation.

`prepare_pairs` visits only training query IDs. For every query it includes all known positive documents, one high-ranking BM25 document that is not a known positive, and one distinct seeded random nonpositive document. The experiment produces 2,024 pairs. An unjudged document is an **assumed negative**, not proven irrelevant; incomplete relevance judgments can introduce false negatives. Negative filtering never consults validation/test judgments. Validation query text and label-free hybrid candidates are used only for checkpoint evaluation, never for gradient updates or negative mining.

`pairs.json` records every query/document ID, label, and sampling source. `plan.json` records the split, parent revision, exact configuration, pair hash, source hashes, and installed package versions. Validation was already explored in earlier milestones, so selection is exploratory rather than a new untouched validation study.

## Training in plain language

The model reads a claim and an abstract together and produces one number, a raw relevance logit. Binary cross entropy with logits penalizes low numbers for known positives and high numbers for sampled negatives. In notation, the per-example loss is `-y log(sigmoid(z)) - (1-y) log(1-sigmoid(z))`. PyTorch computes it stably without manually applying sigmoid first.

AdamW updates weights at learning rate 0.00002 with weight decay 0.01. Gradients are clipped to norm 1.0. Every epoch shuffles training pairs with seed 42. Training enables dropout; validation disables it and uses inference mode. Padding is dynamic and long claim/abstract pairs use longest-first truncation to 512 tokens. There is no hyperparameter sweep, mixed precision, schedule search, or test-based early stopping.

Epoch 0 (the pretrained parent) and epochs 1 and 2 are evaluated on all 162 validation claims using the same fixed top-50 hybrid candidates. The rule, written before training, selects highest nDCG@10, then Recall@10, then earliest epoch. Thus the parent may win. A separate best-trained checkpoint remains available for an honest experimental comparison even if training loses to the parent. Validation metrics are checked against pytrec_eval.

Only safetensors weights and tokenizer/configuration JSON/text files are saved. Each checkpoint directory is named by its weight SHA-256 and includes per-file checksums and training provenance. Loading rechecks these files. Nothing overwrites the pretrained model. There is no pickle checkpoint, remote code, network inference, automatic retraining, or hidden test selection.

## Reproduce

Prepare the pinned pretrained assets, dataset, split, and dense index using the earlier milestone instructions, then run:

```powershell
.\.venv\Scripts\python.exe -u -m retrieval_lab.training --output runs/my-training-run
```

The configuration is `configs/training.json`. Use a new output directory for every run. A crash leaves the explicit run incomplete; this recipe does not resume optimizer state. `history.json`, per-epoch validation rankings, trained safetensors, `best-trained-config.json`, `selected-config.json`, and `result.json` make the experiment inspectable. Copy the best-trained config to `configs/trained-reranker.json` only when preparing an explicitly named release. It identifies the checkpoint by hash, not by a mutable “latest” filename.

To run a fresh comparable validation benchmark:

```powershell
.\.venv\Scripts\python.exe -m retrieval_lab.cli evaluate --system rerank --retriever-config configs/trained-reranker.json --evaluation-config configs/evaluation-m5.json --output runs/my-trained-validation
```

Training selection reuses label-free validation candidates to avoid redundant retrieval. Benchmark timing never uses cached pair scores or query results.

## Verify this milestone

Inspect pair IDs against the split and check that none are validation/test query IDs. Inspect each epoch's actual validation scores and the selection rule. Switch Streamlit to **Fine-tuned** after its config is installed and inspect rank movements. The real-model integration test also trains on a tiny synthetic fixture, saves a safe checkpoint, verifies its checksum, and reloads it for inference; those synthetic scores are never reported as SciFact results.

Interview questions:

1. Why can an unjudged BM25 result be an imperfect negative training example?
2. What do `train()`, `eval()`, gradient clipping, and binary cross entropy each do?
3. Why must checkpoint selection use validation labels rather than official test labels?
