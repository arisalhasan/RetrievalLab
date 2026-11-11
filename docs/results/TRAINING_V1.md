# Fine-tuning experiment v1

Actual Windows CPU execution, seed 42, two epochs, 647 training claims and 162 validation claims. Training used 2,024 explicit pairs, batch size 8, 512-token maximum joint inputs, AdamW learning rate 0.00002 and BCE-with-logits loss. No official test labels were read.

| Checkpoint | Recall@10 | nDCG@10 | Queries missing all evidence |
| --- | ---: | ---: | ---: |
| Original pretrained parent, epoch 0 | 0.819033 | 0.703273 | 26 |
| Epoch 1 | 0.814403 | 0.703021 | 27 |
| Epoch 2, selected | 0.820576 | 0.712615 | 26 |

The declared rule selected epoch 2. All three full-validation metric calculations passed the independent pytrec_eval check (maximum absolute error about 1.1e-16). The validation nDCG increase over the parent is about 0.00934; a point-score increase alone does not prove a reliable improvement. The existing depth-10 hybrid has higher exploratory validation nDCG (0.729131), and remains eligible as the final release choice. No learning-rate, epoch-count, or negative-sampling setting was changed after these results.

The independently loaded selected checkpoint's full validation benchmark reproduced **exactly the same aggregate quality** as the epoch-2 selection pass. Its paired nDCG delta versus the parent is **+0.009342**, with 95% grouped-bootstrap interval **[-0.019866, +0.039041]**, which includes zero. The Recall delta is **+0.001543**, interval **[0.000000, +0.004934]**. These are exploratory development comparisons, not a claim of a reliable generalization gain. Full evidence is in `runs/m7-trained-validation-v1` and `runs/m7-comparison-pretrained-vs-trained.json`.

That validation benchmark measured 160 fresh calls on the same 32 seed-selected timing IDs used by M5: p50 **1,654.37 ms**, p95 **3,206.50 ms**. Background development work and environment snapshots differ from historical M5 runs, so these timings do not establish a training-induced speed change. Both checkpoints have the same architecture and candidate budget. The final test uses a new common timing profile and isolated per-system processes.

The recorded monitor measured **964.51 seconds (16.08 minutes)**, **964.52 process CPU seconds**, and **958,492,672 bytes (914.09 MiB)** sampled peak whole-process RSS. Time includes label-free validation candidate preparation, three full validation scoring passes, both training epochs and checkpoint writes. Other development checks ran on the workstation during this interval; this is an observed experiment cost, not a controlled training-speed benchmark. No CUDA/GPU measurement was executed.

Epoch 1 mean batch loss was 0.770675; epoch 2 was 0.514955. Lower training loss is not the model-selection criterion. Different batch padding and dropout make loss averages descriptive. Model selection uses full ranking metrics on validation only.

Selected checkpoint weight SHA-256:

`363baae60f7a89c5baf54c995c222134a56c0872bdca44f39dbd437e3396451d`

Its parent is the previously pinned MS MARCO TinyBERT revision `81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc`. All model/tokenizer assets have saved checksums. Inspect `runs/m7-training-v1/plan.json`, `pairs.json`, the three `validation-epoch-*.json` files, `history.json`, and `result.json` for exact IDs, scores, versions, seeds, source hashes, and resources.

The training plan was recorded before final package metadata was changed from the development version 0.5 to release 1.0; training/scoring code and neural library versions were unchanged by that packaging update. The later benchmark and release freeze record their own complete environment snapshots. No same-environment timing claim is made across historical validation runs.

Known limits: unjudged negatives may include relevant evidence; the same validation partition has been explored across milestones; only one seed and one small recipe were trained. The two-epoch SciFact experiment has not been independently repeated end to end. A real-model synthetic integration test does separately verify gradient updates, safetensors saving, checksum verification and checkpoint reload; its scores are not SciFact benchmark results.
