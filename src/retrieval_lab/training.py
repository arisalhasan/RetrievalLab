"""Small, explicit CPU fine-tuning experiment; no test-label access or automatic tuning."""

import argparse
import json
import math
import random
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean

from retrieval_lab.bm25 import BM25Index
from retrieval_lab.data import DatasetError, load_dataset, sha256_file
from retrieval_lab.files import install_directory
from retrieval_lab.metrics import aggregate, check_reference, score_query
from retrieval_lab.profiling import ResourceMonitor
from retrieval_lab.provenance import environment, source_hashes
from retrieval_lab.reranking import RerankerConfig, rerank_candidates
from retrieval_lab.splits import load_split

CHECKPOINT_FILES = {
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
}


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 2
    batch_size: int = 8
    learning_rate: float = 0.00002
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    hard_negatives: int = 1
    random_negatives: int = 1

    def __post_init__(self):
        for name in ["seed", "epochs", "batch_size", "hard_negatives", "random_negatives"]:
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"Invalid training setting: {name}")
        if self.epochs > 5 or self.batch_size > 32:
            raise ValueError("This small CPU recipe supports at most 5 epochs and batch size 32")
        for name in ["learning_rate", "weight_decay", "max_grad_norm"]:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid training setting: {name}")


def prepare_pairs(dataset, split: dict, config: TrainingConfig) -> list[dict]:
    if dataset.split != "train" or split["source_split"] != "train":
        raise DatasetError("Training accepts official train-derived IDs only")
    train = split["train_query_ids"]
    if set(train) & set(split["validation_query_ids"]) or not train:
        raise DatasetError("Training/validation query IDs overlap or training is empty")
    index = BM25Index(list(dataset.corpus.values()))
    rng, pairs = random.Random(config.seed), []
    for qid in sorted(train):
        # Only this TRAINING query's known positives are inspected. Never validation/test labels.
        positives = sorted(d for d, grade in dataset.qrels[qid].items() if grade > 0)
        if not positives:
            raise DatasetError("Every training query requires a known positive")
        blocked = set(positives) | {qid}
        hard = [
            h.document.doc_id
            for h in index.search(dataset.queries[qid], len(dataset.corpus))
            if h.document.doc_id not in blocked
        ][: config.hard_negatives]
        pool = sorted(set(dataset.corpus) - blocked - set(hard))
        if len(hard) != config.hard_negatives or len(pool) < config.random_negatives:
            raise DatasetError("Insufficient distinct negatives for this query")
        random_ids = rng.sample(pool, config.random_negatives)
        for label, kind, ids in [
            (1, "official_positive", positives),
            (0, "bm25_unjudged_assumed_negative", hard),
            (0, "random_unjudged_assumed_negative", random_ids),
        ]:
            pairs.extend(
                {"query_id": qid, "doc_id": did, "label": label, "source": kind} for did in ids
            )
    return pairs


def verify_checkpoint(folder: Path, config: RerankerConfig) -> dict:
    try:
        manifest = json.loads((folder / "retrievallab-model.json").read_text(encoding="utf-8"))
        hashes = manifest["file_sha256"]
        if set(hashes) != CHECKPOINT_FILES or any(
            sha256_file(folder / n) != h for n, h in hashes.items()
        ):
            raise DatasetError("Checkpoint file checksum mismatch")
        if hashes["model.safetensors"] != config.checkpoint_sha256:
            raise DatasetError("Checkpoint weights differ from configured SHA-256")
        if (manifest["model_id"], manifest["revision"]) != (config.model_id, config.revision):
            raise DatasetError("Checkpoint parent identity mismatch")
        return manifest
    except (OSError, KeyError, TypeError) as exc:
        raise DatasetError("Verified trained checkpoint unavailable") from exc


def select_epoch(rows: list[dict]) -> dict:
    if not rows or any(not all(math.isfinite(row[k]) for k in ["ndcg", "recall"]) for row in rows):
        raise ValueError("Checkpoint selection requires finite validation scores")
    return max(rows, key=lambda r: (r["ndcg"], r["recall"], -r["epoch"]))


def train(
    data_dir: Path,
    split_path: Path,
    parent_path: Path,
    config_path: Path,
    output: Path,
    model_root: Path = Path("models"),
    cache_root: Path = Path("indexes"),
):
    if output.exists():
        raise FileExistsError("Training run exists; choose a new output directory")
    config = TrainingConfig(**json.loads(config_path.read_text(encoding="utf-8")))
    parent = RerankerConfig.from_file(parent_path)
    if parent.checkpoint_sha256:
        raise ValueError("This recipe starts from the pinned pretrained parent only")
    dataset = load_dataset(data_dir, "train")
    split = load_split(split_path, dataset)
    pairs = prepare_pairs(dataset, split, config)
    output.mkdir(parents=True)

    def write(name, payload):
        (output / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    write("pairs.json", pairs)
    write(
        "plan.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "training": asdict(config),
            "parent": parent.to_dict(),
            "split": split,
            "pairs_sha256": sha256_file(output / "pairs.json"),
            "source_file_sha256": source_hashes(),
            "selection": "highest full-validation nDCG@10, then Recall@10, then earliest epoch; "
            "pretrained epoch 0 eligible; no test access",
            "negative_caveat": "Unjudged negatives may contain unlabeled relevant evidence",
            "environment": environment(),
        },
    )
    import torch

    from retrieval_lab.cross_encoder import CrossEncoderScorer
    from retrieval_lab.hybrid import build_hybrid

    with ResourceMonitor(10) as monitor:
        scorer = CrossEncoderScorer(replace(parent, seed=config.seed), model_root)
        generator = build_hybrid(
            list(dataset.corpus.values()), parent.retriever, model_root, cache_root
        )
        # Validation candidates are label-free and fixed across epochs. They never enter training.
        candidates = {
            qid: generator.search(dataset.queries[qid], parent.rerank_depth)
            for qid in split["validation_query_ids"]
        }

        def validate(epoch):
            scorer.model.eval()
            scores = {}
            rankings = {}
            for qid, hits in candidates.items():
                logits = scorer.score(
                    [(dataset.queries[qid], f"{h.document.title}\n{h.document.text}") for h in hits]
                )
                ranked, _ = rerank_candidates(hits, logits, 10)
                rankings[qid] = [h.document.doc_id for h in ranked if h.document.doc_id != qid]
                scores[qid] = score_query(
                    qid,
                    [h.document.doc_id for h in ranked if h.document.doc_id != qid],
                    dataset.qrels[qid],
                    10,
                )
            reference = check_reference({q: dataset.qrels[q] for q in scores}, rankings, scores, 10)
            write(
                f"validation-epoch-{epoch}.json",
                {"scores": scores, "rankings": rankings, "reference_check": reference},
            )
            return {"epoch": epoch, **aggregate(scores), "reference_check": reference}

        history = [validate(0)]
        print(json.dumps({"event": "validation", **history[-1]}), flush=True)
        optimizer = torch.optim.AdamW(
            scorer.model.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            foreach=False,
        )
        loss_fn = torch.nn.BCEWithLogitsLoss()
        rng = random.Random(config.seed)
        for epoch in range(1, config.epochs + 1):
            order = list(range(len(pairs)))
            rng.shuffle(order)
            scorer.model.model.train()
            losses = []
            for start in range(0, len(order), config.batch_size):
                batch = [pairs[i] for i in order[start : start + config.batch_size]]
                queries = [dataset.queries[p["query_id"]] for p in batch]
                docs = [
                    f"{dataset.corpus[p['doc_id']].title}\n{dataset.corpus[p['doc_id']].text}"
                    for p in batch
                ]
                features = scorer.model.tokenizer(
                    queries,
                    docs,
                    padding=True,
                    truncation="longest_first",
                    max_length=parent.max_length,
                    return_tensors="pt",
                )
                labels = torch.tensor([p["label"] for p in batch], dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                logits = scorer.model.model(**features).logits.reshape(-1)
                loss = loss_fn(logits, labels)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    scorer.model.model.parameters(), config.max_grad_norm
                )
                optimizer.step()
                losses.append(float(loss.detach()))
                if start % (config.batch_size * 50) == 0:
                    print(
                        json.dumps(
                            {
                                "event": "training",
                                "epoch": epoch,
                                "pairs_seen": start,
                                "mean_batch_loss": fmean(losses),
                            }
                        ),
                        flush=True,
                    )
            scorer.model.eval()
            target_root = model_root / "checkpoints"
            target_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=target_root) as temp:
                staged = Path(temp) / "model"
                scorer.model.model.save_pretrained(staged, safe_serialization=True)
                scorer.model.tokenizer.save_pretrained(staged)
                if {p.name for p in staged.iterdir()} != CHECKPOINT_FILES:
                    raise DatasetError("Unexpected saved checkpoint files")
                hashes = {n: sha256_file(staged / n) for n in sorted(CHECKPOINT_FILES)}
                identity = hashes["model.safetensors"]
                manifest = {
                    "model_id": parent.model_id,
                    "revision": parent.revision,
                    "file_sha256": hashes,
                    "epoch": epoch,
                    "training": asdict(config),
                    "training_pairs_sha256": sha256_file(output / "pairs.json"),
                    "split_sha256": split["split_sha256"],
                }
                (staged / "retrievallab-model.json").write_text(
                    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
                )
                target = target_root / identity
                if target.exists():
                    existing = verify_checkpoint(
                        target, replace(parent, checkpoint_sha256=identity)
                    )
                    if existing != manifest:
                        raise DatasetError("Existing checkpoint provenance conflicts")
                else:
                    install_directory(staged, target, target_root)
            row = {
                **validate(epoch),
                "checkpoint_sha256": identity,
                "mean_batch_loss": fmean(losses),
            }
            history.append(row)
            write("history.json", history)
            print(json.dumps({"event": "validation", **row}), flush=True)
    selected = select_epoch(history)
    best_trained = select_epoch(history[1:])
    write(
        "best-trained-config.json",
        replace(parent, checkpoint_sha256=best_trained["checkpoint_sha256"]).to_dict(),
    )
    selected_config = replace(parent, checkpoint_sha256=selected.get("checkpoint_sha256"))
    write("selected-config.json", selected_config.to_dict())
    result = {
        "status": "complete",
        "history": history,
        "selected_epoch": selected["epoch"],
        "selected": selected,
        "best_trained": best_trained,
        "resources": monitor.result,
        "pair_count": len(pairs),
        "training_query_count": len(split["train_query_ids"]),
        "validation_query_count": len(candidates),
        "test_labels_read": False,
        "selected_config_sha256": sha256_file(output / "selected-config.json"),
    }
    write("result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/scifact"))
    parser.add_argument("--split", type=Path, default=Path("configs/splits/scifact-train-v1.json"))
    parser.add_argument("--parent", type=Path, default=Path("configs/reranker.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/training.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            train(args.data_dir, args.split, args.parent, args.config, args.output), indent=2
        )
    )


if __name__ == "__main__":
    main()
