"""Problem 1 champion — fine-tune a token classifier on SkillSpan.

Usage:
  python -m src.p1_extract.train_tagger --column skill --seed 42
  python -m src.p1_extract.train_tagger --column knowledge --seed 42
Options: --base-model, --epochs, --lr, --batch-size, --max-length, --subsample N (smoke tests), --out DIR

Outputs: models/p1_tagger/<column>_seed<seed>/ (best checkpoint by dev strict F1), train_summary.json,
and predictions for dev + test under results/p1/preds/tagger_<column>_seed<seed>_<split>.jsonl
(written with chunked inference so every token of long sentences gets a label)."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from ..utils.config import ROOT, cpu_threads, git_commit, models_cfg, pick_device
from ..utils.io import write_json
from .data import LABEL_SPACES, columns_for_scheme, encode_batch, label_scheme_from_eda, records_with_labels


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    cfg = models_cfg()["p1_extract"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--column", default=None, help="skill | knowledge | merged (default: from EDA decision)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base-model", default=str(ROOT / cfg["base_model"]))
    ap.add_argument("--epochs", type=float, default=cfg["num_train_epochs"])
    ap.add_argument("--lr", type=float, default=cfg["learning_rate"])
    ap.add_argument("--batch-size", type=int, default=cfg["per_device_train_batch_size"])
    ap.add_argument("--max-length", type=int, default=cfg["max_length"])
    ap.add_argument("--subsample", type=int, default=0, help="use only N train sentences (smoke test)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="tagger", help="name prefix for prediction files")
    args = ap.parse_args()

    from datasets import Dataset
    from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer, TrainingArguments
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

    torch.set_num_threads(cpu_threads())
    set_seed(args.seed)
    columns = [args.column] if args.column else columns_for_scheme(label_scheme_from_eda())
    device = pick_device()

    for column in columns:
        labels = LABEL_SPACES[column]
        label2id = {l: i for i, l in enumerate(labels)}
        id2label = {i: l for l, i in label2id.items()}
        out_dir = Path(args.out) if args.out else ROOT / "models" / "p1_tagger" / f"{column}_seed{args.seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        train = records_with_labels("train", column)
        dev = records_with_labels("dev", column)
        test = records_with_labels("test", column)
        if args.subsample:
            random.Random(args.seed).shuffle(train)
            train, dev, test = train[: args.subsample], dev[: max(50, args.subsample // 4)], test[: max(50, args.subsample // 4)]

        tok = AutoTokenizer.from_pretrained(args.base_model)
        model = AutoModelForTokenClassification.from_pretrained(args.base_model, num_labels=len(labels), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)

        def to_ds(rows):
            ds = Dataset.from_dict({"tokens": [r["tokens"] for r in rows], "labels": [r["labels"] for r in rows]})
            return ds.map(lambda b: encode_batch(tok, b, label2id, args.max_length), batched=True, remove_columns=["tokens", "labels"])

        ds_train, ds_dev = to_ds(train), to_ds(dev)

        def compute_metrics(p):
            preds = np.argmax(p.predictions, axis=-1)
            y_true, y_pred = [], []
            for pr, lb in zip(preds, p.label_ids):
                t, q = [], []
                for pi, li in zip(pr, lb):
                    if li == -100:
                        continue
                    t.append(id2label[int(li)])
                    q.append(id2label[int(pi)])
                y_true.append(t)
                y_pred.append(q)
            return {"precision": precision_score(y_true, y_pred), "recall": recall_score(y_true, y_pred), "f1": f1_score(y_true, y_pred)}

        import math
        total_steps = math.ceil(len(ds_train) / args.batch_size) * args.epochs
        targs = TrainingArguments(
            output_dir=str(out_dir / "ckpt"), seed=args.seed, learning_rate=args.lr, num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=64, warmup_steps=int(cfg["warmup_ratio"] * total_steps),
            weight_decay=cfg["weight_decay"], eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
            metric_for_best_model="f1", greater_is_better=True, save_total_limit=1, logging_steps=25, report_to="none",
            dataloader_num_workers=0, use_cpu=(device == "cpu"), fp16=False,
        )
        trainer = Trainer(model=model, args=targs, train_dataset=ds_train, eval_dataset=ds_dev, processing_class=tok,
                          data_collator=DataCollatorForTokenClassification(tok), compute_metrics=compute_metrics)
        trainer.train()
        train_time = time.time() - t0
        dev_metrics = trainer.evaluate()
        model.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)
        # drop the checkpoint folder to save disk (the best model is saved at out_dir)
        import shutil
        shutil.rmtree(out_dir / "ckpt", ignore_errors=True)

        history = [h for h in trainer.state.log_history if "eval_f1" in h]
        summary = {
            "column": column, "labels": labels, "seed": args.seed, "base_model": args.base_model, "device": device,
            "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size, "max_length": args.max_length,
            "n_train": len(train), "n_dev": len(dev), "subsample": args.subsample, "train_time_s": round(train_time, 1),
            "dev_best": {k: round(float(v), 4) for k, v in dev_metrics.items() if k.startswith("eval_")},
            "dev_history": [{k: round(float(v), 4) for k, v in h.items() if k in ("epoch", "eval_f1", "eval_precision", "eval_recall", "eval_loss")} for h in history],
            "git_commit": git_commit(), "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary["dev_best"]), f"train_time={train_time/60:.1f} min")

        # predictions with chunked inference (every token gets a label)
        from .predict import Tagger

        tagger = Tagger(str(out_dir), device=device)
        for split_name, rows in [("dev", dev), ("test", test)]:
            t1 = time.time()
            preds = tagger.tag_batch([r["tokens"] for r in rows], max_length=args.max_length)
            lat = (time.time() - t1) / max(1, len(rows)) * 1000
            pred_path = ROOT / "results" / "p1" / "preds" / f"{args.tag}_{column}_seed{args.seed}_{split_name}.jsonl"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pred_path, "w") as f:
                for r, p in zip(rows, preds):
                    f.write(json.dumps({"sid": r["sid"], "source": r["source"], "tokens": r["tokens"], "gold": r["labels"], "pred": p}) + "\n")
            write_json({"approach": args.tag, "column": column, "seed": args.seed, "split": split_name, "n": len(rows),
                        "latency_ms_per_sentence": round(lat, 2), "model_dir": str(out_dir), "device": device},
                       f"results/p1/preds/{args.tag}_{column}_seed{args.seed}_{split_name}.meta.json")
            print(f"wrote {pred_path.name} ({lat:.1f} ms/sentence)")


if __name__ == "__main__":
    main()
