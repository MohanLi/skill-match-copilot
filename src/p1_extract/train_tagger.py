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
    from transformers import AutoModelForTokenClassification, AutoTokenizer
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

        import math

        from transformers import get_linear_schedule_with_warmup
        from .data import encode_batch

        def encode(rows):
            enc = encode_batch(tok, {"tokens": [r["tokens"] for r in rows], "labels": [r["labels"] for r in rows]}, label2id, args.max_length)
            return [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i], "labels": enc["labels"][i]} for i in range(len(rows))]

        tr_items, dev_items = encode(train), encode(dev)
        pad_id = tok.pad_token_id or 0

        def collate(items):
            L = max(len(x["input_ids"]) for x in items)
            ids = torch.tensor([x["input_ids"] + [pad_id] * (L - len(x["input_ids"])) for x in items])
            att = torch.tensor([x["attention_mask"] + [0] * (L - len(x["attention_mask"])) for x in items])
            lab = torch.tensor([x["labels"] + [-100] * (L - len(x["labels"])) for x in items])
            return {"input_ids": ids.to(device), "attention_mask": att.to(device), "labels": lab.to(device)}

        def length_bucketed_batches(items, batch_size, rng):
            """Shuffle, then sort within mega-batches by length and shuffle the batch order (= group_by_length)."""
            idx = list(range(len(items)))
            rng.shuffle(idx)
            mega = batch_size * 50
            batches = []
            for m in range(0, len(idx), mega):
                chunk = sorted(idx[m : m + mega], key=lambda i: len(items[i]["input_ids"]))
                batches += [chunk[b : b + batch_size] for b in range(0, len(chunk), batch_size)]
            rng.shuffle(batches)
            return batches

        @torch.no_grad()
        def evaluate(items, rows):
            model.eval()
            order = sorted(range(len(items)), key=lambda i: len(items[i]["input_ids"]))
            preds = [None] * len(items)
            for b in range(0, len(order), 64):
                sel = order[b : b + 64]
                batch = collate([items[i] for i in sel])
                logits = model(**batch).logits.argmax(-1).cpu().tolist()
                for j, i in enumerate(sel):
                    preds[i] = logits[j]
            y_true, y_pred = [], []
            for i, it in enumerate(items):
                t, q = [], []
                for pi, li in zip(preds[i], it["labels"]):
                    if li == -100:
                        continue
                    t.append(id2label[int(li)])
                    q.append(id2label[int(pi)])
                y_true.append(t)
                y_pred.append(q)
            model.train()
            return {"eval_precision": precision_score(y_true, y_pred), "eval_recall": recall_score(y_true, y_pred), "eval_f1": f1_score(y_true, y_pred)}

        model.to(device).train()
        steps_per_epoch = math.ceil(len(tr_items) / args.batch_size)
        total_steps = int(steps_per_epoch * args.epochs)
        no_decay = ("bias", "LayerNorm.weight")
        params = [{"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": cfg["weight_decay"]},
                  {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
        optimizer = torch.optim.AdamW(params, lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg["warmup_ratio"] * total_steps), total_steps)
        rng = random.Random(args.seed)
        history, best_f1, step = [], -1.0, 0
        n_epochs = math.ceil(args.epochs)
        for epoch in range(n_epochs):
            t_ep = time.time()
            run_loss, n_loss = 0.0, 0
            for batch_idx in length_bucketed_batches(tr_items, args.batch_size, rng):
                if step >= total_steps:
                    break
                out = model(**collate([tr_items[i] for i in batch_idx]))
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                run_loss += float(out.loss.detach())
                n_loss += 1
                step += 1
                if step % 25 == 0:
                    print(f"  step {step}/{total_steps} loss {run_loss / n_loss:.4f} lr {scheduler.get_last_lr()[0]:.2e} elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
                    run_loss, n_loss = 0.0, 0
            m = evaluate(dev_items, dev)
            m.update({"epoch": epoch + 1, "epoch_time_s": round(time.time() - t_ep, 1)})
            history.append(m)
            print(f"epoch {epoch + 1}: dev P {m['eval_precision']:.4f} R {m['eval_recall']:.4f} F1 {m['eval_f1']:.4f} ({m['epoch_time_s'] / 60:.1f} min)", flush=True)
            if m["eval_f1"] > best_f1:
                best_f1 = m["eval_f1"]
                model.save_pretrained(out_dir)
                tok.save_pretrained(out_dir)
        train_time = time.time() - t0
        dev_metrics = max(history, key=lambda h: h["eval_f1"])
        summary = {
            "column": column, "labels": labels, "seed": args.seed, "base_model": args.base_model, "device": device,
            "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size, "max_length": args.max_length,
            "n_train": len(train), "n_dev": len(dev), "subsample": args.subsample, "train_time_s": round(train_time, 1),
            "dev_best": {k: round(float(v), 4) for k, v in dev_metrics.items() if k.startswith("eval_")},
            "dev_history": [{k: round(float(v), 4) for k, v in h.items()} for h in history], "best_epoch": int(dev_metrics["epoch"]),
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
