"""Problem 2 champion — fine-tune a sentence-transformers bi-encoder on TechWolf's synthetic
(sentence, ESCO skill) pairs with MultipleNegativesRankingLoss.

Usage: python -m src.p2_normalize.train_biencoder [--epochs 1] [--batch-size 64] [--subsample N] [--out models/p2_biencoder]
Writes models/p2_biencoder/ + train_summary.json. Evaluate with: python -m src.p2_normalize.run_workrb --split val --models ours"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
import time
from pathlib import Path

import pandas as pd

from ..data import esco
from ..utils.config import ROOT, cpu_threads, data_cfg, git_commit, models_cfg, pick_device


def load_synthetic_pairs() -> pd.DataFrame:
    d = ROOT / data_cfg()["datasets"]["synthetic_esco"]["local"]
    files = sorted(glob.glob(str(d / "**" / "*.parquet"), recursive=True)) or sorted(glob.glob(str(d / "**" / "*.csv"), recursive=True)) or sorted(glob.glob(str(d / "**" / "*.json*"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no data files under {d}; run scripts/download_assets.py on the Mac first")
    frames = []
    for f in files:
        if f.endswith(".parquet"):
            frames.append(pd.read_parquet(f))
        elif f.endswith(".csv"):
            frames.append(pd.read_csv(f))
        else:
            frames.append(pd.read_json(f, lines=f.endswith(".jsonl")))
    df = pd.concat(frames, ignore_index=True)
    return df


def detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Return (sentence_col, skill_col): the column whose values mostly belong to the ESCO vocabulary is the skill."""
    vocab = set(esco.vocabulary())
    best, best_share = None, 0.0
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith("str"):
            share = df[c].astype(str).head(2000).isin(vocab).mean()
            if share > best_share:
                best, best_share = c, share
    text_cols = [c for c in df.columns if c != best and (df[c].dtype == object or str(df[c].dtype).startswith("str"))]
    sent = max(text_cols, key=lambda c: df[c].astype(str).str.len().mean())
    return sent, best


def main() -> None:
    cfg = models_cfg()["p2_normalize"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / cfg["base_encoder"]))
    ap.add_argument("--epochs", type=float, default=cfg["num_train_epochs"])
    ap.add_argument("--batch-size", type=int, default=cfg["per_device_train_batch_size"])
    ap.add_argument("--lr", type=float, default=cfg["learning_rate"])
    ap.add_argument("--max-seq-length", type=int, default=cfg["max_seq_length"])
    ap.add_argument("--val-fraction", type=float, default=cfg["val_fraction"])
    ap.add_argument("--subsample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "models" / "p2_biencoder"))
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, losses
    from sentence_transformers.trainer import SentenceTransformerTrainer
    from sentence_transformers.training_args import BatchSamplers, SentenceTransformerTrainingArguments

    torch.set_num_threads(cpu_threads())
    random.seed(args.seed)
    df = load_synthetic_pairs()
    sent_col, skill_col = detect_columns(df)
    df = df[[sent_col, skill_col]].dropna().rename(columns={sent_col: "sentence", skill_col: "skill"})
    df["sentence"], df["skill"] = df["sentence"].astype(str).str.strip(), df["skill"].astype(str).str.strip()
    df = df[(df["sentence"].str.len() > 0) & (df["skill"].str.len() > 0)].drop_duplicates().reset_index(drop=True)
    n_total = len(df)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_val = int(len(df) * args.val_fraction)
    val, train = df.iloc[:n_val], df.iloc[n_val:]
    if args.subsample:
        train = train.head(args.subsample)
    print(f"pairs: {n_total:,} total → train {len(train):,} / val {len(val):,}; columns: sentence={sent_col!r}, skill={skill_col!r}; distinct skills in train: {train['skill'].nunique():,}")

    model = SentenceTransformer(args.base, device=pick_device())
    model.max_seq_length = args.max_seq_length
    train_ds = Dataset.from_pandas(train[["sentence", "skill"]], preserve_index=False)
    val_ds = Dataset.from_pandas(val[["sentence", "skill"]].head(2000), preserve_index=False)
    steps = math.ceil(len(train_ds) / args.batch_size) * args.epochs
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(Path(args.out) / "ckpt"), num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=128, learning_rate=args.lr, warmup_steps=int(cfg["warmup_ratio"] * steps), seed=args.seed,
        batch_sampler=BatchSamplers.NO_DUPLICATES, eval_strategy="steps", eval_steps=max(50, int(steps // 4)), save_strategy="no",
        logging_steps=50, report_to="none", fp16=False, use_cpu=(pick_device() == "cpu"), dataloader_num_workers=0,
    )
    loss = losses.MultipleNegativesRankingLoss(model)
    trainer = SentenceTransformerTrainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds, loss=loss)
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    model.save(args.out)
    hist = [h for h in trainer.state.log_history if "eval_loss" in h or "loss" in h]
    summary = {
        "base": args.base, "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "max_seq_length": args.max_seq_length,
        "n_pairs_total": n_total, "n_train": len(train_ds), "n_val": len(val), "distinct_skills_train": int(train["skill"].nunique()),
        "loss": "MultipleNegativesRankingLoss", "batch_sampler": "NO_DUPLICATES", "seed": args.seed, "device": pick_device(),
        "train_time_s": round(train_time, 1), "log_history": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in h.items() if k in ("step", "epoch", "loss", "eval_loss")} for h in hist],
        "git_commit": git_commit(), "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (Path(args.out) / "train_summary.json").write_text(json.dumps(summary, indent=2))
    import shutil
    shutil.rmtree(Path(args.out) / "ckpt", ignore_errors=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "log_history"}, indent=1))


if __name__ == "__main__":
    main()
