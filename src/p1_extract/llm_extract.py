"""Problem 1 challenger — few-shot LLM extraction (prompt prompts/p1_extract_v1.yaml).

For every sentence the model returns {"skills": [...], "knowledge": [...]}; each phrase is mapped back
to token positions by exact (case-insensitive) match. Phrases not found in the sentence are counted as
HALLUCINATIONS and dropped. Every call is cached + logged (tokens, latency, cost).

Usage: python -m src.p1_extract.llm_extract --split test [--subset 1000] [--model bulk] [--workers 4]
Outputs: results/p1/preds/llm_<tag>_{skill,knowledge}_seed0_<split>.jsonl (+ .meta.json),
         results/p1/llm_<tag>_<split>_calls.jsonl, results/p1/llm_<tag>_<split>_summary.json"""
from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from ..data import skillspan as ss
from ..utils.config import ROOT, models_cfg
from ..utils.io import write_json
from ..utils.llm import LLMClient, load_prompt
from ..utils.text import phrases_to_bio


def format_examples(prompt) -> str:
    lines = []
    for ex in prompt.examples:
        lines.append(f"Sentence: {ex['sentence']}\nJSON: " + json.dumps({"skills": ex["skills"], "knowledge": ex["knowledge"]}, ensure_ascii=False))
    return "\n\n".join(lines)


def extract_one(client: LLMClient, prompt, examples_text: str, r: dict, model: str) -> dict:
    sentence = " ".join(r["tokens"])
    t0 = time.perf_counter()
    try:
        obj, res = client.complete_json(prompt, model=model, max_tokens=1200, examples=examples_text, sentence=sentence)
        skills = [str(x) for x in (obj.get("skills") or []) if isinstance(x, (str, int, float))]
        knowledge = [str(x) for x in (obj.get("knowledge") or []) if isinstance(x, (str, int, float))]
        parse_ok = True
    except Exception as e:  # noqa: BLE001
        skills, knowledge, parse_ok, res = [], [], False, None
        err = f"{type(e).__name__}: {str(e)[:120]}"
    tags_sk, un_sk = phrases_to_bio(r["tokens"], skills, "SKILL")
    tags_kn, un_kn = phrases_to_bio(r["tokens"], knowledge, "KNOW")
    rec = {
        "sid": r["sid"], "source": r["source"], "sentence": sentence, "skills": skills, "knowledge": knowledge,
        "unmatched_skills": un_sk, "unmatched_knowledge": un_kn, "parse_ok": parse_ok,
        "tags_skill": tags_sk, "tags_knowledge": tags_kn,
        "input_tokens": res.input_tokens if res else 0, "output_tokens": res.output_tokens if res else 0,
        "latency_ms": res.latency_ms if res else None, "cost_usd": res.cost_usd if res else 0.0, "cached": res.cached if res else None,
        "model": res.model if res else None, "prompt_version": prompt.tag, "wall_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    if not parse_ok:
        rec["error"] = err
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--subset", type=int, default=models_cfg()["p1_extract"].get("llm_subset_size") or 0)
    ap.add_argument("--model", default="bulk")
    ap.add_argument("--tag", default=None, help="name used in file names (default: model family, e.g. haiku)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N sentences")
    args = ap.parse_args()

    client = LLMClient(task=f"p1_llm_extract_{args.split}")
    model_id = client.resolve_model(args.model)
    tag = args.tag or next((f for f in ("haiku", "sonnet", "opus") if f in model_id), model_id.replace("/", "_"))
    prompt = load_prompt("p1_extract_v1")
    examples_text = format_examples(prompt)

    rows = ss.load_split(args.split)
    if args.subset and args.subset < len(rows):
        rng = random.Random(42)
        rows = sorted(rng.sample(rows, args.subset), key=lambda r: int(r["sid"].split("-")[1]))
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} sentences, model={model_id}, prompt={prompt.tag}, budget={client.budget_status()}")

    results: dict[str, dict] = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(extract_one, client, prompt, examples_text, r, args.model): r["sid"] for r in rows}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            results[rec["sid"]] = rec
            if i % 200 == 0 or i == len(rows):
                spent = sum(x["cost_usd"] for x in results.values())
                print(f"  {i}/{len(rows)}  spent so far ${spent:.3f}  elapsed {(time.time()-t0)/60:.1f} min", flush=True)

    recs = [results[r["sid"]] for r in rows]
    out_dir = ROOT / "results" / "p1"
    (out_dir / "preds").mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"llm_{tag}_{args.split}_calls.jsonl", "w") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    for column, key in [("skill", "tags_skill"), ("knowledge", "tags_knowledge")]:
        with open(out_dir / "preds" / f"llm_{tag}_{column}_seed0_{args.split}.jsonl", "w") as f:
            for r, rec in zip(rows, recs):
                f.write(json.dumps({"sid": r["sid"], "source": r["source"], "tokens": r["tokens"], "gold": ss.record_labels(r, column), "pred": rec[key]}) + "\n")

    n_phr = sum(len(x["skills"]) + len(x["knowledge"]) for x in recs)
    n_un = sum(len(x["unmatched_skills"]) + len(x["unmatched_knowledge"]) for x in recs)
    lat = [x["latency_ms"] for x in recs if x["latency_ms"]]
    cost = sum(x["cost_usd"] for x in recs)
    summary = {
        "approach": f"llm_{tag}", "model": model_id, "prompt_version": prompt.tag, "split": args.split, "n_sentences": len(rows), "subset": args.subset,
        "parse_failures": sum(not x["parse_ok"] for x in recs), "phrases_returned": n_phr, "phrases_unmatched": n_un,
        "hallucination_rate": round(n_un / n_phr, 4) if n_phr else 0.0,
        "hallucination_rate_skills": round(sum(len(x["unmatched_skills"]) for x in recs) / max(1, sum(len(x["skills"]) for x in recs)), 4),
        "hallucination_rate_knowledge": round(sum(len(x["unmatched_knowledge"]) for x in recs) / max(1, sum(len(x["knowledge"]) for x in recs)), 4),
        "input_tokens": sum(x["input_tokens"] for x in recs), "output_tokens": sum(x["output_tokens"] for x in recs),
        "cost_usd": round(cost, 4), "cost_usd_per_1000_sentences": round(1000 * cost / len(rows), 4),
        "latency_ms_median": float(np.median(lat)) if lat else None, "latency_ms_p90": float(np.percentile(lat, 90)) if lat else None,
        "cached_calls": sum(1 for x in recs if x["cached"]), "wall_time_min": round((time.time() - t0) / 60, 2),
    }
    write_json(summary, f"results/p1/llm_{tag}_{args.split}_summary.json")
    for column in ("skill", "knowledge"):
        write_json({"approach": f"llm_{tag}", "column": column, "seed": 0, "split": args.split, "n": len(rows), "model": model_id, "prompt_version": prompt.tag,
                    "latency_ms_per_sentence": summary["latency_ms_median"], "cost_usd_per_1000_sentences": summary["cost_usd_per_1000_sentences"],
                    "hallucination_rate": summary[f"hallucination_rate_{'skills' if column=='skill' else 'knowledge'}"]},
                   f"results/p1/preds/llm_{tag}_{column}_seed0_{args.split}.meta.json")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
