#!/usr/bin/env python3
"""
download_assets.py  —  run this ON YOUR MAC (where huggingface.co is reachable).

It is idempotent: re-run it if it gets interrupted and it continues where it stopped.

What it does
  1. Downloads the Hugging Face datasets      -> data/raw/hf/<org>__<name>/
  2. Downloads the base models                -> models/base/<name>/
  3. Downloads ESCO v1.1.0 (English CSV)      -> data/raw/esco/1.1.0/en/
  4. Pulls ONE daily snapshot of open-apply-jobs, keeps the data/AI/ML postings
     plus a 20k random sample and summary stats -> data/raw/live_jobs/
  5. Packs everything into data/transfer/assets.tar.partNN (<= 240 MB each) + manifest.json
     so the bundle can be moved into the cloud workspace in a few steps.

Usage (from the repo root, inside the download venv):
  python scripts/download_assets.py                 # everything
  python scripts/download_assets.py --skip-live     # skip step 4
  python scripts/download_assets.py --with-deberta  # also fetch microsoft/deberta-v3-base
  python scripts/download_assets.py --only pack     # only rebuild the transfer bundle
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
import time
import urllib.parse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
HF_DIR = RAW / "hf"
ESCO_DIR = RAW / "esco"
LIVE_DIR = RAW / "live_jobs"
MODELS_DIR = ROOT / "models" / "base"
TRANSFER_DIR = ROOT / "data" / "transfer"

CHUNK_BYTES = 240 * 1024 * 1024  # 240 MB parts (the workspace accepts <= 400 MB per file)

DATASETS = [
    "jjzha/skillspan",
    "TechWolf/Synthetic-ESCO-skill-sentences",
    "TechWolf/skill-extraction-tech",
    "TechWolf/skill-extraction-house",
    "TechWolf/skill-extraction-techwolf",
    "lang-uk/recruitment-dataset-job-descriptions-english",
    "lang-uk/recruitment-dataset-candidate-profiles-english",
]

MODELS = [
    "jjzha/jobbert-base-cased",
    "sentence-transformers/all-MiniLM-L6-v2",
    "TechWolf/ConTeXT-Skill-Extraction-base",
]
OPTIONAL_MODELS = ["microsoft/deberta-v3-base"]

MODEL_IGNORE = [
    "onnx/*", "openvino/*", "*.onnx", "*.msgpack", "*.h5", "*.ot", "*.tflite",
    "rust_model.ot", "tf_model.h5", "flax_model.msgpack", "coreml/*", "*.ckpt",
]

LIVE_REPO = "edwarddgao/open-apply-jobs"
LIVE_TITLE_REGEX = (
    r"machine learning|data scien|ai engineer|ml engineer|mlops|data engineer|"
    r"nlp|natural language|analytics engineer|data analyst|artificial intelligence|"
    r"deep learning|llm|generative ai|gen ai|computer vision|research scientist|"
    r"applied scientist|ai/ml|ml scientist|ai platform|ai product|data platform"
)
LIVE_MAX_DOWNLOAD_BYTES = 3 * 1024**3  # above this, query the snapshot remotely with DuckDB instead


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def human(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


# --------------------------------------------------------------------------- HF
def hf_local_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def download_datasets() -> None:
    from huggingface_hub import snapshot_download

    HF_DIR.mkdir(parents=True, exist_ok=True)
    for repo in DATASETS:
        dest = HF_DIR / hf_local_name(repo)
        log(f"dataset {repo} -> {dest.relative_to(ROOT)}")
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(dest), max_workers=4)
        log(f"   done ({human(dir_size(dest))})")


def download_models(with_deberta: bool) -> None:
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for repo in MODELS + (OPTIONAL_MODELS if with_deberta else []):
        dest = MODELS_DIR / repo.split("/")[-1]
        files = api.list_repo_files(repo)
        ignore = list(MODEL_IGNORE)
        has_safetensors = any(f.endswith(".safetensors") for f in files)
        if has_safetensors:
            ignore += ["*.bin"]  # safetensors is enough; skip the duplicate pytorch_model.bin
        log(f"model {repo} -> {dest.relative_to(ROOT)}  (safetensors={has_safetensors})")
        snapshot_download(repo_id=repo, local_dir=str(dest), ignore_patterns=ignore, max_workers=4)
        log(f"   done ({human(dir_size(dest))})")


# ------------------------------------------------------------------------- ESCO
ESCO_VERSION = "1.1.0"


def download_esco() -> None:
    import requests

    out_dir = ESCO_DIR / ESCO_VERSION / "en"
    if (out_dir / "skills_en.csv").exists():
        log(f"ESCO {ESCO_VERSION} already present at {out_dir.relative_to(ROOT)}")
        return
    dl_dir = ESCO_DIR / "downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dl_dir / f"ESCO_dataset_v{ESCO_VERSION}_classification_en_csv.zip"
    if not zip_path.exists() or zip_path.stat().st_size < 1_000_000:
        filename = f"ESCO dataset - v{ESCO_VERSION} - classification - en - csv.zip"
        url = "https://ec.europa.eu/esco/download/" + urllib.parse.quote(filename)
        log(f"ESCO: downloading {url}")
        try:
            with requests.get(url, stream=True, timeout=120,
                              headers={"User-Agent": "Mozilla/5.0 (skill-match-copilot)"}) as r:
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
        except Exception as e:  # noqa: BLE001
            log(f"ESCO download FAILED ({e}).")
            log("   Manual fallback: open https://esco.ec.europa.eu/en/use-esco/download , choose")
            log(f"   version v{ESCO_VERSION}, 'classification', language 'en', file type 'csv',")
            log(f"   download the zip and save it as: {zip_path}")
            log("   then re-run this script.")
            return
    try:
        with zipfile.ZipFile(zip_path) as z:
            out_dir.mkdir(parents=True, exist_ok=True)
            z.extractall(out_dir)
    except zipfile.BadZipFile:
        log(f"ESCO: {zip_path} is not a valid zip (the site may have returned an HTML page). Delete it and retry, or use the manual fallback above.")
        return
    # Some zips nest files one level down; flatten if needed.
    if not (out_dir / "skills_en.csv").exists():
        for cand in out_dir.rglob("skills_en.csv"):
            for f in cand.parent.iterdir():
                if f.is_file():
                    f.replace(out_dir / f.name)
            break
    ok = all((out_dir / n).exists() for n in ["skills_en.csv", "occupations_en.csv", "occupationSkillRelations_en.csv"])
    log(f"ESCO extracted to {out_dir.relative_to(ROOT)}  (all required CSVs present: {ok})")


# -------------------------------------------------------------------- live jobs
def download_live_jobs() -> None:
    import duckdb
    from huggingface_hub import HfApi, snapshot_download

    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    marker = LIVE_DIR / "snapshot_date.txt"
    if marker.exists() and any(LIVE_DIR.glob("live_jobs_*_ai_data.parquet")):
        log(f"live jobs already present (snapshot {marker.read_text().strip()})")
        return

    api = HfApi()
    files = api.list_repo_files(LIVE_REPO, repo_type="dataset")
    dates = sorted({m.group(1) for f in files if (m := re.search(r"date=(\d{4}-\d{2}-\d{2})", f))})
    if not dates:
        log("live jobs: could not find date= partitions; skipping (EDA 04 will be marked as unavailable)")
        return
    date = dates[-1]
    part_files = [f for f in files if f"date={date}" in f and f.endswith(".parquet")]
    log(f"live jobs: newest snapshot date={date} with {len(part_files)} parquet file(s); {len(dates)} snapshots exist ({dates[0]} .. {dates[-1]})")

    total = 0
    try:
        for entry in api.list_repo_tree(LIVE_REPO, repo_type="dataset", path_in_repo=f"data/date={date}", recursive=True):
            total += getattr(entry, "size", 0) or 0
    except Exception:  # noqa: BLE001
        total = 0
    log(f"live jobs: snapshot size ~{human(total) if total else 'unknown'}")

    con = duckdb.connect()
    if total and total <= LIVE_MAX_DOWNLOAD_BYTES:
        snap_dir = LIVE_DIR / "snapshot"
        snapshot_download(repo_id=LIVE_REPO, repo_type="dataset", local_dir=str(snap_dir),
                          allow_patterns=[f"data/date={date}/*", f"data/date={date}/**"], max_workers=4)
        src = f"read_parquet('{snap_dir.as_posix()}/data/date={date}/**/*.parquet', hive_partitioning=1, union_by_name=true)"
    else:
        log("live jobs: snapshot is large/unknown; querying it remotely via DuckDB httpfs (this can take a while)")
        con.sql("INSTALL httpfs; LOAD httpfs;")
        src = f"read_parquet('hf://datasets/{LIVE_REPO}/data/date={date}/**/*.parquet', hive_partitioning=1, union_by_name=true)"

    cols = [r[0] for r in con.sql(f"DESCRIBE SELECT * FROM {src}").fetchall()]
    log(f"live jobs: columns = {cols}")
    title_col = next((c for c in cols if c.lower() == "title"), None)
    if title_col is None:
        title_col = next((c for c in cols if "title" in c.lower()), cols[0])

    stats: dict = {"snapshot_date": date, "columns": cols}
    stats["n_rows"] = con.sql(f"SELECT count(*) FROM {src}").fetchone()[0]
    for c in cols:
        lc = c.lower()
        if lc in {"source", "ats", "provider", "platform", "board"} or lc.endswith("_source"):
            stats[f"rows_by_{c}"] = {str(k): v for k, v in con.sql(f"SELECT {c}, count(*) FROM {src} GROUP BY 1 ORDER BY 2 DESC").fetchall()}
        if lc in {"remote", "is_remote", "remote_type", "workplace_type", "employment_type", "department"}:
            stats[f"rows_by_{c}"] = {str(k): v for k, v in con.sql(f"SELECT {c}, count(*) FROM {src} GROUP BY 1 ORDER BY 2 DESC LIMIT 50").fetchall()}
        if "salary" in lc or "compensation" in lc:
            stats[f"nonnull_{c}"] = con.sql(f"SELECT count({c}) FROM {src}").fetchone()[0]
    stats["top_titles"] = {str(k): v for k, v in con.sql(f"SELECT {title_col}, count(*) FROM {src} GROUP BY 1 ORDER BY 2 DESC LIMIT 100").fetchall()}
    desc_col = next((c for c in cols if "description" in c.lower()), None)
    if desc_col:
        q = con.sql(f"SELECT quantile_cont(length({desc_col}), [0.05,0.25,0.5,0.75,0.95]) FROM {src}").fetchone()[0]
        stats["description_length_quantiles_chars"] = dict(zip(["p05", "p25", "p50", "p75", "p95"], q))

    filtered = LIVE_DIR / f"live_jobs_{date}_ai_data.parquet"
    con.sql(f"COPY (SELECT * FROM {src} WHERE regexp_matches(lower({title_col}), '{LIVE_TITLE_REGEX}')) TO '{filtered.as_posix()}' (FORMAT PARQUET)")
    n_filtered = con.sql(f"SELECT count(*) FROM read_parquet('{filtered.as_posix()}')").fetchone()[0]
    stats["n_rows_ai_data_filter"] = n_filtered
    sample = LIVE_DIR / f"live_jobs_{date}_sample20k.parquet"
    con.sql(f"COPY (SELECT * FROM {src} USING SAMPLE reservoir(20000 ROWS) REPEATABLE (42)) TO '{sample.as_posix()}' (FORMAT PARQUET)")
    (LIVE_DIR / f"live_jobs_{date}_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    marker.write_text(date)
    log(f"live jobs: {stats['n_rows']:,} postings in snapshot; {n_filtered:,} data/AI/ML postings kept; 20k sample + stats written")


# ------------------------------------------------------------------------ pack
class ChunkedWriter:
    """File-like object that rotates to a new part file every CHUNK_BYTES."""

    def __init__(self, base: Path, chunk_bytes: int):
        self.base, self.chunk_bytes = base, chunk_bytes
        self.idx, self.written, self.fh = -1, 0, None
        self.parts: list[Path] = []
        self._rotate()

    def _rotate(self):
        if self.fh:
            self.fh.close()
        self.idx += 1
        p = self.base.with_name(f"{self.base.name}.part{self.idx:02d}")
        self.fh = open(p, "wb")
        self.parts.append(p)
        self.written = 0

    def write(self, b: bytes) -> int:
        mv = memoryview(b)
        n = 0
        while len(mv):
            room = self.chunk_bytes - self.written
            if room <= 0:
                self._rotate()
                continue
            take = mv[:room]
            self.fh.write(take)
            self.written += len(take)
            n += len(take)
            mv = mv[len(take):]
        return n

    def close(self):
        if self.fh:
            self.fh.close()


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pack_transfer() -> None:
    TRANSFER_DIR.mkdir(parents=True, exist_ok=True)
    for old in TRANSFER_DIR.glob("assets.tar.part*"):
        old.unlink()
    roots = [HF_DIR, ESCO_DIR / ESCO_VERSION, LIVE_DIR, MODELS_DIR]
    roots = [r for r in roots if r.exists()]

    def _filter(ti: tarfile.TarInfo):
        # skip the raw live-jobs snapshot partition (big; only derived files are needed)
        if "/live_jobs/snapshot/" in ("/" + ti.name) or ti.name.endswith("/live_jobs/snapshot"):
            return None
        if "/.cache/" in ("/" + ti.name):  # huggingface_hub bookkeeping
            return None
        return ti

    log("packing transfer bundle (this streams ~2-3 GB; a few minutes) ...")
    writer = ChunkedWriter(TRANSFER_DIR / "assets.tar", CHUNK_BYTES)
    with tarfile.open(fileobj=writer, mode="w|") as tar:
        for r in roots:
            tar.add(str(r), arcname=str(r.relative_to(ROOT)), filter=_filter)
    writer.close()
    parts = [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_of(p)} for p in writer.parts]
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "roots": [str(r.relative_to(ROOT)) for r in roots],
        "parts": parts,
        "total_bytes": sum(p["bytes"] for p in parts),
    }
    (TRANSFER_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"transfer bundle: {len(parts)} part(s), {human(manifest['total_bytes'])} -> {TRANSFER_DIR.relative_to(ROOT)}/")


# ------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["datasets", "models", "esco", "live", "pack"], help="run a single step")
    ap.add_argument("--skip-live", action="store_true", help="skip the open-apply-jobs snapshot")
    ap.add_argument("--with-deberta", action="store_true", help="also download microsoft/deberta-v3-base (~700 MB)")
    args = ap.parse_args()

    steps = {
        "datasets": download_datasets,
        "models": lambda: download_models(args.with_deberta),
        "esco": download_esco,
        "live": download_live_jobs,
        "pack": pack_transfer,
    }
    order = [args.only] if args.only else ["datasets", "models", "esco", "live", "pack"]
    if args.skip_live and "live" in order:
        order.remove("live")

    t0 = time.time()
    for name in order:
        log(f"===== step: {name} =====")
        try:
            steps[name]()
        except KeyboardInterrupt:
            log("interrupted — re-run the same command to resume")
            return 130
        except Exception as e:  # noqa: BLE001
            log(f"step {name} FAILED: {type(e).__name__}: {e}")
            if name in {"live"}:
                log("   (continuing; the live-jobs part is optional)")
                continue
            return 1
    log(f"ALL DONE in {(time.time() - t0) / 60:.1f} min.")
    log("Next: go back to the Claude chat and say: assets ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
