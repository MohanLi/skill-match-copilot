"""Daily ingestion job: take the newest open-apply-jobs snapshot (filtered parquet produced by
scripts/download_assets.py or an equivalent fetch), extract + normalize skills with the champions and
upsert into the live_jobs index. Idempotent — already-indexed ids are skipped.

  python -m src.monitoring.daily_ingest            # index the snapshot in data/raw/live_jobs
Writes logs/ingest.jsonl (date, new postings, time) used by drift_report.py."""
from __future__ import annotations

import subprocess
import sys
import time

from ..data import live_jobs
from ..utils.config import ROOT
from ..utils.io import append_jsonl


def main() -> None:
    if not live_jobs.available():
        print("no live snapshot in data/raw/live_jobs — run scripts/download_assets.py --only live on a machine with Hugging Face access")
        sys.exit(1)
    date = live_jobs.snapshot_date()
    t0 = time.time()
    cmd = [sys.executable, "-m", "src.index.build_job_index", "--pool", "live"]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(out.stdout[-2000:])
    ok = out.returncode == 0
    append_jsonl({"ts": time.time(), "snapshot_date": date, "ok": ok, "seconds": round(time.time() - t0, 1), "tail": out.stdout[-300:] if ok else out.stderr[-500:]}, "logs/ingest.jsonl")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
