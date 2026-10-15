#!/usr/bin/env python3
"""Reassemble and extract the transfer bundle made by download_assets.py (used in the build workspace).

  python scripts/import_transfer.py [--src data/transfer] [--dest .]
Verifies every part's sha256 against manifest.json, concatenates them and extracts the tar into the repo
(data/raw/hf, data/raw/esco, data/raw/live_jobs, models/base)."""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data" / "transfer"))
    ap.add_argument("--dest", default=str(ROOT))
    ap.add_argument("--skip-verify", action="store_true")
    a = ap.parse_args()
    src = Path(a.src)
    manifest = json.loads((src / "manifest.json").read_text())
    parts = [src / p["name"] for p in manifest["parts"]]
    missing = [p.name for p in parts if not p.exists()]
    if missing:
        raise SystemExit(f"missing parts: {missing}")
    if not a.skip_verify:
        for p, m in zip(parts, manifest["parts"]):
            h = sha256_of(p)
            if h != m["sha256"]:
                raise SystemExit(f"checksum mismatch for {p.name}: {h} != {m['sha256']}")
            print(f"verified {p.name} ({m['bytes']/1e6:.0f} MB)")

    class Cat:
        def __init__(self, files):
            self.files, self.i, self.fh = files, 0, open(files[0], "rb")

        def read(self, n=-1):
            buf = b""
            while n < 0 or len(buf) < n:
                chunk = self.fh.read(n - len(buf) if n >= 0 else -1)
                if chunk:
                    buf += chunk
                    if n < 0:
                        continue
                else:
                    self.fh.close()
                    self.i += 1
                    if self.i >= len(self.files):
                        break
                    self.fh = open(self.files[self.i], "rb")
            return buf

    with tarfile.open(fileobj=Cat(parts), mode="r|") as tar:
        try:
            tar.extractall(a.dest, filter="data")
        except TypeError:
            tar.extractall(a.dest)
    print("extracted roots:", manifest["roots"], "->", a.dest)


if __name__ == "__main__":
    main()
