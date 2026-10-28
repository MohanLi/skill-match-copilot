"""Run a UI search without loading extraction models for the embedding baseline."""
from __future__ import annotations

import os
import time
from pathlib import Path

from src.utils.config import ROOT, load_env, models_cfg
from src.utils.pii import scrub


def index_status(name: str) -> tuple[int, str | None]:
    if not (ROOT / "chroma").exists():
        return 0, "The local job indexes are missing. Restore the team's chroma/ folder or build an index first."
    try:
        from src.index.store import JobIndex

        count = JobIndex(name).count()
        return count, None if count else "This job index is empty. Build or restore it before searching."
    except Exception:
        return 0, "The job index could not be opened. Check the local Chroma installation and index files."


def setup_issues(mode: str, root: Path = ROOT, cfg: dict | None = None) -> list[str]:
    cfg = cfg or models_cfg()
    issues = []
    encoder = root / cfg["index"]["text_embedding_model"]
    if not (encoder / "config.json").is_file():
        issues.append(f"The sentence encoder is missing: {encoder.relative_to(root)}. Run scripts/download_assets.py or restore the team's model bundle.")
    if mode != "baseline":
        taggers = root / "models" / "p1_tagger"
        separate = all((taggers / f"{kind}_seed42" / "config.json").is_file() for kind in ("skill", "knowledge"))
        merged = (taggers / "merged_seed42" / "config.json").is_file()
        if not (separate or merged):
            issues.append("The trained skill taggers are missing from models/p1_tagger/. Restore the team's trained checkpoints; downloading the base models alone is not enough.")
        load_env()
        if not os.getenv("ANTHROPIC_API_KEY"):
            issues.append("Add ANTHROPIC_API_KEY to your local .env file to use an agent. Semantic search does not need an API key.")
    return issues


def run_search(resume: str, preferences: str, index_name: str, mode: str, top_n: int,
               model: str = "agent", model_loader=None) -> dict:
    from src.p3_agent.baseline import EmbeddingBaseline
    from src.p3_agent.common import faithfulness_of_ranking, resume_profile
    from src.p3_agent.tools import ToolBox

    clean = scrub(resume).text
    clean_preferences = scrub(preferences).text
    started = time.perf_counter()
    if mode == "baseline":
        tb = ToolBox(index_name)
        agent = EmbeddingBaseline(tb, top_n=top_n)
    else:
        from src.p3_agent.multi_agent import MultiAgent
        from src.p3_agent.single_agent import SingleAgent
        from src.p3_agent.verified import VerifiedSingleAgent

        extractor, normalizer = model_loader()
        tb = ToolBox(index_name, extractor=extractor, normalizer=normalizer)
        cls = {"single": SingleAgent, "single_verified": VerifiedSingleAgent, "multi": MultiAgent}[mode]
        agent = cls(tb, model=model, top_n=top_n, task=f"ui_{mode}")
    if tb.index.count() == 0:
        raise ValueError("The selected job index is empty. Build or restore it before searching.")
    run = agent.run(clean, clean_preferences)
    jobs = {j["job_id"]: j for j in tb.index.get_many([e["job_id"] for e in run.ranking])} if run.ranking else {}
    profile = resume_profile(tb, clean) if mode != "baseline" else None
    faith = faithfulness_of_ranking(run.ranking, clean, jobs, profile["esco_skills"]) if profile is not None else None
    return {"run": run.to_dict(), "jobs": jobs, "profile": profile, "faithfulness": faith,
            "elapsed": time.perf_counter() - started, "mode": mode, "index": index_name,
            "preferences": clean_preferences, "top_n": top_n, "model": model}
