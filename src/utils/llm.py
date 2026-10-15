"""Thin, provider-swappable LLM wrapper (decision #2) with caching, logging and a budget guard.

Design
- One provider (Anthropic) behind `LLMClient`; `complete()` for text-in/text-out, `chat()` for
  multi-turn / tool-use. Adding a provider = implementing `_call_raw` for it.
- Every call is cached on disk by a hash of (model, prompt_version, system, messages, tools,
  temperature, max_tokens). Re-running an evaluation costs nothing.
- Every *live* call is appended to logs/llm_calls.jsonl with model, prompt version, tokens,
  latency, cost and the calling task. `budget_status()` sums the log; a live call that would
  exceed `llm.budget_usd` raises BudgetExceeded.
- Model ids from configs/models.yaml are verified against GET /v1/models at first use; if an id
  does not exist the newest model of the same family (haiku/sonnet/opus) is used and recorded.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import ROOT, load_env, models_cfg
from .io import append_jsonl, read_jsonl


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    content: list[dict[str, Any]]
    stop_reason: str | None
    model: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    cached: bool
    raw_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Prompt:
    id: str
    version: str
    system: str
    user_template: str
    examples: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def tag(self) -> str:
        return f"{self.id}_v{self.version}"

    def render(self, **kw: Any) -> str:
        return render_template(self.user_template, **kw)


def render_template(template: str, **kw: Any) -> str:
    """Replace {{name}} tokens (double braces, so JSON in prompts needs no escaping)."""

    def _sub(m: re.Match) -> str:
        key = m.group(1).strip()
        if key not in kw:
            raise KeyError(f"prompt placeholder {{{{{key}}}}} not provided")
        v = kw[key]
        return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, indent=2)

    return re.sub(r"\{\{\s*([\w.]+)\s*\}\}", _sub, template)


def load_prompt(name: str) -> Prompt:
    """Load prompts/<name>.yaml (name like 'p1_extract_v1')."""
    import yaml

    p = ROOT / "prompts" / f"{name}.yaml"
    with open(p, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return Prompt(
        id=d["id"],
        version=str(d["version"]),
        system=d.get("system", "").strip(),
        user_template=d.get("user_template", "").strip(),
        examples=d.get("examples", []) or [],
        meta={k: v for k, v in d.items() if k not in {"id", "version", "system", "user_template", "examples"}},
    )


def extract_json(text: str) -> Any:
    """Return the first JSON object/array found in `text` (handles ```json fences and prose)."""
    if text is None:
        raise ValueError("empty response")
    s = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if fence:
        s = fence.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start = min([i for i in [s.find("{"), s.find("[")] if i >= 0], default=-1)
    if start < 0:
        raise ValueError(f"no JSON in response: {text[:200]!r}")
    # walk to the matching bracket
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth == 0:
                return json.loads(s[start : i + 1])
    raise ValueError(f"unbalanced JSON in response: {text[:200]!r}")


class LLMClient:
    FAMILIES = ("haiku", "sonnet", "opus")

    def __init__(self, task: str = "misc", cache: bool = True, dry_run: bool = False):
        load_env()
        cfg = models_cfg()["llm"]
        self.cfg = cfg
        self.task = task
        self.use_cache = cache
        self.dry_run = dry_run
        self.cache_dir = ROOT / cfg.get("cache_dir", ".cache/llm")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = cfg.get("log_path", "logs/llm_calls.jsonl")
        self.budget = float(cfg.get("budget_usd", 25.0))
        self.prices = cfg.get("prices_per_mtok", {})
        self._client = None
        self._available: list[str] | None = None
        self._resolved: dict[str, str] = {}

    # ----------------------------------------------------------------- provider
    @property
    def client(self):
        if self._client is None:
            import anthropic

            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY missing: put it in .env (see .env.example)")
            self._client = anthropic.Anthropic(api_key=key, max_retries=6, timeout=120.0)
        return self._client

    def available_models(self) -> list[str]:
        if self._available is None:
            cache_f = self.cache_dir / "models.json"
            if cache_f.exists() and time.time() - cache_f.stat().st_mtime < 86400:
                self._available = json.loads(cache_f.read_text())
            else:
                ids: list[str] = []
                page = self.client.models.list(limit=100)
                for m in page.data:
                    ids.append(m.id)
                self._available = ids
                cache_f.write_text(json.dumps(ids, indent=2))
        return self._available

    def resolve_model(self, role_or_id: str) -> str:
        """Accepts a role ('bulk', 'agent', 'label_primary', ...) or a model id. Verifies it exists."""
        model = self.cfg.get(role_or_id, role_or_id)
        model = os.environ.get(f"SMC_LLM_{role_or_id.upper()}", model) if role_or_id in self.cfg else model
        if model in self._resolved:
            return self._resolved[model]
        resolved = model
        try:
            avail = self.available_models()
            if model not in avail:
                fam = next((f for f in self.FAMILIES if f in model), None)
                cands = sorted([m for m in avail if fam and fam in m], reverse=True)
                if not cands:
                    raise RuntimeError(f"model {model!r} not available and no {fam} fallback; available: {avail}")
                resolved = cands[0]
                append_jsonl({"event": "model_fallback", "requested": model, "used": resolved, "ts": time.time()}, self.log_path)
        except Exception as e:  # noqa: BLE001
            if "ANTHROPIC_API_KEY" in str(e):
                raise
            # offline / listing failed: trust the configured id
        self._resolved[model] = resolved
        return resolved

    # -------------------------------------------------------------------- cost
    def price(self, model: str) -> tuple[float, float, bool]:
        for k, v in self.prices.items():
            if model.startswith(k):
                return float(v["input"]), float(v["output"]), True
        return 0.0, 0.0, False

    def cost(self, model: str, inp: int, out: int) -> float:
        pi, po, _ = self.price(model)
        return inp / 1e6 * pi + out / 1e6 * po

    def budget_status(self) -> dict[str, float]:
        spent = sum(float(r.get("cost_usd", 0.0)) for r in read_jsonl(self.log_path) if r.get("event") == "call")
        return {"spent_usd": round(spent, 4), "budget_usd": self.budget, "remaining_usd": round(self.budget - spent, 4)}

    # ------------------------------------------------------------------- cache
    def _key(self, payload: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()

    # -------------------------------------------------------------------- calls
    def chat(
        self,
        messages: list[dict[str, Any]],
        system: str = "",
        model: str = "bulk",
        prompt_version: str = "none",
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tool_choice: dict[str, Any] | None = None,
    ) -> LLMResult:
        model_id = self.resolve_model(model)
        temperature = self.cfg.get("temperature", 0.0) if temperature is None else temperature
        max_tokens = int(self.cfg.get("max_tokens", 1024) if max_tokens is None else max_tokens)
        payload = {
            "model": model_id, "system": system, "messages": messages, "tools": tools or [],
            "temperature": temperature, "max_tokens": max_tokens, "prompt_version": prompt_version,
            "tool_choice": tool_choice,
        }
        key = self._key(payload)
        cache_f = self.cache_dir / f"{key}.json"
        if self.use_cache and cache_f.exists():
            d = json.loads(cache_f.read_text())
            d["cached"] = True
            return LLMResult(**d)
        if self.dry_run:
            raise RuntimeError("dry_run: no cached response for this call")

        status = self.budget_status()
        if status["remaining_usd"] <= 0:
            raise BudgetExceeded(f"LLM budget exhausted: {status}")

        kwargs: dict[str, Any] = dict(model=model_id, max_tokens=max_tokens, temperature=temperature, messages=messages)
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        t0 = time.perf_counter()
        resp = self.client.messages.create(**kwargs)
        latency = (time.perf_counter() - t0) * 1000
        content = [b.model_dump() for b in resp.content]
        text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
        inp, out = int(resp.usage.input_tokens), int(resp.usage.output_tokens)
        c = self.cost(model_id, inp, out)
        res = LLMResult(
            text=text, content=content, stop_reason=resp.stop_reason, model=model_id, prompt_version=prompt_version,
            input_tokens=inp, output_tokens=out, latency_ms=round(latency, 1), cost_usd=round(c, 6), cached=False, raw_id=resp.id,
        )
        append_jsonl(
            {"event": "call", "ts": time.time(), "task": self.task, "model": model_id, "prompt_version": prompt_version,
             "input_tokens": inp, "output_tokens": out, "latency_ms": res.latency_ms, "cost_usd": res.cost_usd,
             "price_known": self.price(model_id)[2], "stop_reason": resp.stop_reason, "cache_key": key},
            self.log_path,
        )
        if self.use_cache:
            cache_f.write_text(json.dumps(res.to_dict(), ensure_ascii=False))
        return res

    def complete(self, prompt: Prompt, model: str = "bulk", max_tokens: int | None = None, temperature: float | None = None, **kw: Any) -> LLMResult:
        """Render prompt.user_template with **kw and call the model once."""
        user = prompt.render(**kw)
        return self.chat(
            messages=[{"role": "user", "content": user}], system=prompt.system, model=model,
            prompt_version=prompt.tag, max_tokens=max_tokens, temperature=temperature,
        )

    def complete_json(self, prompt: Prompt, model: str = "bulk", max_tokens: int | None = None, **kw: Any) -> tuple[Any, LLMResult]:
        res = self.complete(prompt, model=model, max_tokens=max_tokens, **kw)
        try:
            return extract_json(res.text), res
        except ValueError:
            # one repair attempt: ask the model to emit only JSON
            messages = [
                {"role": "user", "content": prompt.render(**kw)},
                {"role": "assistant", "content": res.text or "(empty)"},
                {"role": "user", "content": "Return ONLY the JSON object requested above, with no prose."},
            ]
            res2 = self.chat(messages, system=prompt.system, model=model, prompt_version=prompt.tag + "+repair", max_tokens=max_tokens)
            return extract_json(res2.text), res2


def summarize_log(rel_path: str = "logs/llm_calls.jsonl") -> "pd.DataFrame":  # noqa: F821
    """Per task × model totals from the call log (for the cost tables in the report)."""
    import pandas as pd

    rows = [r for r in read_jsonl(rel_path) if r.get("event") == "call"]
    if not rows:
        return pd.DataFrame(columns=["task", "model", "calls", "input_tokens", "output_tokens", "cost_usd", "median_latency_ms"])
    df = pd.DataFrame(rows)
    g = df.groupby(["task", "model"]).agg(
        calls=("ts", "size"), input_tokens=("input_tokens", "sum"), output_tokens=("output_tokens", "sum"),
        cost_usd=("cost_usd", "sum"), median_latency_ms=("latency_ms", "median"),
    ).reset_index()
    return g
