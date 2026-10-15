"""Problem 3 champion — one ReAct agent (LangGraph loop: agent → tools → agent …) with the five tools.

The LLM is called through src.utils.llm.LLMClient (Anthropic tool use), so every call is cached, logged
and budget-guarded like everywhere else in the project."""
from __future__ import annotations

import json
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..utils.config import models_cfg
from ..utils.llm import LLMClient, extract_json, load_prompt
from .common import AgentRun, normalise_ranking
from .tools import ToolBox


class AgentState(TypedDict):
    messages: list[dict[str, Any]]
    steps: int
    done: bool
    final_text: str


class SingleAgent:
    def __init__(self, toolbox: ToolBox, model: str = "agent", top_n: int | None = None, candidate_k: int | None = None, max_steps: int | None = None, task: str = "p3_single_agent"):
        cfg = models_cfg()["p3_agent"]
        self.tb = toolbox
        self.model = model
        self.top_n = top_n or cfg["top_n"]
        self.candidate_k = candidate_k or cfg["candidate_k"]
        self.max_steps = max_steps or cfg["max_agent_steps"]
        self.client = LLMClient(task=task)
        self.prompt = load_prompt("p3_single_agent_v1")
        self.graph = self._build()

    def _system(self) -> str:
        pool_note = f"The candidate pool for this request is fixed: only the {len(self.tb.pool_ids)} jobs returned by search_jobs exist; rank ALL of them." if self.tb.pool_ids else ""
        return self.prompt.system.replace("{{candidate_k}}", str(self.candidate_k)).replace("{{top_n}}", str(self.top_n)).replace("{{pool_note}}", pool_note)

    def _build(self):
        run: AgentRun = None  # type: ignore[assignment]

        def agent_node(state: AgentState) -> AgentState:
            res = self.client.chat(state["messages"], system=self._system(), model=self.model, prompt_version=self.prompt.tag, tools=self.tb.anthropic_tools(), max_tokens=3000)
            self._run.add_usage(res)
            self._run.trace.append({"step": state["steps"] + 1, "type": "llm", "stop_reason": res.stop_reason, "text": res.text[:1500], "tool_uses": [{"name": b["name"], "input": b["input"]} for b in res.content if b.get("type") == "tool_use"], "cached": res.cached, "cost_usd": res.cost_usd})
            msgs = state["messages"] + [{"role": "assistant", "content": res.content}]
            done = res.stop_reason != "tool_use" or state["steps"] + 1 >= self.max_steps
            return {"messages": msgs, "steps": state["steps"] + 1, "done": done, "final_text": res.text if done else state.get("final_text", "")}

        def tools_node(state: AgentState) -> AgentState:
            last = state["messages"][-1]["content"]
            results = []
            for b in last:
                if b.get("type") != "tool_use":
                    continue
                out = self.tb.dispatch(b["name"], b.get("input") or {})
                self._run.tool_calls += 1
                self._run.trace.append({"step": state["steps"], "type": "tool", "name": b["name"], "input": b.get("input"), "output_preview": out[:600]})
                results.append({"type": "tool_result", "tool_use_id": b["id"], "content": out[:12000]})
            return {"messages": state["messages"] + [{"role": "user", "content": results}], "steps": state["steps"], "done": False, "final_text": state.get("final_text", "")}

        g = StateGraph(AgentState)
        g.add_node("agent", agent_node)
        g.add_node("tools", tools_node)
        g.add_edge(START, "agent")
        g.add_conditional_edges("agent", lambda s: END if s["done"] else "tools", {END: END, "tools": "tools"})
        g.add_edge("tools", "agent")
        return g.compile()

    def run(self, resume: str, preferences: str = "") -> AgentRun:
        self._run = AgentRun(approach="single_agent", ranking=[])
        t0 = time.perf_counter()
        user = self.prompt.render(resume=resume, preferences=preferences or "none", top_n=self.top_n)
        try:
            final = self.graph.invoke({"messages": [{"role": "user", "content": user}], "steps": 0, "done": False, "final_text": ""}, config={"recursion_limit": 4 * self.max_steps + 4})
            text = final.get("final_text") or ""
            ranking = []
            try:
                ranking = normalise_ranking(extract_json(text), self.top_n)
            except Exception:  # noqa: BLE001
                pass
            if not ranking:  # ask once more, without tools, for the JSON only
                msgs = final["messages"] + [{"role": "user", "content": f"Give the FINAL ANSWER now as the JSON object described in the instructions (top {self.top_n} jobs). Output only JSON."}]
                res = self.client.chat(msgs, system=self._system(), model=self.model, prompt_version=self.prompt.tag + "+final", max_tokens=3000)
                self._run.add_usage(res)
                self._run.trace.append({"type": "llm_final", "text": res.text[:1500], "cost_usd": res.cost_usd})
                try:
                    ranking = normalise_ranking(extract_json(res.text), self.top_n)
                except Exception as e:  # noqa: BLE001
                    self._run.error = f"no parseable ranking: {e}"
            self._run.ranking = ranking
        except Exception as e:  # noqa: BLE001
            self._run.error = f"{type(e).__name__}: {str(e)[:300]}"
        self._run.extra["wall_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        self._run.extra["steps"] = sum(1 for t in self._run.trace if t.get("type") == "llm")
        return self._run
