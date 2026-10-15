"""Streamlit demo — paste a resume, pick single- or multi-agent mode, see ranked jobs with highlighted
evidence and the agent trace.   Run: streamlit run src/ui/app.py"""
from __future__ import annotations

import html
import json
import re
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.index.store import JobIndex  # noqa: E402
from src.p3_agent.common import faithfulness_of_ranking, resume_profile  # noqa: E402
from src.utils.pii import scrub  # noqa: E402

st.set_page_config(page_title="Skill-Match Copilot", page_icon="🧭", layout="wide")
st.title("🧭 Skill-Match Copilot")
st.caption("Paste a resume → extracted skills → ESCO normalization → ranked jobs with quoted evidence. Every claim is verified against the texts.")


@st.cache_resource(show_spinner="Loading the fine-tuned models…")
def load_models():
    from src.p1_extract.predict import SkillExtractor
    from src.p2_normalize.normalizer import get_normalizer

    return SkillExtractor(), get_normalizer()


def highlight(text: str, quotes: list[str], color: str) -> str:
    out = html.escape(text)
    for q in sorted({q for q in quotes if q}, key=len, reverse=True):
        pat = re.compile(re.escape(html.escape(q)), re.I)
        out = pat.sub(lambda m: f"<mark style='background:{color};padding:0 2px;border-radius:3px'>{m.group(0)}</mark>", out, count=1)
    return out.replace("\n", "<br>")


with st.sidebar:
    st.header("Settings")
    index_name = st.selectbox("Job index", ["live_jobs", "djinni_jobs"], help="live_jobs = this week's data/AI/ML postings; djinni_jobs = evaluation pool")
    mode = st.radio("Approach", ["single", "multi", "baseline"], format_func=lambda m: {"single": "Single ReAct agent (champion)", "multi": "Multi-agent + verifier (challenger)", "baseline": "Embedding baseline (no LLM)"}[m])
    top_n = st.slider("Jobs to return", 3, 10, 5)
    model = st.text_input("LLM role / model id", "agent")
    try:
        st.info(f"{index_name}: {JobIndex(index_name).count():,} postings indexed")
    except Exception as e:  # noqa: BLE001
        st.warning(f"index not available: {e}")

default_resume = (ROOT / "data" / "demo" / "synthetic_resume.md").read_text() if (ROOT / "data" / "demo" / "synthetic_resume.md").exists() else ""
resume = st.text_area("Resume", value=default_resume, height=260)
prefs = st.text_input("Preferences (optional)", "machine learning roles, remote or Chicago")

if st.button("Find matching jobs", type="primary", disabled=not resume.strip()):
    extractor, normalizer = load_models()
    from src.p3_agent.baseline import EmbeddingBaseline
    from src.p3_agent.multi_agent import MultiAgent
    from src.p3_agent.single_agent import SingleAgent
    from src.p3_agent.tools import ToolBox

    clean = scrub(resume).text
    tb = ToolBox(index_name, extractor=extractor, normalizer=normalizer)
    t0 = time.time()
    with st.spinner("Extracting skills, searching and ranking…"):
        if mode == "baseline":
            run = EmbeddingBaseline(tb, top_n=top_n).run(clean, prefs)
        elif mode == "single":
            run = SingleAgent(tb, model=model, top_n=top_n, task="ui_single").run(clean, prefs)
        else:
            run = MultiAgent(tb, model=model, top_n=top_n, task="ui_multi").run(clean, prefs)
        prof = resume_profile(tb, clean)
    elapsed = time.time() - t0
    jobs = {j["job_id"]: j for j in tb.index.get_many([e["job_id"] for e in run.ranking])} if run.ranking else {}
    faith = faithfulness_of_ranking(run.ranking, clean, jobs, prof["esco_skills"]) if mode != "baseline" else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jobs ranked", len(run.ranking))
    c2.metric("LLM calls / cost", f"{run.llm_calls} / ${run.cost_usd:.3f}")
    c3.metric("Wall time", f"{elapsed:.1f} s")
    c4.metric("Unsupported claims", f"{faith['hallucination_rate']:.0%}" if faith and faith["n_claims"] else "n/a")
    if run.error:
        st.error(run.error)

    with st.expander("Resume skill profile (tagger + ESCO normalization)"):
        st.write("**ESCO skills:**", ", ".join(prof["esco_skills"]) or "—")
        st.write("**Not in ESCO:**", ", ".join(prof["not_in_esco"][:30]) or "—")
        st.write("**Raw phrases:**", ", ".join((prof["skills"] + prof["knowledge"])[:40]) or "—")

    for e in run.ranking:
        j = jobs.get(e["job_id"], {})
        with st.container(border=True):
            st.subheader(f"#{e['rank']} · {j.get('title', e['job_id'])}  —  fit {e.get('fit_score', 0):.0f}/100")
            meta = " · ".join(str(x) for x in [j.get("company"), j.get("category"), j.get("published")] if x)
            if meta:
                st.caption(meta + (f" · [apply]({j['url']})" if j.get("url") else ""))
            if e.get("summary"):
                st.write(e["summary"])
            mcol, xcol = st.columns(2)
            with mcol:
                st.markdown("**Matched skills** (resume ↔ job)")
                for c in e.get("matched_skills", []):
                    st.markdown(f"- **{c.get('skill')}** — _“{c.get('resume_quote')}”_ ↔ _“{c.get('job_quote')}”_")
            with xcol:
                st.markdown("**Missing skills**")
                for c in e.get("missing_skills", []):
                    st.markdown(f"- **{c.get('skill')}** — _“{c.get('job_quote')}”_")
                for d in e.get("dropped", []) or []:
                    st.markdown(f"- ~~{d.get('skill')}~~ dropped by verifier: {d.get('reason')}")
            with st.expander("Job text with evidence highlighted"):
                st.markdown(highlight(j.get("text", "")[:5000], [c.get("job_quote", "") for c in e.get("matched_skills", []) + e.get("missing_skills", [])], "#cde2fb"), unsafe_allow_html=True)
    with st.expander("Agent trace"):
        st.json(run.trace)
