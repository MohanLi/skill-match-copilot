"""Resume input and persistent, evidence-backed job results. Run: streamlit run src/ui/app.py"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ui import service  # noqa: E402
from src.ui.helpers import highlight, read_resume, results_csv, safe_url, search_key  # noqa: E402

MODES = {"single_verified": "Verified matches", "single": "Single agent",
         "multi": "Multi-agent", "baseline": "Semantic search · no LLM"}
INDEXES = {"live_jobs": "Latest local job snapshot", "djinni_jobs": "Djinni evaluation pool"}

st.set_page_config(page_title="Skill-Match Copilot", page_icon="🧭", layout="wide")
st.markdown("""<style>
  .block-container {padding-top:2rem; max-width:1450px;}
  h1 {letter-spacing:-.04em;}
  [data-testid="stMetric"] {background:var(--secondary-background-color); border-radius:10px; padding:12px 16px;}
  [data-testid="stSidebar"] {min-width:290px; max-width:320px;}
  mark {color:#111827;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the skill models…")
def load_models():
    from src.p1_extract.predict import SkillExtractor
    from src.p2_normalize.normalizer import get_normalizer

    return SkillExtractor(), get_normalizer()


@st.cache_data(ttl=15, show_spinner=False)
def check_index(name):
    return service.index_status(name)


with st.sidebar:
    st.header("Search settings")
    index_name = st.selectbox("Job source", list(INDEXES), format_func=INDEXES.get)
    mode = st.radio("Matching method", list(MODES), format_func=MODES.get)
    top_n = st.slider("Jobs to return", 3, 10, 5)
    with st.expander("Advanced"):
        model = st.text_input("LLM role or model ID", "agent", disabled=mode == "baseline")
        st.caption("Agent modes use the configured API. Semantic search makes no LLM calls.")
    count, index_error = check_index(index_name)
    st.metric("Jobs in this index", f"{count:,}")
    if st.button("Refresh index status", use_container_width=True):
        check_index.clear()
        st.rerun()

st.title("Skill-Match Copilot")
st.write("Find jobs that fit my experience, with the evidence to check each match.")

issues = service.setup_issues(mode)
if index_error:
    issues.insert(0, index_error)
if issues:
    st.warning("This checkout needs local files before it can search.")
    with st.expander("What needs setting up", expanded=True):
        for issue in issues:
            st.write("• " + issue)
        st.caption("The trained checkpoints and job indexes are kept outside Git. The README has the setup steps.")

example_path = ROOT / "data" / "demo" / "synthetic_resume.md"
example_resume = example_path.read_text() if example_path.exists() else ""
st.session_state.setdefault("resume_text", example_resume)

with st.expander("My resume", expanded=not bool(st.session_state.get("search_result"))):
    uploaded = st.file_uploader("Upload a resume, or paste it below", type=["pdf", "docx", "txt", "md"])
    if uploaded is not None:
        digest = hashlib.sha256(uploaded.getvalue()).hexdigest()
        if digest != st.session_state.get("uploaded_resume_hash"):
            try:
                st.session_state["resume_text"] = read_resume(uploaded.getvalue(), uploaded.name)
                st.session_state["uploaded_resume_hash"] = digest
            except ValueError as error:
                st.error(str(error))
    if st.button("Load example resume"):
        st.session_state["resume_text"] = example_resume
    st.caption("The example is synthetic. Contact details are removed before matching.")
    with st.form("match_form"):
        resume = st.text_area("Resume text", key="resume_text", height=260)
        preferences = st.text_input("What kind of job am I looking for?", "machine learning roles, remote or Chicago",
                                    disabled=mode == "baseline")
        if mode == "baseline":
            st.caption("Semantic search ranks the resume text. Agent modes also use job preferences and explain skill gaps.")
        submitted = st.form_submit_button("Find matching jobs", type="primary", disabled=bool(issues))

key = search_key(resume, preferences, index_name, mode, top_n, model)
if submitted:
    if len(resume.strip()) < 50:
        st.error("Please add a little more resume text (at least 50 characters).")
    elif len(resume) > 20_000:
        st.error("Please shorten the resume to 20,000 characters before searching.")
    else:
        try:
            with st.spinner("Finding matching jobs…"):
                result = service.run_search(resume, preferences, index_name, mode, top_n, model, model_loader=load_models)
            result["search_key"] = key
            st.session_state["search_result"] = result
            st.rerun()
        except Exception:
            st.error("The search couldn't finish. Check the local models, index and API connection, then try again. Any previous results are still below.")

result = st.session_state.get("search_result")
if not result:
    st.info("Add my resume, choose a method, then find matching jobs. Results will appear here.")
    st.stop()

st.divider()
st.subheader("My job matches")
st.caption(f"{MODES[result['mode']]} · {INDEXES[result['index']]} · {result['top_n']} requested")
if key != result["search_key"]:
    st.info("These are the results of my previous search. Run a new search to apply the edited resume or settings.")

run, jobs, profile, faith = result["run"], result["jobs"], result["profile"], result["faithfulness"]
metrics = st.columns(4)
metrics[0].metric("Jobs returned", len(run["ranking"]))
metrics[1].metric("Search time", f"{result['elapsed']:.1f} s")
metrics[2].metric("LLM cost", f"${run['cost_usd']:.3f}")
metrics[3].metric("Unsupported claims", f"{faith['hallucination_rate']:.0%}" if faith and faith["n_claims"] else "N/A")
if run.get("error"):
    st.warning("The run returned a partial result. Check the run details before using it.")
if not run["ranking"]:
    st.info("No jobs were returned for this search. Try another source or a broader request.")

matches_tab, skills_tab, details_tab = st.tabs(["Matches", "My skills", "Run details"])
with matches_tab:
    if result["mode"] == "baseline":
        st.caption("Similarity scores use 100 × nonnegative cosine similarity. Skill explanations are available in agent modes.")
    for entry in run["ranking"]:
        job = jobs.get(entry["job_id"], {})
        with st.container(border=True):
            title_col, score_col = st.columns([5, 1])
            title_col.subheader(f"{entry['rank']}. {job.get('title') or entry['job_id']}")
            score_col.metric("Similarity" if result["mode"] == "baseline" else "Fit score", f"{entry.get('fit_score', 0):.0f}/100")
            metadata = [job.get("company"), job.get("category"), job.get("location"), job.get("published")]
            st.caption(" · ".join(str(item) for item in metadata if item))
            url = safe_url(job.get("url"))
            if url:
                st.link_button("View job posting", url)
            if entry.get("summary"):
                st.write(entry["summary"])
            matched, missing = entry.get("matched_skills") or [], entry.get("missing_skills") or []
            if result["mode"] != "baseline":
                left, right = st.columns(2)
                with left:
                    st.markdown("**Skills I match**")
                    if not matched:
                        st.caption("No matched skills were returned.")
                    for claim in matched:
                        st.write(claim.get("skill", ""))
                        st.caption("My resume")
                        st.text(claim.get("resume_quote") or "No quote returned")
                        st.caption("Job posting")
                        st.text(claim.get("job_quote") or "No quote returned")
                with right:
                    st.markdown("**Skills to check**")
                    if not missing:
                        st.caption("No missing skills were returned.")
                    for claim in missing:
                        st.write(claim.get("skill", ""))
                        st.text(claim.get("job_quote") or "No quote returned")
                if entry.get("dropped"):
                    with st.expander("Claims removed by the verifier"):
                        for dropped in entry["dropped"]:
                            st.write(f"{dropped.get('skill', '')}: {dropped.get('reason', '')}")
            with st.expander("Read the job text"):
                # Keep the whole text so evidence after character 5,000 is visible too.
                st.markdown(highlight(job.get("text", ""), [c.get("job_quote", "") for c in matched + missing]), unsafe_allow_html=True)
    if run["ranking"]:
        st.download_button("Download results (CSV)", results_csv(result), "skill-match-results.csv", "text/csv")

with skills_tab:
    if profile is None:
        st.info("Semantic search does not extract a skill profile. Choose an agent mode to see my skills and ESCO matches.")
    else:
        st.markdown("**My ESCO skills**")
        st.write(", ".join(profile["esco_skills"]) or "No ESCO skills returned.")
        st.markdown("**Phrases without an ESCO match**")
        st.write(", ".join(profile["not_in_esco"]) or "None returned.")
        with st.expander("Extracted phrases"):
            st.write({"skills": profile["skills"], "knowledge": profile["knowledge"]})

with details_tab:
    st.write({"method": result["mode"], "source": result["index"], "model": result["model"],
              "preferences": result["preferences"], "LLM calls": run["llm_calls"], "tool calls": run["tool_calls"]})
    with st.expander("Agent trace"):
        st.json(run["trace"])
    if run.get("error"):
        st.write("Run error:", run["error"])
    if st.button("Clear results"):
        del st.session_state["search_result"]
        st.rerun()
