"""Resume and requirements in, evidence-backed job matches out. Run: streamlit run src/ui/app.py"""
from __future__ import annotations

import hashlib
import sys
import traceback
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.p3_agent.requirements import Requirements  # noqa: E402
from src.ui import service  # noqa: E402
from src.ui.helpers import (esco_table, evidence_table, facts_line, group_profile, highlight, md, pool_sentence, read_resume,  # noqa: E402
                            relaxed_sentence, requirement_badges, results_csv, safe_url, saved_csv, search_key, skills_summary,
                            sort_ranking, summary_badges)

MODES = {"single_verified": "Verified matches", "single": "Single agent",
         "multi": "Multi-agent", "baseline": "Semantic search · no LLM"}
INDEXES = {"live_jobs": "Latest local job snapshot", "djinni_jobs": "Djinni evaluation pool"}
ROLE_SUGGESTIONS = ["AI engineer", "Machine learning engineer", "Data scientist", "Data engineer", "Data analyst",
                    "Applied scientist", "Research scientist", "MLOps engineer", "Software engineer"]
SORTS = {"fit": "Best fit", "newest": "Newest first"}

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


@st.cache_data(ttl=300, show_spinner=False)
def source_options(name):
    return service.requirement_options(name)


def options_with(current, *groups):
    """Suggestions for a multiselect, always including what is already selected (typed-in values are allowed)."""
    return list(dict.fromkeys([*current, *(item for group in groups for item in group)]))


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
    if st.button("Refresh index status", width="stretch"):
        check_index.clear()
        source_options.clear()
        service.clear_cache()
        st.rerun()

st.title("Skill-Match Copilot")
st.write("Find jobs that fit my experience and my requirements, with the evidence to check each match.")

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
has_result = bool(st.session_state.get("search_result"))

with st.expander("My resume", expanded=not has_result):
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
    resume = st.text_area("Resume text", key="resume_text", height=260)

source = source_options(index_name)
has_places = bool(source.get("location_data", True))
if not has_places and index_name == "live_jobs" and not index_error:
    st.info("This job index was built before job facts were stored, so cards cannot show company or location and the location "
            "requirement is off. Run `python -m src.index.job_facts` once (it takes seconds and re-embeds nothing), then press "
            "“Refresh index status”.")
st.session_state.setdefault("req_roles", ["Machine learning engineer"])
st.session_state.setdefault("req_locations", ["Chicago"])
st.session_state.setdefault("req_remote", True)
st.session_state.setdefault("req_remote_within", "United States")

with st.form("match_form"):
    st.markdown("**What I'm looking for**")
    role_col, place_col = st.columns(2)
    roles = role_col.multiselect("Role, any of", options_with(st.session_state["req_roles"], ROLE_SUGGESTIONS), key="req_roles",
                                 accept_new_options=True, placeholder="Pick or type a role — empty means any role",
                                 help="A job matches a role when its title contains every word of it. ML = machine learning and AI = artificial "
                                      "intelligence; nothing else is assumed, so add a second role to widen the search.")
    locations = place_col.multiselect("Location, any of", options_with(st.session_state["req_locations"], source.get("locations", [])),
                                      key="req_locations", accept_new_options=True, disabled=not has_places,
                                      placeholder="Pick or type a place — empty means anywhere",
                                      help="Matched as whole words in the posting's location, so 'Chicago' also finds 'Chicago, IL (Hybrid)'.")
    remote_col, within_col, notes_col = st.columns([1.1, 1.4, 2.5])
    remote_ok = remote_col.checkbox("Remote jobs count too", key="req_remote", disabled=not has_places)
    remote_within = within_col.text_input("Remote within (optional)", key="req_remote_within", placeholder="United States", disabled=not has_places,
                                          help="Keep a remote job only if its location mentions this region. Leave empty for remote anywhere.")
    notes = notes_col.text_input("Anything else the agent should know (optional)", key="req_notes", disabled=mode == "baseline",
                                 placeholder="early-stage startups, no on-call")
    if not has_places:
        st.caption("This job source has no location data, so only the role requirement applies.")
    submitted = st.form_submit_button("Find matching jobs", type="primary", disabled=bool(issues))

requirements = Requirements.from_any({"roles": roles, "locations": locations if has_places else [], "remote_ok": bool(remote_ok and has_places),
                                      "remote_within": remote_within if (remote_ok and has_places) else ""}).to_dict()
requirements.pop("notes")
notes = "" if mode == "baseline" else (notes or "")
key = search_key(resume, notes, index_name, mode, top_n, model, requirements)
if submitted:
    if len(resume.strip()) < 50:
        st.error("Please add a little more resume text (at least 50 characters).")
    elif len(resume) > 20_000:
        st.error("Please shorten the resume to 20,000 characters before searching.")
    else:
        try:
            with st.spinner("Finding matching jobs…"):
                result = service.run_search(resume, notes, index_name, mode, top_n, model, model_loader=load_models, requirements=requirements)
            result["search_key"] = key
            st.session_state["search_result"] = result
            st.session_state["similar"] = {}
            st.rerun()
        except Exception as error:
            st.error("The search couldn't finish. Check the local models, index and API connection, then try again. Any previous results are still below.")
            with st.expander("Error details"):
                st.code(f"{type(error).__name__}: {error}\n\n{traceback.format_exc()}")

result = st.session_state.get("search_result")
if not result:
    st.info("Add my resume, say what I'm looking for, then find matching jobs. Results will appear here.")
    st.stop()

st.divider()
st.subheader("My job matches")
st.caption(f"{MODES[result['mode']]} · {INDEXES[result['index']]} · {result['top_n']} requested")
if key != result["search_key"]:
    st.info("These are the results of my previous search. Run a new search to apply the edited resume, requirements or settings.")

run, jobs, profile, faith = result["run"], result["jobs"], result["profile"], result["faithfulness"]
asked = Requirements.from_any(result.get("requirements"))
pool, checks = result.get("pool"), result.get("checks") or {}
agent_mode = result["mode"] != "baseline"
if asked.filters:
    st.markdown(f"**Looking for:** {md(asked.describe())}")
    if pool_sentence(pool):
        st.caption(pool_sentence(pool) + " Only those are ranked." * bool(pool.get("restricted") and not pool.get("relaxed")))
    if relaxed_sentence(pool):
        st.warning(relaxed_sentence(pool))
    if asked.has_location and pool and not pool.get("location_data"):
        st.info("This job source has no location data, so the location requirement was not applied.")

metrics = st.columns(4)
metrics[0].metric("Jobs returned", len(run["ranking"]))
metrics[1].metric("Search time", f"{result['elapsed']:.1f} s")
metrics[2].metric("LLM cost", f"${run['cost_usd']:.3f}")
metrics[3].metric("Unsupported claims", f"{faith['hallucination_rate']:.0%}" if faith and faith["n_claims"] else "N/A")
if run.get("error"):
    st.warning("The run returned a partial result. Check the run details before using it.")
if not run["ranking"]:
    st.info("No jobs were returned for this search. Try another source or a broader request.")

saved = st.session_state.setdefault("saved_jobs", {})
similar = st.session_state.setdefault("similar", {})
matches_tab, skills_tab, saved_tab, details_tab = st.tabs(["Matches", "My skills", "Saved", "Run details"])
with matches_tab:
    if not agent_mode:
        st.caption("Similarity scores use 100 × nonnegative cosine similarity. Skill explanations are available in agent modes.")
    if agent_mode and run["ranking"]:
        overview = skills_summary(run["ranking"])
        if overview["missing"] or overview["matched"]:
            with st.container(border=True):
                st.markdown(f"**Across my {overview['n_jobs']} matches**")
                if overview["missing"]:
                    st.markdown("Asked for, not shown in my resume:  " + summary_badges(overview["missing"], overview["n_jobs"], "orange"))
                if overview["matched"]:
                    st.markdown("Shown in my resume:  " + summary_badges(overview["matched"], overview["n_jobs"], "green"))
                st.caption("A count means the skill appears in that many of these jobs; grey ones appear in one. All of it comes from the evidence on the cards below.")
    if len(run["ranking"]) > 1:
        _, sort_col = st.columns([4, 1])
        order = sort_col.selectbox("Sort by", list(SORTS), format_func=SORTS.get, key="sort_by")
    else:
        order = "fit"
    for entry in sort_ranking(run["ranking"], jobs, order):
        job_id = entry["job_id"]
        job = jobs.get(job_id, {})
        matched, missing = entry.get("matched_skills") or [], entry.get("missing_skills") or []
        with st.container(border=True):
            title_col, score_col = st.columns([5, 1])
            title_col.subheader(f"{entry['rank']}. {job.get('title') or job_id}")
            score_col.metric("Fit score" if agent_mode else "Similarity", f"{entry.get('fit_score', 0):.0f}/100")
            facts = facts_line(job, prefer_locations=asked.locations)
            if facts:
                st.caption(facts)
            if entry.get("also"):
                places = [f"[{md(other.get('locations') or 'another posting')}](<{safe_url(other.get('url'))}>)" if safe_url(other.get("url"))
                          else md(other.get("locations") or "another posting") for other in entry["also"]]
                st.caption(":material/content_copy: Same job also posted for: " + " · ".join(places))
            badges = requirement_badges(checks.get(job_id), len(matched) if agent_mode else None, len(missing) if agent_mode else None)
            if badges:
                st.markdown(badges)
            url = safe_url(job.get("url"))
            if url:
                link_col, save_col, more_col, _ = st.columns([1.5, 1.1, 1.6, 4.8])
                link_col.link_button("View job posting", url, width="stretch")
            else:   # the Djinni evaluation pool has no posting links
                save_col, more_col, _ = st.columns([1.1, 1.6, 6.3])
            if save_col.button("Saved" if job_id in saved else "Save", key=f"save-{job_id}", width="stretch",
                               icon=":material/bookmark_added:" if job_id in saved else ":material/bookmark_add:"):
                if job_id in saved:
                    del saved[job_id]
                else:
                    saved[job_id] = {**{k: v for k, v in job.items() if k not in ("text", "esco_skills", "raw_skills")}, "score": entry.get("fit_score", 0)}
                st.rerun()
            if more_col.button("More like this", key=f"more-{job_id}", width="stretch", icon=":material/travel_explore:"):
                try:
                    similar[job_id] = service.similar_jobs(result["index"], job_id, result.get("requirements"), exclude=[e["job_id"] for e in run["ranking"]])
                except Exception as error:
                    similar.pop(job_id, None)
                    st.warning(f"Similar postings could not be loaded ({type(error).__name__}).")
            if entry.get("summary"):
                st.write(entry["summary"])
            if agent_mode:
                st.markdown(f"**Skills I match** · {len(matched)}")
                st.markdown(evidence_table(matched, "matched") or "_No matched skills were returned._")
                st.markdown(f"**Skills to check** · {len(missing)}")
                st.markdown(evidence_table(missing, "missing") or "_No missing skills were returned._")
                st.caption("Every quote is copied verbatim from the resume or the posting; claims the verifier could not support were removed.")
                if entry.get("dropped"):
                    with st.expander(f"Claims removed by the verifier · {len(entry['dropped'])}"):
                        for dropped in entry["dropped"]:
                            st.markdown(f"- **{dropped.get('skill', '')}** — {dropped.get('reason', '')}")
            if job_id in similar:
                with st.expander(f"Similar postings · {len(similar[job_id])}", expanded=True):
                    if not similar[job_id]:
                        st.write("No other postings like this one meet my requirements.")
                    for other in similar[job_id]:
                        other_url = safe_url(other.get("url"))
                        name = md(other.get("title") or other["job_id"])
                        st.markdown(f"**[{name}](<{other_url}>)**" if other_url else f"**{name}**")
                        st.caption("  ·  ".join(part for part in (facts_line(other, prefer_locations=asked.locations), f"similarity {other.get('score', 0):.2f}") if part))
                    st.caption("Nearest postings by text similarity (no LLM), inside the same requirements.")
            with st.expander("Read the job text"):
                # Keep the whole text so evidence after character 5,000 is visible too.
                st.markdown(highlight(job.get("text", ""), [c.get("job_quote", "") for c in matched + missing]), unsafe_allow_html=True)
    if run["ranking"]:
        st.download_button("Download results (CSV)", results_csv(result), "skill-match-results.csv", "text/csv")

with skills_tab:
    if profile is None:
        st.info("Semantic search does not extract a skill profile. Choose an agent mode to see my skills and ESCO matches.")
    elif not profile.get("mapped"):
        st.markdown("**My ESCO skills**")
        st.write(", ".join(profile["esco_skills"]) or "No ESCO skills returned.")
        st.markdown("**Phrases without an ESCO match**")
        st.write(", ".join(profile["not_in_esco"]) or "None returned.")
    else:
        groups = group_profile(profile)
        st.markdown(f"**{len(profile['esco_skills'])} ESCO skills** recognised from {len(groups['esco'])} resume phrases · "
                    f"**{len(groups['tools'])} tools and technologies** that ESCO does not list · {len(groups['activities'])} activities.")
        st.markdown("##### Recognised ESCO skills")
        st.caption("Each resume phrase with the ESCO skill it was mapped to. A weak match is where the normaliser is most often wrong "
                   "(a tool name landing on an unrelated skill is its known failure mode).")
        st.markdown(esco_table(groups["esco"]) or "_No phrase reached the ESCO threshold._")
        st.markdown("##### Tools and technologies outside ESCO")
        st.caption("Mostly modern tools and frameworks that ESCO 1.1.0 does not contain. They still count: agents match them by name, with quotes from both texts.")
        st.markdown(" ".join(f":blue-badge[{md(tool)}]" for tool in groups["tools"]) or "_None._")
        if groups["activities"]:
            st.markdown("##### What the resume says I did")
            st.markdown("\n".join(f"- {md(activity)}" for activity in groups["activities"]))
        with st.expander(f"Other extracted phrases · {len(groups['fragments'])}"):
            st.caption("Leftovers of the span tagger: phrases cut off mid-way, stray brackets, lone verbs. Shown for completeness; none of them is in ESCO.")
            st.write(", ".join(groups["fragments"]) or "None.")
        with st.expander("Raw tagger output"):
            st.write({"skills": profile["skills"], "knowledge": profile["knowledge"]})

with saved_tab:
    if not saved:
        st.info("Nothing saved yet. Press Save on a match to keep it here while this browser tab stays open.")
    else:
        st.caption(f"{len(saved)} saved · kept for this browser session only — download the list to keep it.")
        for saved_id, item in list(saved.items()):
            item_col, remove_col = st.columns([7, 1])
            item_url = safe_url(item.get("url"))
            item_name = md(item.get("title") or saved_id)
            item_col.markdown(f"**[{item_name}](<{item_url}>)**" if item_url else f"**{item_name}**")
            item_col.caption(facts_line(item))
            if remove_col.button("Remove", key=f"unsave-{saved_id}", width="stretch"):
                del saved[saved_id]
                st.rerun()
        st.download_button("Download saved jobs (CSV)", saved_csv(saved), "saved-jobs.csv", "text/csv")

with details_tab:
    st.write({"method": result["mode"], "source": result["index"], "model": result["model"], "requirements": result.get("requirements"),
              "job pool": pool, "notes for the agent": result["preferences"], "LLM calls": run["llm_calls"], "tool calls": run["tool_calls"]})
    with st.expander("Agent trace"):
        st.json(run["trace"])
    if run.get("error"):
        st.write("Run error:", run["error"])
    if st.button("Clear results"):
        del st.session_state["search_result"]
        st.rerun()
