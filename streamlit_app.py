"""
streamlit_app.py

Day 8: the interactive "floating GPT" front-end for RepoCoder-Agent.
Wraps the full pipeline built in Days 1-7 — paste a bug report, watch the
planner retrieve context, the localizer rank suspects, and the patch
generator + Docker sandbox + reflection loop attempt a real, tested fix.

This does NOT reimplement any pipeline logic — it only calls the existing
agents and renders their output live.

Usage:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from bug_localizer_agent import BugLocalizerAgent
from planner_agent import PlannerAgent
from reflection_agent import run_patch_with_reflection
from test_runner import build_image_if_needed

st.set_page_config(page_title="RepoCoder-Agent", page_icon=":mag:", layout="wide")


@st.cache_resource(show_spinner=False)
def get_planner():
    return PlannerAgent()


@st.cache_resource(show_spinner=False)
def get_localizer():
    return BugLocalizerAgent()


@st.cache_resource(show_spinner=False)
def ensure_docker_image():
    build_image_if_needed()
    return True


st.title("RepoCoder-Agent")
st.caption(
    "Multi-agent bug localization and automated patch generation for the requests library. "
    "Runs entirely on local models (Qwen2.5-Coder via Ollama) — no cloud API calls."
)

with st.form("issue_form"):
    issue_query = st.text_area(
        "Describe the bug",
        placeholder="e.g. session cookies are not persisted across requests",
        height=80,
    )
    col1, col2 = st.columns(2)
    with col1:
        test_file = st.text_input("Test file to run", value="tests/test_requests.py")
    with col2:
        test_keyword = st.text_input("Pytest -k filter (leave blank to run whole file)", value="")
    submitted = st.form_submit_button("Diagnose and fix", type="primary")

if submitted and issue_query.strip():
    with st.spinner("Loading models (first run downloads/caches embeddings)..."):
        planner = get_planner()
        localizer = get_localizer()
        ensure_docker_image()

    # ---------- Step 1: Planner ----------
    st.subheader("1. Retrieving context")
    with st.spinner("Running planner agent..."):
        plan_result = planner.run(issue_query)

    st.write(
        f"Retrieved **{len(plan_result['context'])} chunks** "
        f"across **{plan_result['attempt']}** attempt(s). "
        f"Sufficiency verdict: **{plan_result['sufficient']}**"
    )
    if plan_result["current_query"] != issue_query:
        st.caption(f"Reformulated query: _{plan_result['current_query']}_")

    with st.expander("Show retrieved chunks"):
        for c in plan_result["context"]:
            st.markdown(f"- `[{c['relation']}]` **{c['kind']}** `{c['id']}`")

    # ---------- Step 2: Localization ----------
    st.subheader("2. Localizing the bug")
    with st.spinner("Running bug localizer..."):
        loc_result = localizer.localize(issue_query, plan_result["context"])

    if not loc_result["candidates"]:
        st.error("No suspect chunks identified. Try rephrasing the issue.")
        st.stop()

    for i, cand in enumerate(loc_result["candidates"], 1):
        st.markdown(f"**{i}. `{cand['chunk_id']}`** — confidence: `{cand.get('confidence', '?')}`")
        st.caption(cand.get("reasoning", ""))

    if loc_result["discarded_hallucinations"]:
        st.warning(f"Discarded {len(loc_result['discarded_hallucinations'])} hallucinated candidate id(s).")

    # pick first non-test candidate, same guardrail logic as the CLI pipeline
    chunk_by_id = {c["id"]: c for c in plan_result["context"]}
    target_chunk = None
    for cand in loc_result["candidates"]:
        c = chunk_by_id.get(cand["chunk_id"])
        if c and not c["file_path"].startswith("tests/") and c["kind"] != "class":
            target_chunk = c
            break

    if target_chunk is None:
        st.error("Only test-file candidates were identified — refusing to patch a test file.")
        st.stop()

    st.info(f"Patch target: `{target_chunk['id']}`")

    # ---------- Step 3: Patch generation + testing + reflection ----------
    st.subheader("3. Generating and testing a patch")
    test_target = [test_file, "-k", test_keyword] if test_keyword.strip() else test_file

    with st.spinner("Running generate -> test -> reflect -> regenerate loop (this can take a few minutes)..."):
        result = run_patch_with_reflection(issue_query, target_chunk, test_target)

    if result["final_status"] == "passed":
        st.success(f"Patch passed tests after {result['attempt_count']} attempt(s).")
    else:
        st.error(f"No passing patch found after {result['attempt_count']} attempt(s).")

    for a in result["attempts"]:
        with st.expander(f"Attempt {a['attempt']} — {'PASSED' if a.get('test_passed') else 'FAILED'}"):
            if a["stage"] == "generation_failed":
                st.write("Patch generation failed:", a["detail"])
                continue
            st.write("**Explanation:**", a["explanation"])
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Original code**")
                st.code(target_chunk["code"], language="python")
            with col_b:
                st.markdown("**Patched code**")
                st.code(a["patched_code"], language="python")
            st.markdown("**Test output**")
            st.code(a["test_result"]["stdout"][-2000:] or "(no output)", language="text")
            if "reflection" in a:
                st.markdown("**Reflection guidance for next attempt**")
                st.info(a["reflection"])

elif submitted:
    st.warning("Please describe a bug first.")