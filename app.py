from __future__ import annotations

import copy
import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from ats_poc.gemini_client import configure_genai, run_raw_call, run_structured_call
from ats_poc.prompts import (
    CALL_1_SYSTEM,
    CALL_1_TEMPLATE,
    CALL_2_SYSTEM,
    CALL_2_TEMPLATE,
    CALL_3_SYSTEM,
    CALL_3_TEMPLATE,
    CALL_PREVIEW_SYSTEM,
    CALL_PREVIEW_TEMPLATE,
    CALL_SYNTHESIZE_SYSTEM,
    CALL_SYNTHESIZE_TEMPLATE,
    GENERIC_SYSTEM,
)
from ats_poc.resume_parser import extract_text_from_pdf, parse_resume_pdf
from ats_poc.sample_selection import compress_resume, extract_keywords, pick_representative_sample

# Enable caching for better performance


@st.cache_data(show_spinner=False, max_entries=10)
def cached_extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Cached version of PDF text extraction."""
    return extract_text_from_pdf(pdf_bytes)


@st.cache_data(show_spinner=False, max_entries=5)
def cached_parse_resume_pdf(pdf_bytes: bytes) -> dict[str, Any]:
    """Cached version of resume parsing."""
    return parse_resume_pdf(pdf_bytes)


load_dotenv()

st.set_page_config(page_title="ATS Prompt Lab", page_icon="📄", layout="wide")

MODEL_NAME = "gemini-2.0-flash"
EVALUATION_LIMIT = 5  # Reduced from 15 to conserve API quota
SUBJECTIVE_PATTERNS = {
    "team player": "Replace with a measurable proxy such as cross-functional stakeholder work.",
    "good communication": "Replace with a measurable proxy such as client-facing or presentation experience.",
    "culture fit": "Replace with a measurable proxy such as prior startup or enterprise environment exposure.",
    "self starter": "Replace with a measurable proxy such as 0-to-1 project ownership.",
    "leadership presence": "Replace with a measurable proxy such as team size led or hiring responsibility.",
}


def init_session_state() -> None:
    defaults = {
        "call_1_template": CALL_1_TEMPLATE,
        "call_2_template": CALL_2_TEMPLATE,
        "call_3_template": CALL_3_TEMPLATE,
        "generic_system": GENERIC_SYSTEM,
        "api_usage_history": [],
        "last_usage_by_label": {},
        "token_totals": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "runtime_api_key": "",
        "jd_text": "",
        "jd_analysis": None,
        "edited_requirements": None,
        "gap_answers": {},
        "manual_notes": "",
        "final_config": None,
        "resume_batch": [],
        "selected_sample": [],
        "sample_strategy": "",
        "resume_parse_summary": None,
        "sample_results": None,
        "sandbox_raw_output": "",
        "sandbox_parsed_json": None,
        # --- iterative preview loop ---
        "preview_loop_state": "idle",
        "base_criteria": None,
        "synthesized_config": None,
        "preview_resumes": [],
        "preview_field_results": None,
        "extra_params_history": [],
        "current_include_input": "",
        "current_exclude_input": "",
        "preview_iteration_count": 0,
        "preview_seen_files": set(),
        "iteration_history": [],
        "call_preview_template": CALL_PREVIEW_TEMPLATE,
        "call_synthesize_template": CALL_SYNTHESIZE_TEMPLATE,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_usage(label: str, usage: dict[str, Any]) -> None:
    st.session_state.api_usage_history.append({"label": label, **usage})
    st.session_state.last_usage_by_label[label] = usage
    for key in st.session_state.token_totals:
        st.session_state.token_totals[key] += usage.get(key, 0)


def render_usage_metrics(label: str, usage: dict[str, Any]) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"{label} Input", usage["input_tokens"])
    col2.metric(f"{label} Output", usage["output_tokens"])
    col3.metric(f"{label} Total", usage["total_tokens"])
    col4.metric(f"{label} Latency", f"{usage.get('latency_seconds', 0)}s")

    totals = st.session_state.token_totals
    st.caption(
        f"Running session total | input: {totals['input_tokens']} | output: {totals['output_tokens']} | total: {totals['total_tokens']}"
    )


def render_persisted_usage(label: str) -> None:
    usage = st.session_state.last_usage_by_label.get(label)
    if usage:
        render_usage_metrics(label, usage)


def resolve_api_key() -> str | None:
    if st.session_state.runtime_api_key:
        return st.session_state.runtime_api_key
    if os.getenv("GOOGLE_API_KEY"):
        return os.getenv("GOOGLE_API_KEY")
    if os.getenv("GEMINI_API_KEY"):
        return os.getenv("GEMINI_API_KEY")
    return st.secrets.get("GOOGLE_API_KEY", None) or st.secrets.get("GEMINI_API_KEY", None)


def api_key_available() -> bool:
    return bool(resolve_api_key())


def clone_requirements(jd_analysis: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(jd_analysis)


def parse_multiline_list(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.splitlines() if item.strip()]


def parse_signal_lines(raw_value: str) -> list[dict[str, str]]:
    signals = []
    for line in raw_value.splitlines():
        if not line.strip():
            continue
        if "|" in line:
            signal, why = [part.strip() for part in line.split("|", 1)]
        else:
            signal, why = line.strip(), ""
        signals.append({"signal": signal, "why_it_matters": why})
    return signals


def find_unmeasurable_requirements(requirements: dict[str, Any], manual_notes: str) -> list[str]:
    haystacks = [
        manual_notes,
        requirements.get("baseline", {}).get("education", ""),
        requirements.get("baseline", {}).get("domain_required", ""),
        "\n".join(requirements.get("baseline", {}).get("skills_required", [])),
        "\n".join(requirements.get("baseline", {}).get("other_hard_filters", [])),
        "\n".join(item.get("signal", "") for item in requirements.get("p0_signals", [])),
        "\n".join(requirements.get("red_flags", [])),
    ]
    combined = "\n".join(haystacks).lower()
    warnings = []
    for phrase, suggestion in SUBJECTIVE_PATTERNS.items():
        if phrase in combined:
            warnings.append(f"{phrase}: {suggestion}")
    return warnings


def build_manual_edit_payload(original: dict[str, Any], edited: dict[str, Any], notes: str) -> dict[str, Any]:
    return {
        "original_requirements": original,
        "edited_requirements": edited,
        "notes": notes,
    }


PREVIEW_BATCH_SIZE = 6


def run_preview_call2_silent(criteria: dict[str, Any]) -> dict[str, Any]:
    """Run Call 2 silently with no gap answers or manual edits."""
    manual_payload = build_manual_edit_payload(criteria, criteria, "")
    parsed_json, _raw, usage, _prompt = run_structured_call(
        model_name=MODEL_NAME,
        system_instruction=CALL_2_SYSTEM,
        template=st.session_state.call_2_template,
        replacements={
            "CALL_1_JSON_OUTPUT": criteria,
            "ROHAN_GAP_ANSWERS": {},
            "ROHAN_EDITS": manual_payload,
        },
    )
    add_usage("Preview Call 2", usage)
    st.session_state.synthesized_config = parsed_json
    return parsed_json


def run_preview_scoring(criteria_config: dict[str, Any], resumes: list[dict[str, Any]]) -> dict[str, Any]:
    """Run field-level scoring on 5-6 resumes using CALL_PREVIEW."""
    resume_jsons = [item["resume_json"] for item in resumes]
    parsed_json, _raw, usage, _prompt = run_structured_call(
        model_name=MODEL_NAME,
        system_instruction=CALL_PREVIEW_SYSTEM,
        template=st.session_state.call_preview_template,
        replacements={
            "CRITERIA_JSON": criteria_config,
            "RESUME_JSON_ARRAY": resume_jsons,
        },
    )
    add_usage("Preview Score", usage)
    st.session_state.preview_field_results = parsed_json
    return parsed_json


def run_synthesis() -> dict[str, Any]:
    """Synthesize base criteria + all extra params into a new rubric."""
    parsed_json, _raw, usage, _prompt = run_structured_call(
        model_name=MODEL_NAME,
        system_instruction=CALL_SYNTHESIZE_SYSTEM,
        template=st.session_state.call_synthesize_template,
        replacements={
            "BASE_CRITERIA_JSON": st.session_state.base_criteria,
            "EXTRA_PARAMS_HISTORY": st.session_state.extra_params_history,
            "PREVIEW_RESULTS_JSON": st.session_state.preview_field_results or {},
        },
    )
    add_usage("Synthesis", usage)
    st.session_state.synthesized_config = parsed_json
    return parsed_json


def choose_preview_batch() -> None:
    """Pick 5-6 unseen readable resumes for preview."""
    config = st.session_state.synthesized_config or st.session_state.base_criteria
    keywords = extract_keywords(st.session_state.jd_text, st.session_state.base_criteria, config)
    readable = [r for r in st.session_state.resume_batch if r.get("quality", {}).get("readable")]

    unseen = [r for r in readable if r["file_name"] not in st.session_state.preview_seen_files]
    if not unseen:
        st.session_state.preview_seen_files = set()
        unseen = readable

    if len(unseen) <= PREVIEW_BATCH_SIZE:
        batch = unseen
    else:
        batch = pick_representative_sample(unseen, keywords, sample_size=PREVIEW_BATCH_SIZE)

    for r in batch:
        st.session_state.preview_seen_files.add(r["file_name"])
    st.session_state.preview_resumes = batch


def run_preview_iteration() -> None:
    """Orchestrate one full preview iteration."""
    st.session_state.iteration_history.append({
        "iteration": st.session_state.preview_iteration_count,
        "extra_params_snapshot": list(st.session_state.extra_params_history),
        "results_snapshot": st.session_state.preview_field_results,
    })

    if st.session_state.extra_params_history:
        run_synthesis()
    elif st.session_state.synthesized_config is None:
        run_preview_call2_silent(st.session_state.base_criteria)

    choose_preview_batch()
    run_preview_scoring(st.session_state.synthesized_config, st.session_state.preview_resumes)
    st.session_state.preview_iteration_count += 1
    st.session_state.preview_loop_state = "showing_results"


def accept_and_run_full_eval() -> None:
    """Accept preview criteria and transition to full evaluation."""
    st.session_state.final_config = st.session_state.synthesized_config
    st.session_state.edited_requirements = clone_requirements(st.session_state.base_criteria)
    choose_resume_batch()
    st.session_state.preview_loop_state = "done"


def render_sidebar() -> None:
    st.sidebar.header("Configuration")
    st.sidebar.write(f"Model: `{MODEL_NAME}`")
    st.session_state.runtime_api_key = st.sidebar.text_input(
        "Gemini API Key",
        value=st.session_state.runtime_api_key,
        type="password",
        help="Optional. Leave blank to use `.env` or Streamlit secrets.",
    )
    api_present = api_key_available()
    st.sidebar.write(f"API key detected: {'Yes' if api_present else 'No'}")
    st.sidebar.caption(f"Current test batch size: {EVALUATION_LIMIT} resumes")

    if st.sidebar.button("Reset Session State", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    with st.sidebar.expander("Prompt Templates", expanded=False):
        st.session_state.call_1_template = st.text_area(
            "Call 1 Template",
            value=st.session_state.call_1_template,
            height=260,
        )
        st.session_state.call_2_template = st.text_area(
            "Call 2 Template",
            value=st.session_state.call_2_template,
            height=260,
        )
        st.session_state.call_3_template = st.text_area(
            "Call 3 Template",
            value=st.session_state.call_3_template,
            height=260,
        )
        st.session_state.call_preview_template = st.text_area(
            "Preview Scoring Template",
            value=st.session_state.call_preview_template,
            height=260,
        )
        st.session_state.call_synthesize_template = st.text_area(
            "Synthesis Template",
            value=st.session_state.call_synthesize_template,
            height=260,
        )

    with st.sidebar.expander("Token History", expanded=True):
        totals = st.session_state.token_totals
        st.metric("Session Total", totals["total_tokens"])
        st.write(
            {
                "input_tokens": totals["input_tokens"],
                "output_tokens": totals["output_tokens"],
            }
        )
        if st.session_state.api_usage_history:
            st.dataframe(st.session_state.api_usage_history, use_container_width=True)


def render_run_overview() -> None:
    st.subheader("Run Overview")
    totals = st.session_state.token_totals
    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Session Input Tokens", totals["input_tokens"])
    top2.metric("Session Output Tokens", totals["output_tokens"])
    top3.metric("Session Total Tokens", totals["total_tokens"])
    top4.metric(
        "Readable Resumes",
        sum(1 for item in st.session_state.resume_batch if item.get("quality", {}).get("readable")),
    )

    def step_row(title: str, complete: bool, detail: str) -> None:
        state = "Complete" if complete else "Pending"
        icon = "COMPLETE" if complete else "PENDING"
        st.markdown(f"**{title}**  \n{icon} • {detail} • {state}")

    col1, col2, col3 = st.columns(3)
    with col1:
        step_row("Step 1: JD Loaded", bool(st.session_state.jd_text.strip()), "Local input")
        step_row("Step 2: JD Analyzed", st.session_state.jd_analysis is not None, "Gemini Call 1")
    with col2:
        step_row("Step 3: Criteria Reviewed", st.session_state.edited_requirements is not None, "HM review")
        step_row("Step 4: Prompt Finalized", st.session_state.final_config is not None, "Gemini Call 2")
    with col3:
        step_row("Step 5: Resumes Parsed", bool(st.session_state.resume_batch), "Local PDF parsing")
        step_row("Step 6: 15 Resume Evaluation", st.session_state.sample_results is not None, "Gemini Call 3")

    call_labels = ["Call 1", "Call 2", "Call 3"]
    metric_columns = st.columns(3)
    for index, label in enumerate(call_labels):
        usage = st.session_state.last_usage_by_label.get(label)
        metric_columns[index].metric(f"{label} Tokens", usage["total_tokens"] if usage else 0)


def analyze_jd(jd_text: str) -> None:
    parsed_json, raw_text, usage, final_prompt = run_structured_call(
        model_name=MODEL_NAME,
        system_instruction=CALL_1_SYSTEM,
        template=st.session_state.call_1_template,
        replacements={"JD_TEXT": jd_text},
    )
    st.session_state.jd_analysis = parsed_json
    st.session_state.edited_requirements = clone_requirements(parsed_json)
    st.session_state.sample_results = None
    st.session_state.final_config = None
    st.session_state.call_1_raw = raw_text
    st.session_state.call_1_prompt = final_prompt
    add_usage("Call 1", usage)


def finalize_prompt() -> None:
    manual_payload = build_manual_edit_payload(
        st.session_state.jd_analysis,
        st.session_state.edited_requirements,
        st.session_state.manual_notes,
    )
    parsed_json, raw_text, usage, final_prompt = run_structured_call(
        model_name=MODEL_NAME,
        system_instruction=CALL_2_SYSTEM,
        template=st.session_state.call_2_template,
        replacements={
            "CALL_1_JSON_OUTPUT": st.session_state.edited_requirements,
            "ROHAN_GAP_ANSWERS": st.session_state.gap_answers,
            "ROHAN_EDITS": manual_payload,
        },
    )
    st.session_state.final_config = parsed_json
    st.session_state.call_2_raw = raw_text
    st.session_state.call_2_prompt = final_prompt
    add_usage("Call 2", usage)


def run_sample_evaluation() -> None:
    required_fields = st.session_state.final_config.get("required_resume_fields", [])
    compressed_resumes = [
        compress_resume(item["resume_json"], required_fields) for item in st.session_state.selected_sample
    ]
    parsed_json, raw_text, usage, final_prompt = run_structured_call(
        model_name=MODEL_NAME,
        system_instruction=CALL_3_SYSTEM,
        template=st.session_state.call_3_template,
        replacements={
            "FINAL_EVALUATION_PROMPT": st.session_state.final_config["final_evaluation_prompt"],
            "SCORING_RUBRIC_JSON": st.session_state.final_config["scoring_rubric"],
            "ARRAY_OF_15_COMPRESSED_RESUMES": compressed_resumes,
        },
    )
    st.session_state.sample_results = parsed_json
    st.session_state.call_3_raw = raw_text
    st.session_state.call_3_prompt = final_prompt
    add_usage("Call 3", usage)


def execute_call_1() -> None:
    if not st.session_state.jd_text.strip():
        st.error("Provide JD text before running Call 1.")
        return
    if not api_key_available():
        st.error("Add a Gemini API key in the sidebar or `.env` before running Call 1.")
        return

    with st.status("Running Call 1: JD analysis", expanded=True) as status:
        status.write("Sending the JD to Gemini for structured requirement extraction.")
        configure_genai(api_key=resolve_api_key())
        analyze_jd(st.session_state.jd_text)
        usage = st.session_state.last_usage_by_label.get("Call 1", {})
        status.write(
            f"Completed Call 1 with {usage.get('input_tokens', 0)} input tokens and {usage.get('output_tokens', 0)} output tokens."
        )
        status.update(label="Call 1 complete", state="complete")
    render_persisted_usage("Call 1")


def execute_call_2() -> None:
    if not api_key_available():
        st.error("Add a Gemini API key in the sidebar or `.env` before running Call 2.")
        return

    with st.status("Running Call 2: screening configuration", expanded=True) as status:
        status.write("Finalizing the evaluation prompt, required resume fields, and scoring rubric.")
        configure_genai(api_key=resolve_api_key())
        finalize_prompt()
        usage = st.session_state.last_usage_by_label.get("Call 2", {})
        status.write(f"Completed Call 2 with {usage.get('total_tokens', 0)} total tokens.")
        status.update(label="Call 2 complete", state="complete")
    render_persisted_usage("Call 2")


def execute_call_3() -> None:
    if not st.session_state.selected_sample:
        st.error("Select or prepare a resume batch before running Call 3.")
        return
    if not api_key_available():
        st.error("Add a Gemini API key in the sidebar or `.env` before running Call 3.")
        return

    with st.status("Running Call 3: evaluating resumes", expanded=True) as status:
        status.write(f"Scoring {len(st.session_state.selected_sample)} resumes against the finalized rubric.")
        configure_genai(api_key=resolve_api_key())
        run_sample_evaluation()
        usage = st.session_state.last_usage_by_label.get("Call 3", {})
        status.write(f"Completed Call 3 with {usage.get('total_tokens', 0)} total tokens.")
        status.update(label="Call 3 complete", state="complete")
    render_persisted_usage("Call 3")


def execute_sandbox(system_instruction: str, user_prompt: str) -> None:
    if not api_key_available():
        st.error("Add a Gemini API key in the sidebar or `.env` before running the sandbox.")
        return

    with st.status("Running sandbox prompt", expanded=True) as status:
        status.write("Submitting the prompt to Gemini and capturing the raw JSON output.")
        configure_genai(api_key=resolve_api_key())
        raw_text, usage, parsed_json = run_raw_call(
            model_name=MODEL_NAME,
            system_instruction=system_instruction,
            user_prompt=user_prompt,
        )
        st.session_state.sandbox_raw_output = raw_text
        st.session_state.sandbox_parsed_json = parsed_json
        add_usage("Sandbox", usage)
        status.write(f"Sandbox call completed with {usage.get('total_tokens', 0)} total tokens.")
        status.update(label="Sandbox call complete", state="complete")
    render_persisted_usage("Sandbox")


def parse_uploaded_resumes(uploaded_resumes: list[Any]) -> None:
    parsed_batch = []
    with st.status("Parsing uploaded resume PDFs", expanded=True) as status:
        status.write(f"Starting local parsing for {len(uploaded_resumes)} files. No API tokens are used here.")
        for index, uploaded_file in enumerate(uploaded_resumes, start=1):
            try:
                parsed_batch.append(cached_parse_resume_pdf(uploaded_file.getvalue()))
            except Exception as exc:
                parsed_batch.append(
                    {
                        "file_name": uploaded_file.name,
                        "raw_text": "",
                        "resume_json": {
                            "name": "",
                            "education": [],
                            "work_experience": [],
                            "skills": [],
                            "certifications": [],
                            "projects": [],
                            "publications": [],
                            "github_url": "",
                            "total_experience_years": 0,
                            "career_gaps_months": [],
                        },
                        "quality": {"readable": False, "score": 0, "reasons": [str(exc)]},
                    }
                )
            status.write(f"Parsed {index}/{len(uploaded_resumes)}: {uploaded_file.name}")
        status.update(label="Resume parsing complete", state="complete")

    st.session_state.resume_batch = parsed_batch
    readable_count = sum(1 for item in parsed_batch if item["quality"]["readable"])
    st.session_state.resume_parse_summary = {
        "uploaded": len(parsed_batch),
        "readable": readable_count,
        "unreadable": len(parsed_batch) - readable_count,
        "api_tokens": 0,
    }


def choose_resume_batch() -> None:
    keywords = extract_keywords(
        st.session_state.jd_text,
        st.session_state.edited_requirements,
        st.session_state.final_config,
    )
    readable_resumes = [item for item in st.session_state.resume_batch if item["quality"]["readable"]]
    if len(readable_resumes) <= EVALUATION_LIMIT:
        st.session_state.selected_sample = readable_resumes
        st.session_state.sample_strategy = "Using all readable uploaded resumes because the batch is 15 or fewer."
        return

    st.session_state.selected_sample = pick_representative_sample(
        st.session_state.resume_batch,
        keywords,
        sample_size=EVALUATION_LIMIT,
    )
    st.session_state.sample_strategy = (
        "Using a representative sample of 15 resumes: top, middle, and bottom keyword-overlap bands."
    )


def render_jd_input() -> None:
    st.subheader("1. JD Input")
    uploaded_jd = st.file_uploader("Upload JD PDF", type=["pdf"], accept_multiple_files=False)
    if uploaded_jd and not st.session_state.jd_text:
        with st.spinner("Extracting text from PDF..."):
            st.session_state.jd_text = cached_extract_text_from_pdf(uploaded_jd.getvalue())

    st.session_state.jd_text = st.text_area(
        "Paste the job description",
        value=st.session_state.jd_text,
        height=280,
        placeholder="Paste the JD here or upload a PDF above.",
    )

    if st.button("Run Call 1: Analyze JD", type="primary", use_container_width=True):
        try:
            execute_call_1()
        except Exception as exc:
            st.exception(exc)


def render_quality_check() -> None:
    jd_analysis = st.session_state.jd_analysis
    if not jd_analysis:
        return

    st.subheader("2. Quality Check")
    render_persisted_usage("Call 1")
    if jd_analysis.get("jd_quality_score", 0) < 6:
        st.warning("JD quality is below 6. Screening may be unreliable without tightening the requirements.")
    if jd_analysis.get("conflicting_requirements"):
        st.error("Conflicting requirements detected. Resolve them before continuing to Call 2.")
    if jd_analysis.get("inferred_not_explicit"):
        st.info("Some criteria were inferred from context, not explicitly stated in the JD.")
    if jd_analysis.get("vague_requirements"):
        st.warning("Unverifiable requirements found: " + ", ".join(jd_analysis["vague_requirements"]))

    with st.expander("Raw Call 1 JSON", expanded=False):
        st.json(jd_analysis)
    with st.expander("Raw Call 1 Prompt", expanded=False):
        st.code(st.session_state.get("call_1_prompt", ""), language="text")
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Baseline Filters",
        len(jd_analysis.get("baseline", {}).get("skills_required", []))
        + len(jd_analysis.get("baseline", {}).get("other_hard_filters", [])),
    )
    col2.metric("P0 Signals", len(jd_analysis.get("p0_signals", [])))
    col3.metric("Red Flags", len(jd_analysis.get("red_flags", [])))


def render_plain_english_editor() -> None:
    requirements = st.session_state.edited_requirements
    if not requirements:
        return

    st.subheader("3. Review Criteria")

    left, right = st.columns(2)
    with left:
        requirements["role"] = st.text_input("Role", value=requirements.get("role", ""))
        requirements["role_type"] = st.selectbox(
            "Role Type",
            options=["engineering", "product", "sales", "design", "data", "other"],
            index=["engineering", "product", "sales", "design", "data", "other"].index(
                requirements.get("role_type", "other")
            )
            if requirements.get("role_type", "other") in ["engineering", "product", "sales", "design", "data", "other"]
            else 5,
        )
        baseline = requirements.setdefault("baseline", {})
        baseline["experience_years_min"] = st.number_input(
            "Minimum Experience (years)",
            min_value=0,
            max_value=40,
            value=int(baseline.get("experience_years_min", 0) or 0),
        )
        baseline["education"] = st.text_input("Education Filter", value=baseline.get("education", ""))
        baseline["domain_required"] = st.text_input(
            "Domain Requirement",
            value=baseline.get("domain_required", ""),
        )

    with right:
        baseline["skills_required"] = parse_multiline_list(
            st.text_area(
                "Baseline Skills Required (one per line)",
                value="\n".join(baseline.get("skills_required", [])),
                height=160,
            )
        )
        baseline["other_hard_filters"] = parse_multiline_list(
            st.text_area(
                "Other Hard Filters (one per line)",
                value="\n".join(baseline.get("other_hard_filters", [])),
                height=160,
            )
        )

    requirements["p0_signals"] = parse_signal_lines(
        st.text_area(
            "P0 Signals (`signal | why it matters` per line)",
            value="\n".join(
                f"{item.get('signal', '')} | {item.get('why_it_matters', '')}"
                for item in requirements.get("p0_signals", [])
            ),
            height=180,
        )
    )
    requirements["red_flags"] = parse_multiline_list(
        st.text_area(
            "Red Flags (one per line)",
            value="\n".join(requirements.get("red_flags", [])),
            height=120,
        )
    )
    st.session_state.manual_notes = st.text_area(
        "Additional recruiter notes",
        value=st.session_state.manual_notes,
        height=120,
        placeholder="Optional free-text edits. Keep them measurable.",
    )

    warnings = find_unmeasurable_requirements(requirements, st.session_state.manual_notes)
    for warning in warnings:
        st.warning(f"Unmeasurable requirement detected: {warning}")

    with st.expander("Full Prompt Logic View", expanded=False):
        st.json(requirements)


def render_gap_questions() -> None:
    requirements = st.session_state.edited_requirements
    if not requirements:
        return

    st.subheader("4. Gap Questions")
    gap_questions = requirements.get("gap_questions", [])[:4]
    if not gap_questions:
        st.info("No gap questions were generated. You can go straight to Call 2.")

    for index, question in enumerate(gap_questions):
        question_key = f"gap_answer_{index}"
        default_option = question.get("options", ["Not required"])[0]
        selected = st.radio(
            question.get("question", f"Gap question {index + 1}"),
            options=question.get("options", ["Not required"]),
            key=question_key,
            horizontal=True,
        )
        st.session_state.gap_answers[question.get("field_target", question_key)] = {
            "question": question.get("question", ""),
            "selected_option": selected or default_option,
            "impact": question.get("impacts", ""),
        }

    conflicts = requirements.get("conflicting_requirements", [])
    conflicts_resolved = True
    if conflicts:
        conflicts_resolved = st.checkbox(
            "I resolved the conflicts above in the edited criteria.",
            value=False,
        )

    warnings = find_unmeasurable_requirements(requirements, st.session_state.manual_notes)
    disabled = bool(conflicts and not conflicts_resolved) or bool(warnings)
    if st.button("Run Call 2: Finalize Screening Logic", disabled=disabled, use_container_width=True):
        try:
            execute_call_2()
        except Exception as exc:
            st.exception(exc)


def render_resume_upload() -> None:
    if not st.session_state.final_config:
        return

    st.subheader("5. Upload Resumes and Pick Validation Sample")
    render_persisted_usage("Call 2")
    st.write(st.session_state.final_config.get("screening_summary", ""))
    st.caption("Required resume fields: " + ", ".join(st.session_state.final_config.get("required_resume_fields", [])))
    st.info(
        "Current test mode is JD vs 15 resumes. If you upload 15 or fewer readable resumes, the app evaluates all of them. "
        "If you upload more than 15, it picks a representative 15."
    )

    with st.expander("Scoring Rubric", expanded=False):
        st.json(st.session_state.final_config.get("scoring_rubric", {}))
    with st.expander("Raw Call 2 JSON", expanded=False):
        st.json(st.session_state.final_config)

    uploaded_resumes = st.file_uploader(
        "Upload resume PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if st.button("Parse Uploaded Resumes", use_container_width=True):
        if not uploaded_resumes:
            st.error("Upload at least one resume PDF.")
            return
        parse_uploaded_resumes(uploaded_resumes)
        choose_resume_batch()

    if st.session_state.resume_batch:
        summary = st.session_state.resume_parse_summary or {}
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Uploaded", summary.get("uploaded", 0))
        col2.metric("Readable", summary.get("readable", 0))
        col3.metric("Unreadable", summary.get("unreadable", 0))
        col4.metric("Parse API Tokens", summary.get("api_tokens", 0))

        if st.button("Refresh 15-Resume Test Batch", use_container_width=True):
            choose_resume_batch()

        unreadable = [item for item in st.session_state.resume_batch if not item["quality"]["readable"]]
        if unreadable:
            with st.expander("Unreadable / Low Parse Quality Resumes", expanded=False):
                st.json(
                    [
                        {
                            "file_name": item["file_name"],
                            "reasons": item["quality"]["reasons"],
                        }
                        for item in unreadable
                    ]
                )


def render_sample_results() -> None:
    sample = st.session_state.selected_sample
    if not sample:
        return

    st.subheader("6. Sample Evaluation")
    if st.session_state.sample_strategy:
        st.caption(st.session_state.sample_strategy)
    st.dataframe(
        [
            {
                "file_name": item["file_name"],
                "candidate_name": item["resume_json"].get("name", ""),
                "keyword_score": item.get("keyword_score", 0),
                "total_experience_years": item["resume_json"].get("total_experience_years", 0),
            }
            for item in sample
        ],
        use_container_width=True,
    )

    if st.button("Run Call 3: Evaluate Sample", type="primary", use_container_width=True):
        try:
            execute_call_3()
        except Exception as exc:
            st.exception(exc)

    results = st.session_state.sample_results
    if not results:
        return

    render_persisted_usage("Call 3")
    st.dataframe(results.get("results", []), use_container_width=True)
    st.json(results.get("summary", {}))

    st.markdown("**Hiring Manager Feedback**")
    for index, result in enumerate(results.get("results", [])):
        col1, col2 = st.columns([1, 3])
        with col1:
            st.selectbox(
                f"{result.get('candidate_name', 'Candidate')} verdict",
                options=["Agree", "Disagree", "Needs Review"],
                key=f"feedback_decision_{index}",
            )
        with col2:
            st.text_input(
                f"{result.get('candidate_name', 'Candidate')} comment",
                key=f"feedback_comment_{index}",
                placeholder="Optional disagreement reason for later Mode 2 refinement.",
            )

    with st.expander("Raw Call 3 JSON", expanded=False):
        st.json(results)
    with st.expander("Raw Call 3 Prompt", expanded=False):
        st.code(st.session_state.get("call_3_prompt", ""), language="text")


def render_resume_upload_early() -> None:
    """Upload and parse resumes right after Call 1, before the preview loop."""
    if st.session_state.resume_batch:
        st.success(
            f"Resumes ready: {sum(1 for r in st.session_state.resume_batch if r.get('quality', {}).get('readable'))} "
            f"readable of {len(st.session_state.resume_batch)} uploaded."
        )
        return

    st.subheader("3. Upload Resumes")
    st.info(
        "Upload resumes now so the preview loop can score a sample batch "
        "immediately against the extracted criteria."
    )

    jd = st.session_state.jd_analysis or {}
    baseline = jd.get("baseline", {})
    col1, col2 = st.columns(2)
    with col1:
        skills = baseline.get("skills_required", [])
        if skills:
            st.markdown("**Baseline skills being checked:**")
            for s in skills:
                st.markdown(f"- {s}")
    with col2:
        p0 = jd.get("p0_signals", [])
        if p0:
            st.markdown("**P0 ranking signals:**")
            for p in p0:
                st.markdown(f"- {p.get('signal', '')}")

    uploaded = st.file_uploader(
        "Upload resume PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="early_resume_uploader",
    )
    if st.button("Parse Resumes", use_container_width=True, key="early_parse_btn"):
        if not uploaded:
            st.error("Upload at least one resume PDF.")
            return
        parse_uploaded_resumes(uploaded)
        st.session_state.base_criteria = clone_requirements(st.session_state.jd_analysis)
        if st.session_state.resume_batch:
            st.session_state.preview_loop_state = "ready"
        st.rerun()


def render_field_match_table() -> None:
    """Show per-candidate field-level match breakdown for the current preview batch."""
    results = st.session_state.preview_field_results
    if not results:
        return

    summary = results.get("summary", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Evaluated", summary.get("total_evaluated", 0))
    c2.metric("P0", summary.get("p0_count", 0))
    c3.metric("Baseline", summary.get("baseline_count", 0))
    c4.metric("Reject", summary.get("reject_count", 0))

    render_persisted_usage("Preview Score")

    for candidate in results.get("results", []):
        name = candidate.get("candidate_name", "Unknown")
        cls = candidate.get("classification", "—")
        score = candidate.get("overall_score", 0)
        label = f"{name}  —  {cls}  —  {score}/100"
        with st.expander(label, expanded=False):
            field_rows = []
            for fm in candidate.get("field_matches", []):
                match_icon = {"pass": "✅", "fail": "❌", "partial": "⚠️"}.get(fm.get("match", ""), "—")
                field_rows.append({
                    "Field": fm.get("field", ""),
                    "Resume Value": str(fm.get("value_from_resume", ""))[:120],
                    "Criteria Checked": str(fm.get("criteria_checked", ""))[:120],
                    "Match": f"{match_icon} {fm.get('match', '')}",
                    "Note": fm.get("note", ""),
                })
            if field_rows:
                st.dataframe(field_rows, use_container_width=True)

            if candidate.get("baseline_failures"):
                st.error("Baseline failures: " + ", ".join(candidate["baseline_failures"]))
            if candidate.get("p0_matches"):
                st.success("P0 matches: " + ", ".join(candidate["p0_matches"]))
            if candidate.get("extra_param_matches"):
                st.info("Extra param matches: " + ", ".join(candidate["extra_param_matches"]))
            st.caption(candidate.get("reasoning", ""))


def render_extra_params_input() -> None:
    """Render the include/exclude input and action buttons."""
    st.markdown("---")
    st.markdown("**Refine the criteria for the next batch:**")

    col_a, col_b = st.columns(2)
    with col_a:
        include_val = st.text_area(
            "Also include candidates who...",
            value=st.session_state.current_include_input,
            height=100,
            placeholder="e.g. 2+ years B2B SaaS, experience with enterprise clients",
            key="include_text_area",
        )
        st.session_state.current_include_input = include_val
    with col_b:
        exclude_val = st.text_area(
            "Exclude candidates who...",
            value=st.session_state.current_exclude_input,
            height=100,
            placeholder="e.g. only consulting background, no product experience",
            key="exclude_text_area",
        )
        st.session_state.current_exclude_input = exclude_val

    history = st.session_state.extra_params_history
    if history:
        with st.expander(f"Parameters added so far ({len(history)} iteration(s))", expanded=False):
            for entry in history:
                st.markdown(f"**Iteration {entry['iteration'] + 1}**")
                if entry.get("include"):
                    st.markdown(f"- Include: {entry['include']}")
                if entry.get("exclude"):
                    st.markdown(f"- Exclude: {entry['exclude']}")

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("Re-evaluate with updated parameters", use_container_width=True, type="primary"):
            include = st.session_state.current_include_input.strip()
            exclude = st.session_state.current_exclude_input.strip()
            if include or exclude:
                st.session_state.extra_params_history.append({
                    "iteration": st.session_state.preview_iteration_count,
                    "include": include,
                    "exclude": exclude,
                })
            st.session_state.current_include_input = ""
            st.session_state.current_exclude_input = ""
            st.session_state.preview_loop_state = "running"
            try:
                with st.status("Running preview iteration...", expanded=True) as status:
                    configure_genai(api_key=resolve_api_key())
                    run_preview_iteration()
                    status.update(label="Preview complete", state="complete")
            except Exception as exc:
                st.session_state.preview_loop_state = "showing_results"
                st.exception(exc)
            st.rerun()
    with btn_col2:
        if st.button("Accept & Run Full Evaluation", use_container_width=True):
            accept_and_run_full_eval()
            st.rerun()


def render_iteration_history() -> None:
    """Show the history of preview iterations."""
    history = st.session_state.iteration_history
    if not history:
        return
    with st.expander("Iteration History", expanded=False):
        for entry in history:
            i = entry["iteration"]
            st.markdown(f"**Iteration {i + 1}**")
            params = entry.get("extra_params_snapshot", [])
            active = [p for p in params if p["iteration"] == i]
            if active:
                for p in active:
                    if p.get("include"):
                        st.markdown(f"- Include: {p['include']}")
                    if p.get("exclude"):
                        st.markdown(f"- Exclude: {p['exclude']}")
            snapshot = entry.get("results_snapshot")
            if snapshot and snapshot.get("summary"):
                s = snapshot["summary"]
                st.caption(
                    f"Results: {s.get('p0_count', 0)} P0 / "
                    f"{s.get('baseline_count', 0)} Baseline / "
                    f"{s.get('reject_count', 0)} Reject"
                )
            st.divider()


def render_preview_loop() -> None:
    """Main dispatcher for the iterative preview loop."""
    if not st.session_state.jd_analysis or not st.session_state.resume_batch:
        return

    st.subheader("4. Iterative Preview Loop")
    state = st.session_state.preview_loop_state

    if state == "idle":
        return

    if state == "ready":
        readable = sum(1 for r in st.session_state.resume_batch if r.get("quality", {}).get("readable"))
        st.info(
            f"**{readable} readable resumes** ready. The preview will score "
            f"{min(readable, PREVIEW_BATCH_SIZE)} of them against the extracted "
            "baseline and P0 criteria, showing a field-by-field breakdown. "
            "You can then refine criteria iteratively before running the full evaluation."
        )
        if not api_key_available():
            st.error("Add a Gemini API key in the sidebar before starting the preview.")
            return
        if st.button("Start Preview", use_container_width=True, type="primary"):
            st.session_state.preview_loop_state = "running"
            try:
                with st.status("Running first preview batch...", expanded=True) as status:
                    configure_genai(api_key=resolve_api_key())
                    run_preview_iteration()
                    status.update(label="Preview complete", state="complete")
            except Exception as exc:
                st.session_state.preview_loop_state = "ready"
                st.exception(exc)
            st.rerun()

    elif state == "showing_results":
        st.markdown(f"**Preview — Iteration {st.session_state.preview_iteration_count}**")
        if st.session_state.synthesized_config:
            st.caption(
                st.session_state.synthesized_config.get("screening_summary", "")
            )
        render_field_match_table()
        render_extra_params_input()

    elif state == "done":
        st.success("Preview loop complete — criteria locked in.")
        config = st.session_state.synthesized_config or {}
        notes = config.get("synthesis_notes", "")
        if notes:
            st.info(f"Synthesis summary: {notes}")
        render_iteration_history()


def render_prompt_sandbox() -> None:
    st.subheader("Prompt Sandbox")
    st.caption("Use this to test prompt variants and inspect raw model JSON plus token usage.")
    render_persisted_usage("Sandbox")

    system_instruction = st.text_area(
        "System Instruction",
        value=st.session_state.generic_system,
        height=100,
    )
    user_prompt = st.text_area(
        "User Prompt",
        value=st.session_state.jd_text or st.session_state.call_1_template,
        height=280,
    )

    if st.button("Run Sandbox Prompt", use_container_width=True):
        try:
            execute_sandbox(system_instruction, user_prompt)
        except Exception as exc:
            st.exception(exc)

    if st.session_state.sandbox_raw_output:
        st.code(st.session_state.sandbox_raw_output, language="json")
    if st.session_state.sandbox_parsed_json is not None:
        st.json(st.session_state.sandbox_parsed_json)


def main() -> None:
    init_session_state()
    render_sidebar()

    st.title("AI Resume Shortlisting PoC")
    st.write(
        "Mode 1 only: ingest a JD, generate screening logic, and run Gemini against a 15-resume batch with visible token consumption after each API step."
    )

    flow_tab, sandbox_tab = st.tabs(["Hiring Manager Flow", "Prompt Sandbox"])
    with flow_tab:
        render_run_overview()
        render_jd_input()
        render_quality_check()

        # Early resume upload + iterative preview loop
        if st.session_state.jd_analysis:
            render_resume_upload_early()
        if st.session_state.jd_analysis and st.session_state.resume_batch:
            render_preview_loop()

        # Optional deep-edit path (gap questions + manual criteria editor)
        render_plain_english_editor()
        render_gap_questions()

        # Full evaluation (shown after preview loop accepts, or via manual Call 2 path)
        if st.session_state.final_config:
            render_resume_upload()
            render_sample_results()

    with sandbox_tab:
        render_prompt_sandbox()


if __name__ == "__main__":
    main()
