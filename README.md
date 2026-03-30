# ATS Prompt Lab

A Streamlit proof-of-concept for AI-powered resume shortlisting — focused on prompt transparency, local PDF parsing, and token-efficient screening using the **Gemini API**.

---

## Overview

Traditional ATS tools are black boxes. This PoC flips that by making every screening decision visible and editable — you see the prompts, the scoring rubric, token costs, and latency after every API call.

The app implements a **3-call Gemini pipeline**:

| Call | Purpose |
|------|---------|
| **Call 1** | Analyze the JD — extract criteria, flag vague/conflicting requirements, generate gap questions |
| **Call 2** | Build the evaluation prompt and scoring rubric from recruiter-reviewed criteria |
| **Call 3** | Score resumes against the rubric and return ranked candidates with reasoning |

---

## Features

- **JD quality analysis** — flags conflicting requirements, vague language, and inferred (not explicit) criteria
- **Editable screening criteria** — review and tweak every baseline filter, P0 signal, and red flag before finalizing
- **Gap questions** — up to 4 role-specific clarifying questions that update the final rubric
- **Subjective language detector** — catches phrases like "team player" and suggests measurable proxies
- **Local PDF parsing** — resumes parsed entirely on-device using `pdfplumber` + rule-based extraction; zero resume tokens in Call 1/2
- **Smart batching** — if ≤ 15 readable resumes, evaluates all; for larger uploads, selects a representative sample across top, middle, and bottom keyword-overlap bands
- **Token + latency tracking** — input tokens, output tokens, total tokens, and latency shown after every Gemini call, with a session running total
- **Prompt sandbox** — test any prompt variant outside the main flow with raw JSON output visible

---

## Project Structure

```
ats_poc/
├── gemini_client.py      # Gemini SDK wrapper — structured calls, token tracking, JSON extraction
├── prompts.py            # All 3 system/user prompt templates
├── resume_parser.py      # PDF text extraction and rule-based resume JSON builder
└── sample_selection.py   # Keyword extraction and representative sample selection

app.py                    # Streamlit app — UI, session state, tab layout
requirements.txt          # Python dependencies
.env.example              # Environment variable reference
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- A [Google Gemini API key](https://aistudio.google.com/app/apikey)

### Installation

```bash
git clone https://github.com/rish106-hub/ats-prompt-lab.git
cd ats-prompt-lab

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Edit .env and add your Gemini API key:
# GOOGLE_API_KEY=your_key_here
```

Alternatively, paste your key directly in the app sidebar at runtime — no restart needed.

### Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Workflow

```
1. Paste or upload a JD PDF
        ↓
2. Run Call 1 → review extracted criteria, answer gap questions
        ↓
3. Edit baseline filters, P0 signals, and red flags in plain English
        ↓
4. Run Call 2 → review the generated evaluation prompt and scoring rubric
        ↓
5. Upload resume PDFs → local parsing, no API tokens used
        ↓
6. Run Call 3 → ranked results with scores, reasoning, and confidence levels
```

---

## Screening Output

Each evaluated candidate gets:

- `baseline_pass` — hard filter pass/fail
- `p0_score` — weighted score (0–100) on priority signals
- `overall_score` — final composite score
- `classification` — `P0`, `Baseline`, or `Reject`
- `reasoning` — 2–3 sentence specific explanation
- `confidence` — `high`, `medium`, or `low` (low = flagged for human review)

---

## Prompt Templates

All prompts live in `ats_poc/prompts.py` and are fully editable from the sidebar during a session. The templates use `{{VARIABLE}}` placeholders replaced at runtime.

---

## Tech Stack

| Layer | Library |
|-------|---------|
| UI | [Streamlit](https://streamlit.io) |
| LLM | [Google Gemini 1.5 Flash](https://ai.google.dev/) via `google-generativeai` |
| PDF parsing | [pdfplumber](https://github.com/jsvine/pdfplumber) |
| Config | [python-dotenv](https://github.com/theskumar/python-dotenv) |

---

## Limitations

- Mode 1 only (JD vs. batch). Iterative refinement from recruiter feedback (Mode 2) is not yet implemented.
- Rule-based resume parsing works best on clean, text-based PDFs. Image-heavy or heavily styled PDFs may parse poorly.
- Default evaluation batch is capped at 5 resumes to conserve API quota during development (`EVALUATION_LIMIT` in `app.py`).

---

## License

MIT
