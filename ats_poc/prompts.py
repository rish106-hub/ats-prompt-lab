"""Default prompt templates for the ATS PoC."""

CALL_1_SYSTEM = (
    "You are a hiring requirements analyst. Your job is to extract "
    "structured screening criteria from a Job Description and identify "
    "what is missing. Return ONLY valid JSON. No markdown. No preamble. "
    "No explanation."
)

CALL_1_TEMPLATE = """
Analyze this Job Description and return the following JSON structure:

JD:
{{JD_TEXT}}

Return this exact JSON structure:

{
  "role": "",
  "role_type": "engineering|product|sales|design|data|other",
  "baseline": {
    "experience_years_min": 0,
    "skills_required": [],
    "education": "",
    "domain_required": "",
    "other_hard_filters": []
  },
  "p0_signals": [
    {
      "signal": "",
      "why_it_matters": ""
    }
  ],
  "red_flags": [],
  "jd_quality_score": 0,
  "vague_requirements": [],
  "inferred_not_explicit": false,
  "conflicting_requirements": [],
  "gap_questions": [
    {
      "question": "",
      "options": ["Option A", "Option B", "Not required"],
      "impacts": "baseline|p0|red_flag",
      "field_target": ""
    }
  ]
}

Rules:
1. gap_questions: max 4. Must be specific to this role and JD.
   Not generic.
   Bad: "Do you need experience?"
   Good: "Should candidates have B2B SaaS experience specifically?"
2. jd_quality_score: 1 to 10. Below 6 means JD too vague to
   screen reliably.
3. vague_requirements: anything unverifiable from a resume.
   Examples: team player, good communication.
4. conflicting_requirements: flag impossible combinations.
5. inferred_not_explicit: true if requirements were inferred
   from context, not stated explicitly.
6. p0_signals: max 5. Ranking boosters only, not filters.
7. red_flags: only verifiable from a resume.
""".strip()

CALL_2_SYSTEM = (
    "You are a resume screening engine configurator. Take hiring "
    "requirements and recruiter inputs and produce a final evaluation "
    "prompt and field selection list. Return ONLY valid JSON. No "
    "markdown. No preamble."
)

CALL_2_TEMPLATE = """
EXTRACTED REQUIREMENTS:
{{CALL_1_JSON_OUTPUT}}

RECRUITER ANSWERS TO GAP QUESTIONS:
{{ROHAN_GAP_ANSWERS}}

RECRUITER MANUAL EDITS:
{{ROHAN_EDITS}}

Return this exact JSON:

{
  "final_evaluation_prompt": "",
  "required_resume_fields": [],
  "scoring_rubric": {
    "baseline_checks": [
      {
        "check": "",
        "resume_field": "",
        "reject_if_missing": true
      }
    ],
    "p0_weights": [
      {
        "signal": "",
        "weight": 0,
        "resume_field": ""
      }
    ],
    "red_flag_checks": [
      {
        "check": "",
        "resume_field": "",
        "deprioritize_if_present": true
      }
    ]
  },
  "screening_summary": ""
}

Rules:
1. final_evaluation_prompt: clear LLM instruction including all
   baseline checks, p0 signals, red flags, and scoring logic.
2. required_resume_fields: ONLY fields actually needed for this
   role from: name, education, work_experience, skills,
   certifications, projects, publications, github_url,
   total_experience_years, career_gaps_months
3. p0 weights must add up to 100.
4. screening_summary: 2 sentence plain English summary for a
   non-technical recruiter.
""".strip()

CALL_3_SYSTEM = (
    "You are a resume screening assistant. Evaluate candidates strictly "
    "against provided criteria. Be objective. Do not infer information "
    "not present in the resume. Return ONLY valid JSON. No markdown. "
    "No preamble."
)

CALL_3_TEMPLATE = """
SCREENING CRITERIA:
{{FINAL_EVALUATION_PROMPT}}

SCORING RUBRIC:
{{SCORING_RUBRIC_JSON}}

RESUMES (relevant fields only):
{{ARRAY_OF_15_COMPRESSED_RESUMES}}

Return this JSON:

{
  "results": [
    {
      "candidate_name": "",
      "baseline_pass": true,
      "baseline_failures": [],
      "p0_score": 0,
      "p0_matches": [],
      "red_flags_found": [],
      "overall_score": 0,
      "classification": "P0|Baseline|Reject",
      "reasoning": "",
      "confidence": "high|medium|low"
    }
  ],
  "summary": {
    "total_evaluated": 0,
    "p0_count": 0,
    "baseline_count": 0,
    "reject_count": 0,
    "low_confidence_count": 0
  }
}

Rules:
1. overall_score: 0 to 100.
2. baseline_pass false = automatic Reject regardless of p0_score.
3. reasoning: 2-3 sentences max. Specific, not generic.
4. confidence low = resume data insufficient to evaluate reliably.
   Flag for human review.
5. Rank results array by overall_score descending.
""".strip()

GENERIC_SYSTEM = "You are a precise prompt testing assistant. Follow the user's instruction exactly."

CALL_PREVIEW_SYSTEM = (
    "You are a resume screening assistant performing field-level analysis. "
    "For each resume, check every relevant field against the provided criteria. "
    "Be specific about which criteria each field satisfies or fails. "
    "Return ONLY valid JSON. No markdown. No preamble."
)

CALL_PREVIEW_TEMPLATE = """
SCREENING CRITERIA (baseline + P0 signals + extra parameters):
{{CRITERIA_JSON}}

RESUMES (full JSON, 5-6 candidates):
{{RESUME_JSON_ARRAY}}

For each resume, return a detailed field-level analysis. Return this exact JSON:

{
  "results": [
    {
      "candidate_name": "",
      "field_matches": [
        {
          "field": "",
          "value_from_resume": "",
          "criteria_checked": "",
          "match": "pass|fail|partial",
          "note": ""
        }
      ],
      "baseline_pass": true,
      "baseline_failures": [],
      "p0_score": 0,
      "p0_matches": [],
      "extra_param_matches": [],
      "overall_score": 0,
      "classification": "P0|Baseline|Reject",
      "reasoning": ""
    }
  ],
  "summary": {
    "total_evaluated": 0,
    "p0_count": 0,
    "baseline_count": 0,
    "reject_count": 0
  }
}

Rules:
1. field_matches: include one entry per resume field that was checked.
   Use the actual field names from the resume JSON: name, education,
   work_experience, skills, certifications, projects, total_experience_years.
2. criteria_checked: quote the specific criterion from SCREENING CRITERIA
   that was applied to this field.
3. match values:
   - "pass" = field clearly satisfies the criterion
   - "fail" = field clearly fails the criterion
   - "partial" = field partially satisfies (e.g. related but not exact skill)
4. extra_param_matches: list which extra parameters (if any) this candidate satisfies.
5. baseline_pass false = automatic Reject regardless of p0_score.
6. overall_score: 0 to 100. baseline_pass false candidates score below 30.
7. reasoning: 2-3 sentences. Reference specific field values, not generalities.
8. Rank results array by overall_score descending.
""".strip()

CALL_SYNTHESIZE_SYSTEM = (
    "You are a hiring criteria synthesis engine. You take a base set of hiring "
    "criteria and a history of recruiter-added include/exclude parameters, and "
    "synthesize them into a single unified evaluation rubric. Preserve all base "
    "criteria. Layer in recruiter preferences as tighter or looser filters and "
    "updated P0 signals. Return ONLY valid JSON. No markdown. No preamble."
)

CALL_SYNTHESIZE_TEMPLATE = """
BASE CRITERIA (from JD analysis — do not remove any of these):
{{BASE_CRITERIA_JSON}}

RECRUITER EXTRA PARAMETERS (accumulated across all iterations):
{{EXTRA_PARAMS_HISTORY}}

LAST PREVIEW RESULTS (for context — shows what the current criteria produced):
{{PREVIEW_RESULTS_JSON}}

Synthesize all of the above into a single unified rubric. Return this exact JSON:

{
  "final_evaluation_prompt": "",
  "required_resume_fields": [],
  "scoring_rubric": {
    "baseline_checks": [
      {
        "check": "",
        "resume_field": "",
        "reject_if_missing": true
      }
    ],
    "p0_weights": [
      {
        "signal": "",
        "weight": 0,
        "resume_field": ""
      }
    ],
    "red_flag_checks": [
      {
        "check": "",
        "resume_field": "",
        "deprioritize_if_present": true
      }
    ]
  },
  "screening_summary": "",
  "synthesis_notes": ""
}

Rules:
1. final_evaluation_prompt: comprehensive LLM instruction combining ALL
   base criteria AND all recruiter extra parameters.
2. required_resume_fields: ONLY fields actually needed from: name, education,
   work_experience, skills, certifications, projects, publications, github_url,
   total_experience_years, career_gaps_months.
3. p0_weights must sum to 100.
4. Each "include" extra parameter should become either a new baseline_check
   or an increase in p0_weight for a relevant signal.
5. Each "exclude" extra parameter should become a new red_flag_check or a
   tighter baseline_check.
6. screening_summary: 2 sentences plain English for a non-technical recruiter.
7. synthesis_notes: 2-3 sentences explaining what changed from the base
   criteria and why (based on the recruiter parameters added).
""".strip()
