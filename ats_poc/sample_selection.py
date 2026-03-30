"""Utilities for sample selection and token-efficient resume compression."""

from __future__ import annotations

import math
import re
from typing import Any


STOPWORDS = {
    "and",
    "the",
    "with",
    "for",
    "from",
    "that",
    "this",
    "have",
    "your",
    "will",
    "into",
    "role",
    "team",
    "years",
    "year",
    "candidate",
    "candidates",
    "experience",
    "required",
}


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9+#.\-]{1,}", text.lower())
        if token not in STOPWORDS
    ]


def extract_keywords(
    jd_text: str,
    jd_analysis: dict[str, Any] | None = None,
    final_config: dict[str, Any] | None = None,
) -> list[str]:
    keywords = set(tokenize(jd_text))
    if jd_analysis:
        baseline = jd_analysis.get("baseline", {})
        for skill in baseline.get("skills_required", []):
            keywords.update(tokenize(skill))
        keywords.update(tokenize(baseline.get("domain_required", "")))
        for signal in jd_analysis.get("p0_signals", []):
            keywords.update(tokenize(signal.get("signal", "")))
    if final_config:
        rubric = final_config.get("scoring_rubric", {})
        for check in rubric.get("baseline_checks", []):
            keywords.update(tokenize(check.get("check", "")))
        for weight in rubric.get("p0_weights", []):
            keywords.update(tokenize(weight.get("signal", "")))
    return sorted(keyword for keyword in keywords if len(keyword) > 2)[:120]


def flatten_resume(resume_json: dict[str, Any]) -> str:
    segments = [resume_json.get("name", "")]
    for education in resume_json.get("education", []):
        segments.extend([education.get("degree", ""), education.get("institution", "")])
    for job in resume_json.get("work_experience", []):
        segments.extend([job.get("company", ""), job.get("role", ""), job.get("description", "")])
    for field in ["skills", "certifications", "projects", "publications"]:
        segments.extend(resume_json.get(field, []))
    segments.append(resume_json.get("github_url", ""))
    return " ".join(segment for segment in segments if segment)


def score_resume_against_keywords(resume_json: dict[str, Any], keywords: list[str]) -> int:
    haystack = flatten_resume(resume_json).lower()
    matches = {keyword for keyword in keywords if keyword in haystack}
    return len(matches)


def compress_resume(resume_json: dict[str, Any], required_fields: list[str]) -> dict[str, Any]:
    compressed = {}
    for field in required_fields:
        value = resume_json.get(field)
        if field == "work_experience" and isinstance(value, list):
            compressed[field] = [
                {
                    "company": item.get("company", ""),
                    "role": item.get("role", ""),
                    "duration_months": item.get("duration_months", 0),
                    "type": item.get("type", ""),
                    "description": item.get("description", "")[:240],
                }
                for item in value[:5]
            ]
        elif field == "education" and isinstance(value, list):
            compressed[field] = value[:3]
        elif field in {"skills", "certifications", "projects", "publications"} and isinstance(value, list):
            compressed[field] = value[:12]
        else:
            compressed[field] = value
    return compressed


def pick_representative_sample(
    parsed_resumes: list[dict[str, Any]],
    keywords: list[str],
    sample_size: int = 15,
) -> list[dict[str, Any]]:
    scored = []
    for item in parsed_resumes:
        if not item.get("quality", {}).get("readable", False):
            continue
        score = score_resume_against_keywords(item["resume_json"], keywords)
        enriched = dict(item)
        enriched["keyword_score"] = score
        scored.append(enriched)

    scored.sort(key=lambda item: item["keyword_score"], reverse=True)
    if len(scored) <= sample_size:
        return scored

    top = scored[:5]

    middle_pool = scored[5:-5]
    middle = []
    if middle_pool:
        step = max(1, len(middle_pool) // 5)
        middle = [middle_pool[index] for index in range(0, len(middle_pool), step)[:5]]

    bottom = scored[-5:]

    selected = []
    seen = set()
    for item in top + middle + bottom:
        key = item.get("file_name")
        if key not in seen:
            selected.append(item)
            seen.add(key)

    if len(selected) < sample_size:
        for item in scored:
            key = item.get("file_name")
            if key in seen:
                continue
            selected.append(item)
            seen.add(key)
            if len(selected) >= sample_size:
                break

    return selected[:sample_size]
