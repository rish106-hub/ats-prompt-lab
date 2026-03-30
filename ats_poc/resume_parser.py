"""Local PDF extraction and rule-based resume parsing."""

from __future__ import annotations

import io
import math
import re
from collections import defaultdict
from datetime import datetime
from typing import Any

import pdfplumber


MONTH_MAP = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

SECTION_ALIASES = {
    "education": {"education", "academic background", "academics"},
    "work_experience": {
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "career history",
    },
    "skills": {"skills", "technical skills", "core skills", "core competencies"},
    "certifications": {"certifications", "licenses", "certificates"},
    "projects": {"projects", "selected projects", "personal projects"},
    "publications": {"publications", "research", "papers"},
}

GENERIC_CONTACT_WORDS = {
    "resume",
    "curriculum vitae",
    "linkedin",
    "github",
    "email",
    "phone",
    "mobile",
    "address",
}


def extract_text_from_pdf(file_bytes: bytes) -> str:
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages).strip()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).replace("\r", "").strip()


def split_lines(text: str) -> list[str]:
    lines = [normalize_whitespace(line) for line in text.splitlines()]
    return [line for line in lines if line]


def canonical_heading(line: str) -> str:
    return re.sub(r"[^a-z ]", "", line.lower()).strip()


def sectionize_resume(lines: list[str]) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = defaultdict(list)
    current_section = "header"
    heading_map = {
        alias: canonical
        for canonical, aliases in SECTION_ALIASES.items()
        for alias in aliases
    }

    for line in lines:
        normalized = canonical_heading(line)
        if normalized in heading_map:
            current_section = heading_map[normalized]
            continue
        sections[current_section].append(line)
    return dict(sections)


def parse_name(lines: list[str]) -> str:
    for line in lines[:6]:
        lower = line.lower()
        if any(word in lower for word in GENERIC_CONTACT_WORDS):
            continue
        if "@" in line or re.search(r"\+?\d[\d\s\-()]{7,}", line):
            continue
        if 1 <= len(line.split()) <= 5 and re.fullmatch(r"[A-Za-z .'-]+", line):
            return line.strip()
    return ""


def parse_github_url(text: str) -> str:
    match = re.search(r"https?://(?:www\.)?github\.com/[A-Za-z0-9_.-]+", text, flags=re.IGNORECASE)
    return match.group(0) if match else ""


def parse_skills(section_lines: list[str]) -> list[str]:
    if not section_lines:
        return []
    text = " | ".join(section_lines)
    parts = re.split(r"[,|/•;\n]", text)
    cleaned = []
    seen = set()
    for part in parts:
        item = normalize_whitespace(part).strip("- ").strip()
        if not item or len(item) > 40:
            continue
        key = item.lower()
        if key not in seen:
            cleaned.append(item)
            seen.add(key)
    return cleaned[:25]


def parse_simple_list(section_lines: list[str]) -> list[str]:
    items = []
    seen = set()
    for line in section_lines:
        item = normalize_whitespace(re.sub(r"^[•\-\*]\s*", "", line)).strip()
        if not item:
            continue
        key = item.lower()
        if key not in seen:
            items.append(item)
            seen.add(key)
    return items[:12]


def _parse_date_token(token: str) -> tuple[int, int] | None:
    token = token.strip().lower()
    if token in {"present", "current", "now"}:
        today = datetime.today()
        return today.year, today.month

    month_match = re.match(r"([A-Za-z]{3,9})\s+(\d{4})", token)
    if month_match:
        month_name = month_match.group(1)[:4].lower().rstrip(".")
        month_num = MONTH_MAP.get(month_name[:3]) or MONTH_MAP.get(month_name)
        if month_num:
            return int(month_match.group(2)), month_num

    year_match = re.match(r"(\d{4})", token)
    if year_match:
        return int(year_match.group(1)), 1

    return None


def parse_date_range(text: str) -> tuple[int, tuple[int, int], tuple[int, int]] | None:
    match = re.search(
        r"((?:[A-Za-z]{3,9}\s+)?\d{4})\s*(?:-|to|–|—)\s*((?:[A-Za-z]{3,9}\s+)?\d{4}|Present|Current|Now)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    start = _parse_date_token(match.group(1))
    end = _parse_date_token(match.group(2))
    if not start or not end:
        return None

    start_months = start[0] * 12 + start[1]
    end_months = end[0] * 12 + end[1]
    duration_months = max(0, end_months - start_months)
    return duration_months, start, end


def infer_company_type(text: str) -> str:
    lower = text.lower()
    if any(word in lower for word in {"consulting", "services", "agency", "outsourcing"}):
        return "service"
    if any(word in lower for word in {"freelance", "contractor", "independent"}):
        return "freelance"
    if any(word in lower for word in {"startup", "seed", "series a", "early stage"}):
        return "startup"
    return "product"


def parse_work_experience(section_lines: list[str]) -> tuple[list[dict[str, Any]], list[tuple[tuple[int, int], tuple[int, int]]]]:
    if not section_lines:
        return [], []

    blocks: list[list[str]] = []
    current: list[str] = []
    for line in section_lines:
        if current and re.fullmatch(r"[A-Z][A-Z /&-]{2,}", line):
            blocks.append(current)
            current = [line]
            continue
        current.append(line)
        if len(current) >= 3 and parse_date_range(" ".join(current[-2:])):
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)

    experience = []
    ranges = []
    for block in blocks:
        joined = " | ".join(block)
        date_info = parse_date_range(joined)
        duration_months = date_info[0] if date_info else 0
        if date_info:
            ranges.append((date_info[1], date_info[2]))

        role = ""
        company = ""
        for line in block[:3]:
            if " at " in line.lower():
                left, right = re.split(r"\bat\b", line, maxsplit=1, flags=re.IGNORECASE)
                role = left.strip(" |-")
                company = right.strip(" |-")
                break
            if "|" in line:
                left, right = [part.strip() for part in line.split("|", 1)]
                if not parse_date_range(line):
                    role = left
                    company = right
                    break

        if not role and block:
            role = block[0]
        if not company and len(block) > 1:
            company = block[1] if not parse_date_range(block[1]) else ""

        description_lines = [line for line in block if line not in {role, company}]
        description = " ".join(description_lines)[:500]
        experience.append(
            {
                "company": company,
                "role": role,
                "duration_months": duration_months,
                "type": infer_company_type(joined),
                "description": description,
            }
        )

    cleaned = [item for item in experience if any([item["company"], item["role"], item["description"]])]
    return cleaned[:12], ranges


def parse_education(section_lines: list[str]) -> list[dict[str, str]]:
    degrees = []
    degree_keywords = (
        "b.tech",
        "btech",
        "m.tech",
        "mtech",
        "b.e",
        "be ",
        "m.e",
        "mba",
        "b.sc",
        "m.sc",
        "bachelor",
        "master",
        "phd",
        "doctor",
    )

    for line in section_lines:
        lower = line.lower()
        if any(keyword in lower for keyword in degree_keywords):
            year_match = re.search(r"(19|20)\d{2}", line)
            institution_match = re.search(r"(?:at|from)\s+(.+)$", line, flags=re.IGNORECASE)
            degrees.append(
                {
                    "degree": line[:120],
                    "institution": institution_match.group(1).strip() if institution_match else "",
                    "year": year_match.group(0) if year_match else "",
                    "tier": "",
                }
            )
    return degrees[:6]


def infer_total_experience_years(text: str, work_experience: list[dict[str, Any]]) -> float:
    explicit_match = re.search(
        r"(\d+(?:\.\d+)?)\+?\s+years?\s+(?:of\s+)?experience",
        text,
        flags=re.IGNORECASE,
    )
    if explicit_match:
        return round(float(explicit_match.group(1)), 1)

    total_months = sum(item.get("duration_months", 0) for item in work_experience)
    return round(total_months / 12, 1) if total_months else 0.0


def infer_career_gaps_months(ranges: list[tuple[tuple[int, int], tuple[int, int]]]) -> list[int]:
    if len(ranges) < 2:
        return []

    normalized = sorted(ranges, key=lambda item: (item[0][0], item[0][1]), reverse=True)
    gaps = []
    for index in range(len(normalized) - 1):
        current_start = normalized[index][0]
        previous_end = normalized[index + 1][1]
        current_start_months = current_start[0] * 12 + current_start[1]
        previous_end_months = previous_end[0] * 12 + previous_end[1]
        gap = max(0, current_start_months - previous_end_months)
        if gap > 2:
            gaps.append(gap)
    return gaps[:5]


def assess_resume_quality(resume_json: dict[str, Any], raw_text: str) -> dict[str, Any]:
    populated_fields = 0
    tracked_fields = [
        resume_json.get("name"),
        resume_json.get("education"),
        resume_json.get("work_experience"),
        resume_json.get("skills"),
    ]
    for field in tracked_fields:
        if field:
            populated_fields += 1

    has_text = len(raw_text.split()) >= 80
    readable = populated_fields >= 2 and has_text
    score = min(100, populated_fields * 20 + min(40, len(raw_text.split()) // 20))
    reasons = []
    if not has_text:
        reasons.append("Very little extractable text in PDF.")
    if populated_fields < 2:
        reasons.append("Most structured resume fields are empty after parsing.")

    return {"readable": readable, "score": score, "reasons": reasons}


def parse_resume_text(text: str) -> dict[str, Any]:
    lines = split_lines(text)
    sections = sectionize_resume(lines)
    work_experience, date_ranges = parse_work_experience(sections.get("work_experience", []))
    resume_json = {
        "name": parse_name(lines),
        "education": parse_education(sections.get("education", [])),
        "work_experience": work_experience,
        "skills": parse_skills(sections.get("skills", [])),
        "certifications": parse_simple_list(sections.get("certifications", [])),
        "projects": parse_simple_list(sections.get("projects", [])),
        "publications": parse_simple_list(sections.get("publications", [])),
        "github_url": parse_github_url(text),
        "total_experience_years": infer_total_experience_years(text, work_experience),
        "career_gaps_months": infer_career_gaps_months(date_ranges),
    }
    return resume_json


def parse_resume_pdf(file_name: str, file_bytes: bytes) -> dict[str, Any]:
    raw_text = extract_text_from_pdf(file_bytes)
    resume_json = parse_resume_text(raw_text)
    quality = assess_resume_quality(resume_json, raw_text)
    return {
        "file_name": file_name,
        "raw_text": raw_text,
        "resume_json": resume_json,
        "quality": quality,
    }
