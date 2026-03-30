"""Gemini helpers for structured prompt execution and token tracking."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import google.generativeai as genai


def configure_genai(api_key: str | None = None) -> str:
    """Configure the Gemini SDK and return the resolved API key."""
    resolved_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not resolved_key:
        raise ValueError("Missing GOOGLE_API_KEY or GEMINI_API_KEY.")
    genai.configure(api_key=resolved_key)
    return resolved_key


def render_template(template: str, replacements: dict[str, Any]) -> str:
    rendered = template
    for key, value in replacements.items():
        if isinstance(value, (dict, list)):
            serialized = json.dumps(value, indent=2, ensure_ascii=True)
        else:
            serialized = str(value)
        rendered = rendered.replace(f"{{{{{key}}}}}", serialized)
    return rendered


def _response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return text.strip()

    chunks: list[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", []) if content else []
        for part in parts:
            maybe_text = getattr(part, "text", None)
            if maybe_text:
                chunks.append(maybe_text)
    return "\n".join(chunks).strip()


def extract_json_from_text(text: str) -> Any:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Model returned an empty response.")

    fenced_match = re.search(r"```(?:json)?\s*(.+?)\s*```", cleaned, flags=re.DOTALL)
    if fenced_match:
        cleaned = fenced_match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = min(
        [index for index in [cleaned.find("{"), cleaned.find("[")] if index != -1],
        default=-1,
    )
    end = max(cleaned.rfind("}"), cleaned.rfind("]"))
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Unable to locate JSON in the model response.")

    snippet = cleaned[start : end + 1]
    return json.loads(snippet)


def usage_to_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    total_tokens = int(getattr(usage, "total_token_count", input_tokens + output_tokens) or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def generate_response(
    model_name: str,
    system_instruction: str,
    user_prompt: str,
    temperature: float = 0.2,
) -> tuple[str, dict[str, int]]:
    started_at = time.perf_counter()
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_instruction,
        generation_config={
            "temperature": temperature,
            "response_mime_type": "application/json",
        },
    )
    response = model.generate_content(user_prompt)
    usage = usage_to_dict(response)
    usage["latency_seconds"] = round(time.perf_counter() - started_at, 2)
    return _response_text(response), usage


def run_structured_call(
    model_name: str,
    system_instruction: str,
    template: str,
    replacements: dict[str, Any],
    temperature: float = 0.2,
) -> tuple[Any, str, dict[str, int], str]:
    final_prompt = render_template(template, replacements)
    raw_text, usage = generate_response(model_name, system_instruction, final_prompt, temperature)
    parsed_json = extract_json_from_text(raw_text)
    return parsed_json, raw_text, usage, final_prompt


def run_raw_call(
    model_name: str,
    system_instruction: str,
    user_prompt: str,
    temperature: float = 0.2,
) -> tuple[str, dict[str, int], Any | None]:
    raw_text, usage = generate_response(model_name, system_instruction, user_prompt, temperature)
    parsed_json = None
    try:
        parsed_json = extract_json_from_text(raw_text)
    except Exception:
        parsed_json = None
    return raw_text, usage, parsed_json
