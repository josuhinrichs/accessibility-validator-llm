from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json
import os
import re

from render import RenderedPage


SYSTEM_PROMPT_BASELINE = (
    "You are a web accessibility auditor. You identify ONLY the semantic and "
    "layout WCAG accessibility violations considering meaning, context, or the "
    "match between what a sighted user sees and what assistive technology "
    "exposes."
)


SYSTEM_PROMPT_EVIDENCE_FIRST = (
    "You are a strict web accessibility auditor. Stay grounded in the provided "
    "evidence and return only violations supported by the page artifacts.\n\n"
    "Do not guess. Do not invent elements. Return STRICT JSON only."
)


OUTPUT_SCHEMA_HINT = """
Return a strict JSON object of this exact shape:
{"violations": [
  {"wcagCode": "<JUST the WCAG code>", "violationName": "<taxonomy name>", "category": "semantic|layout",
   "html": "<offending element HTML>", "target": "<selector or null>",
   "description": "<why, citing what you observed>",
   "impact": "critical|serious|moderate|minor", "confidence": <0.0-1.0>}
]}
Report only names in the taxonomy. If none, return {"violations": []}.
"""


MAX_AXTREE_CHARS = 120000
MAX_HTML_CHARS = 40000

SMART_HTML_TRUNCATION_ENV = "SMART_HTML_TRUNCATION"
SMART_SNIPPET_CONTEXT_CHARS = 220
SMART_MAX_SNIPPETS = 120

A11Y_TAGS = (
    "img", "a", "button", "input", "select", "textarea", "label", "form",
    "iframe", "video", "audio", "table", "th", "td", "canvas", "object", "area",
)

A11Y_TAG_PATTERN = re.compile(
    rf"<(?:{'|'.join(A11Y_TAGS)})\b[^>]*>",
    flags=re.IGNORECASE,
)

A11Y_ATTR_TAG_PATTERN = re.compile(
    r"<[a-zA-Z][a-zA-Z0-9:-]*\b[^>]*(?:\srole\s*=|\saria-[a-zA-Z0-9:_-]+\s*=|\salt\s*=|\sfor\s*=|\sid\s*=|\stabindex\s*=|\slang\s*=|\stitle\s=)[^>]*>",
    flags=re.IGNORECASE,
)

section_order = ("taxonomy", "schema", "task", "evidence")


def _is_env_true(env_var: str) -> bool:
    raw = (os.getenv(env_var, "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _strip_low_value_html_content(html: str) -> str:
    cleaned = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
    cleaned = re.sub(r"<script\b[^>]*>.*?</script>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<style\b[^>]*>.*?</style>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)

    # Remove atributos muito longos e pouco informativos para reduzir ruído/token.
    cleaned = re.sub(
        r"\s(?:style|data-[a-zA-Z0-9:_-]+)\s*=\s*(?:\"[^\"]{200,}\"|'[^']{200,}'|[^\s>]{200,})",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    return cleaned


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []

    ordered = sorted(ranges, key=lambda x: x[0])
    merged: list[tuple[int, int]] = [ordered[0]]

    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 40:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def _smart_truncate_html(html: str, max_chars: int) -> str:
    cleaned = _strip_low_value_html_content(html)

    if len(cleaned) <= max_chars:
        return cleaned

    ranges: list[tuple[int, int]] = []

    html_open_match = re.search(r"<html\b[^>]*>", cleaned, flags=re.IGNORECASE)
    if html_open_match:
        ranges.append((html_open_match.start(), min(len(cleaned), html_open_match.end() + 200)))

    title_match = re.search(r"<title\b[^>]*>.*?</title>", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if title_match:
        ranges.append((title_match.start(), title_match.end()))

    collected = 0
    for pattern in (A11Y_TAG_PATTERN, A11Y_ATTR_TAG_PATTERN):
        for match in pattern.finditer(cleaned):
            start = max(0, match.start() - SMART_SNIPPET_CONTEXT_CHARS)
            end = min(len(cleaned), match.end() + SMART_SNIPPET_CONTEXT_CHARS)
            ranges.append((start, end))
            collected += 1
            if collected >= SMART_MAX_SNIPPETS:
                break
        if collected >= SMART_MAX_SNIPPETS:
            break

    merged_ranges = _merge_ranges(ranges)

    snippets: list[str] = []
    for idx, (start, end) in enumerate(merged_ranges, start=1):
        snippet = cleaned[start:end].strip()
        if not snippet:
            continue
        snippets.append(f"<!-- SMART_SNIPPET_{idx} -->\n{snippet}")
        if sum(len(s) + 1 for s in snippets) >= max_chars:
            break

    smart_html = "\n".join(snippets).strip()

    if not smart_html:
        smart_html = cleaned[:max_chars]

    if len(smart_html) > max_chars:
        smart_html = smart_html[:max_chars]

    if len(smart_html) < len(cleaned):
        smart_html = smart_html + "\n[TRUNCATED]"

    return smart_html


def evidence_sections(
    page: RenderedPage,
    evidence_inputs: set[str],
) -> list[str]:
    parts: list[str] = []

    if "axtree" in evidence_inputs and page.ax_tree is not None:
        ax_tree_json = json.dumps(page.ax_tree)

        if len(ax_tree_json) > MAX_AXTREE_CHARS:
            ax_tree_json = ax_tree_json[:MAX_AXTREE_CHARS] + "\n[TRUNCATED]"

        parts.extend([
            "",
            "ACCESSIBILITY TREE (JSON):",
            ax_tree_json,
        ])

    if "html" in evidence_inputs and page.html:
        html = page.html

        if len(html) > MAX_HTML_CHARS:
            if _is_env_true(SMART_HTML_TRUNCATION_ENV):
                html = _smart_truncate_html(html, MAX_HTML_CHARS)
            else:
                html = html[:MAX_HTML_CHARS] + "\n[TRUNCATED]"

        parts.extend([
            "",
            "DOM (HTML):",
            html,
        ])

    if "screenshot" in evidence_inputs and page.screenshot_path:
        parts.extend([
            "",
            "A full-page screenshot is attached as an image.",
        ])

    return parts


@dataclass(frozen=True)
class PromptRecipe:
    name: str
    description: str
    system_prompt: str

    @staticmethod
    def build_user_prompt(
        page: RenderedPage,
        evidence_inputs: Iterable[str],
        taxonomy_block: str = "",
    ) -> str:
        parts: list[str] = [
            "Audit this page for accessibility violations from this taxonomy:"
        ]

        evidence_inputs = {
            str(item).strip().lower()
            for item in evidence_inputs
        }

        for section in section_order:
            if section == "taxonomy":
                parts.extend(["", taxonomy_block])

            elif section == "schema":
                parts.extend(["", OUTPUT_SCHEMA_HINT])

            elif section == "task":
                parts.extend([
                    "",
                    "Focus on violations directly supported by the inputs.",
                ])
            elif section == "evidence":
                parts.extend(
                    evidence_sections(page, evidence_inputs)
                )

        return "\n".join(parts)