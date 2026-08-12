from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterable

from render import PageRenderer, RenderedPage


SYSTEM_PROMPT_BASELINE = (
    "You are a web accessibility auditor. You identify ONLY the semantic and "
    "layout WCAG accessibility violations considering meaning, context, or the "
    "match between what a sighted user sees and what assistive technology "
    "exposes."
)


SYSTEM_PROMPT_EVIDENCE_FIRST = (
    "You are a strict web accessibility auditor. Stay grounded in the provided "
    "evidence and return only violations supported by the page artifacts.\n\n"
    "Do not summarize the page. Do not explain the HTML. Do not offer help. "
    "Return one valid JSON object only."
)


OUTPUT_SCHEMA_HINT = """
Return a strict JSON object of this exact shape:
{"violations": [
  {"wcagCode": "<JUST the WCAG code>", "violationName": "<short violation name>", "category": "semantic|layout",
   "html": "<offending element HTML>", "target": "<selector or null>",
   "description": "<why, citing what you observed>",
   "impact": "critical|serious|moderate|minor", "confidence": <0.0-1.0>}
]}
If none, return {"violations": []}.
"""


FINAL_JSON_REMINDER = """
Now return ONLY the JSON object. No markdown, no prose, no explanation.
If the evidence is insufficient for a supported violation, return {"violations": []}.
"""


section_order = ("taxonomy", "schema", "task", "evidence")


def evidence_sections(
    page: RenderedPage,
    evidence_inputs: set[str],
) -> list[str]:
    parts: list[str] = []

    if "axtree" in evidence_inputs and page.ax_tree is not None:
        ax_tree_json = json.dumps(page.ax_tree)

        if len(ax_tree_json) > 60000:
            ax_tree_json = ax_tree_json[:60000] + "\n[TRUNCATED]"

        parts.extend([
            "",
            "ACCESSIBILITY TREE (JSON):",
            ax_tree_json,
        ])

    if "html" in evidence_inputs and page.html:
        html = page.html

        if len(html) > 60000:
            html = html[:60000] + "\n[TRUNCATED]"

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
        extra_context: str = "",
    ) -> str:
        parts: list[str] = []

        evidence_inputs = {
            str(item).strip().lower()
            for item in evidence_inputs
        }

        for section in section_order:
            if section == "taxonomy":
                if taxonomy_block.strip():
                    parts.extend([
                        "Audit this page for accessibility violations from this taxonomy:",
                        "",
                        taxonomy_block,
                    ])
                else:
                    parts.append("Audit this page for WCAG accessibility violations.")

            elif section == "schema":
                parts.extend(["", OUTPUT_SCHEMA_HINT])

            elif section == "task":
                parts.extend([
                    "",
                    "Focus on violations directly supported by the inputs.",
                ])
                if extra_context.strip():
                    parts.extend([
                        "",
                        "ADDITIONAL EVIDENCE:",
                        extra_context.strip(),
                    ])

            elif section == "evidence":
                parts.extend(
                    evidence_sections(page, evidence_inputs)
                )

        parts.extend(["", FINAL_JSON_REMINDER])

        return "\n".join(parts)
