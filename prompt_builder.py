from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json

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


# OUTPUT_SCHEMA_HINT = """
# Return only a strict JSON object of this exact shape:
# {"violations": [
#   {"wcagCode": "<JUST the WCAG code(X.X.X)>", "violationName": "<taxonomy name>", "category": "semantic|layout",
#    "html": "<offending element HTML>", "target": "<selector or null>",
#    "description": "<why, citing what you observed>",
#    "impact": "critical|serious|moderate|minor", "confidence": <0.0-1.0>}
# ]}
# Make sure to return a valid WCAG violation. If none, return {"violations": []}.
# """


OUTPUT_SCHEMA_HINT = """
STRICT JSON OUTPUT RULES:
1. Return ONLY the JSON object. 
2. NO markdown, NO narrative, NO "Here is the audit".
3. Use ONLY WCAG code format (e.g., "1.3.1").
4. If no violations, return exactly: {"violations": []}
"""

FEW_SHOT = """
EXAMPLE OF EXPECTED OUTPUT:
{"violations": [
  {"wcagCode": "1.3.1", "violationName": "Missing Label", "category": "semantic",
   "html": "<input type='text' />", "target": "input",
   "description": "Form field lacks associated label.",
   "impact": "serious", "confidence": 0.9}
]}
"""


# OUTPUT_SCHEMA_HINT = """
# Return only a strict JSON object of this exact shape:
# {"violations": [
#   {"wcagCode": "<JUST the WCAG code>", "violationName": "<taxonomy name>", "category": "semantic|layout",
#    "html": "<offending element HTML>", "target": "<selector or null>",
#    "description": "<why, citing what you observed>",
#    "impact": "critical|serious|moderate|minor", "confidence": <0.0-1.0>}
# ]}
# Make sure to return a valid WCAG violation. Report only names in the taxonomy. If none, return {"violations": []}.
# """

MAX_AXTREE_CHARS = 120000
MAX_HTML_CHARS = 40000

#section_order = ("taxonomy", "schema", "task", "evidence")
section_order = ("schema", "few_shot", "task", "evidence")

from bs4 import BeautifulSoup


def sanitize_html(html_content: str) -> str:
    """
    Strips out tags that are irrelevant for accessibility analysis 
    to save context window space.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove tags that don't impact accessibility logic
    for tag in soup(["script", "style"]):
        tag.decompose()
        
    # Optional: If you find specific IDs or classes that are pure noise (e.g., tracking pixels),
    # remove them here.
    
    return str(soup)


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
        html = sanitize_html(page.html)

        if len(html) > MAX_HTML_CHARS:
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
            ""
        ]

        # parts: list[str] = [
        #     "Audit this page for accessibility violations from this taxonomy:"
        # ]

        evidence_inputs = {
            str(item).strip().lower()
            for item in evidence_inputs
        }
        
        for section in section_order:
            if section == "taxonomy":
                parts.extend(["", taxonomy_block])
            
            elif section == "schema":
                parts.extend(["", SYSTEM_PROMPT_BASELINE])
                parts.extend(["", OUTPUT_SCHEMA_HINT])

            elif section == "few_shot":
                parts.extend(["", FEW_SHOT])

            elif section == "task":
                parts.extend([
                    "",
                    "Audit this page for accessibility violations:",
                ])
            elif section == "evidence":
                parts.extend(
                    evidence_sections(page, evidence_inputs)
                )

        return "\n".join(parts)