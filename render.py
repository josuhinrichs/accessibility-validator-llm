"""
render.py — turn a SAVED HTML file into the RQ1 input representations.

We render the dataset's saved HTML with Playwright via page.set_content(html)
instead of page.goto(url), so evaluation is reproducible and offline: the page
cannot drift. One session per page keeps screenshot and a11y tree consistent.

Playwright removed page.accessibility, so the a11y tree comes from CDP:
Accessibility.getFullAXTree — a flat list of nodes, closer to what AT consumes.

Layout violations (color-contrast) need pixels, so the screenshot is mandatory
for the Layout scope: it's the visual evidence the LLM reasons over, and Axe reads
the rendered colors from the same DOM state.

NOTE: set_content does not resolve relative asset URLs unless a base URL is
reachable. Most AccessGuru pages inline enough for contrast checks; document this
as a rendering caveat and pass base_url if the saved page needs it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from playwright.sync_api import sync_playwright

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_dataset_path(raw_path: str | Path) -> Path:
    """Map dataset paths from their original sandbox location into this repo.

    The CSV stores values such as /content/workspace/FullPipeline/... while the
    project is checked out locally under accessibility-validator-llm/content/... .
    We keep the CSV untouched and resolve the path at runtime.
    """
    raw_value = str(raw_path).strip()
    if not raw_value:
        raise FileNotFoundError("html_file_path is empty")

    candidate = Path(raw_value)
    if candidate.exists():
        return candidate

    candidates: list[Path] = []

    if candidate.is_absolute():
        candidates.append(PROJECT_ROOT.joinpath(*candidate.parts[1:]))
    else:
        candidates.append(PROJECT_ROOT / raw_value)
        candidates.append(PROJECT_ROOT / "content" / "workspace" / "FullPipeline" / candidate.name)
        candidates.append(PROJECT_ROOT / "content" / "workspace" / "FullPipeline" / raw_value.lstrip("./"))

    if raw_value.startswith("/content/workspace/"):
        mapped = PROJECT_ROOT.joinpath(*Path(raw_value).parts[1:])
        candidates.append(mapped)

    for path in candidates:
        if path.exists():
            return path

    fallback = PROJECT_ROOT / "content" / "workspace" / "FullPipeline" / Path(raw_value).name
    if fallback.exists():
        return fallback

    return candidate


@dataclass
class RenderedPage:
    web_url_id: str
    html: str
    screenshot_path: str | None = None
    ax_tree: dict | None = None


class PageRenderer:
    def __init__(self, viewport=(1280, 800), want_screenshot=True,
                 want_ax_tree=True, base_url: str | None = None,
                 timeout_ms: int = 30_000, retries: int = 3):
        self.viewport = {"width": viewport[0], "height": viewport[1]}
        self.want_screenshot = want_screenshot
        self.want_ax_tree = want_ax_tree
        self.base_url = base_url
        self.timeout_ms = timeout_ms
        self.retries = retries

    def render(self, html_path: Path, out_dir: Path, web_url_id: str) -> RenderedPage:
        html_path, out_dir = resolve_dataset_path(html_path), Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        html = html_path.read_text(encoding="utf-8", errors="replace")

        screenshot_path = None
        ax_tree = None

        def _run():
            nonlocal screenshot_path, ax_tree
            with sync_playwright() as p:
                browser = p.chromium.launch()
                try:
                    page = browser.new_page(viewport=self.viewport)
                    page.set_default_timeout(self.timeout_ms)
                    page.set_default_navigation_timeout(self.timeout_ms)
                    page.set_content(html, wait_until="domcontentloaded")

                    if self.want_screenshot:
                        screenshot_path = out_dir / f"{web_url_id}.png"
                        page.screenshot(path=str(screenshot_path))

                    if self.want_ax_tree:
                        cdp = page.context.new_cdp_session(page)
                        cdp.send("Accessibility.enable")
                        ax_tree = cdp.send("Accessibility.getFullAXTree")
                        cdp.detach()
                finally:
                    browser.close()

        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                _run()
                break
            except Exception as exc:
                last_exc = exc
                if attempt >= self.retries:
                    raise
                delay = float(attempt)
                logger.warning("render failed for %s (%s/%s): %s; retrying in %.1fs",
                               web_url_id, attempt, self.retries, exc, delay)
                time.sleep(delay)
        if last_exc and screenshot_path is None and ax_tree is None:
            logger.debug("render for %s ended after retries", web_url_id)

        return RenderedPage(web_url_id=web_url_id, html=html,
                           screenshot_path=screenshot_path, ax_tree=ax_tree)