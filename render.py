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

from playwright.sync_api import (
    Browser,
    BrowserContext,
    CDPSession,
    Page,
    Playwright,
    sync_playwright,
)
from typing_extensions import final

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_dataset_path(raw_path: str | Path) -> Path:
    # Maps dataset paths from their sandbox location to the local repo.
    # Ensures reproducibility by using local files instead of external URLs.
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


# Stores the rendered page data: HTML, screenshot path, and accessibility tree.
@dataclass
class RenderedPage:
    web_url_id: str
    html: str
    screenshot_path: str | None = None
    ax_tree: dict | None = None

# Renders HTML content using Playwright to capture screenshots and accessibility trees.
@final
class PageRenderer:
    # Playwright instances (optional, initialized on start)
    _playwright: Playwright | None
    _browser: Browser | None
    _context: BrowserContext | None
    _page: Page | None
    _cdp: CDPSession | None

    # Initializes the renderer with configuration options.
    def __init__(self, viewport: tuple[int, int] = (1280, 800), want_screenshot: bool = True,
                 want_ax_tree: bool = True, base_url: str | None = None,
                 timeout_ms: int = 30_000, retries: int = 3):

        # Viewport configuration for the browser.
        self.viewport = {
            "width": viewport[0],
            "height": viewport[1],
        }

        # Flags to enable/disable screenshot and accessibility tree generation.
        self.want_screenshot = want_screenshot
        self.want_ax_tree = want_ax_tree
        # Base URL for resolving relative assets in HTML.
        self.base_url = base_url
        self.timeout_ms = timeout_ms
        self.retries = retries
        
        # Initialize internal Playwright objects to None.
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._cdp = None

    # Starts Playwright and initializes browser components.
    def start(self):
        logger.info("starting playwright")
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch()
        self._context = self._browser.new_context(viewport=self.viewport,)

        self._page = self._context.new_page()
        self._page.set_default_timeout(self.timeout_ms)
        self._page.set_default_navigation_timeout(self.timeout_ms)

        # Enable CDP session if accessibility tree is requested.
        if self.want_ax_tree:
            self._cdp = self._context.new_cdp_session(self._page)
            self._cdp.send("Accessibility.enable")

        return self

    # Closes all Playwright resources.
    def close(self):
        # Detach CDP session if it exists.
        if self._cdp:
            try:
                self._cdp.detach()
            except Exception:
                # Suppress errors during detach, as cleanup is best-effort.
                pass

        # Close browser context, browser, and stop Playwright.
        if self._context:
            self._context.close()

        if self._browser:
            self._browser.close()

        if self._playwright:
            self._playwright.stop()

        # Reset attributes to None after closing.
        self._cdp = None
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None

    # Enables the use of PageRenderer as a context manager.
    def __enter__(self):
        return self.start()

    # Ensures resources are closed when exiting the context manager.
    def __exit__(self, exc_type, exc, tb):
        self.close()

    # Renders HTML content and captures output. Handles retries on failure.
    def render(
        self,
        html_path: Path,
        out_dir: Path,
        web_url_id: str,
    ) -> RenderedPage:

        # Ensure Playwright has been started.
        if self._page is None:
            raise RuntimeError(
                "PageRenderer must be started before render(). "+
                "Use 'with PageRenderer(...) as renderer:'"
            )

        # Resolve the HTML path and create the output directory.
        html_path = resolve_dataset_path(html_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Read HTML content.
        html = html_path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        screenshot_path = None
        ax_tree = None

        # Attempt to render the page, with retries.
        for attempt in range(1, self.retries + 1):
            try:
                # Set the page content and wait for DOM to load.
                self._page.set_content(
                    html,
                    wait_until="domcontentloaded",
                )

                # Capture screenshot if requested.
                if self.want_screenshot:
                    screenshot_path = (
                        out_dir / f"{web_url_id}.png"
                    )
                    self._page.screenshot(
                        path=str(screenshot_path)
                    )

                # Capture accessibility tree if requested.
                if self.want_ax_tree:
                    ax_tree = self._cdp.send(
                        "Accessibility.getFullAXTree"
                    )

                # Return the rendered page data on success.
                return RenderedPage(
                    web_url_id=web_url_id,
                    html=html,
                    screenshot_path=str(screenshot_path)
                    if screenshot_path
                    else None,
                    ax_tree=ax_tree,
                )

            except Exception as exc:
                # If max retries reached, re-raise the exception.
                if attempt >= self.retries:
                    raise

                # Calculate delay before retrying.
                delay = float(attempt)

                # Log the failure and retry attempt.
                logger.warning(
                    "render failed for %s (%s/%s): %s; " +
                    "retrying in %.1fs",
                    web_url_id,
                    attempt,
                    self.retries,
                    exc,
                    delay,
                )

                # Wait before the next retry.
                time.sleep(delay)

        # Should be unreachable if retries are handled correctly.
        raise RuntimeError("Unreachable")

    # def render(self, html_path: Path, out_dir: Path, web_url_id: str) -> RenderedPage:
    #     html_path, out_dir = resolve_dataset_path(html_path), Path(out_dir)
    #     out_dir.mkdir(parents=True, exist_ok=True)
    #     html = html_path.read_text(encoding="utf-8", errors="replace")

    #     screenshot_path = None
    #     ax_tree = None

    #     def _run():
    #         nonlocal screenshot_path, ax_tree
    #         with sync_playwright() as p:
    #             browser = p.chromium.launch()
    #             try:
    #                 page = browser.new_page(viewport=self.viewport)
    #                 page.set_default_timeout(self.timeout_ms)
    #                 page.set_default_navigation_timeout(self.timeout_ms)
    #                 page.set_content(html, wait_until="domcontentloaded")

    #                 if self.want_screenshot:
    #                     screenshot_path = out_dir / f"{web_url_id}.png"
    #                     page.screenshot(path=str(screenshot_path))

    #                 if self.want_ax_tree:
    #                     cdp = page.context.new_cdp_session(page)
    #                     cdp.send("Accessibility.enable")
    #                     ax_tree = cdp.send("Accessibility.getFullAXTree")
    #                     cdp.detach()
    #             finally:
    #                 browser.close()

    #     last_exc: Exception | None = None
    #     for attempt in range(1, self.retries + 1):
    #         try:
    #             _run()
    #             break
    #         except Exception as exc:
    #             last_exc = exc
    #             if attempt >= self.retries:
    #                 raise
    #             delay = float(attempt)
    #             logger.warning("render failed for %s (%s/%s): %s; retrying in %.1fs",
    #                            web_url_id, attempt, self.retries, exc, delay)
    #             time.sleep(delay)
    #     if last_exc and screenshot_path is None and ax_tree is None:
    #         logger.debug("render for %s ended after retries", web_url_id)

    #     return RenderedPage(web_url_id=web_url_id, html=html,
    #                        screenshot_path=screenshot_path, ax_tree=ax_tree)
