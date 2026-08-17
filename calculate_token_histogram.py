from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from config import CSV_PATH, logger
from prompt_builder import PromptRecipe, SYSTEM_PROMPT_EVIDENCE_FIRST
from render import RenderedPage


def _save_histogram_png(hist_df: pd.DataFrame, output_png_path: Path, title: str):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required to export PNG histogram. "
            "Install it with: pip install matplotlib"
        ) from exc

    labels = hist_df["token_bin"].astype(str).tolist()
    counts = hist_df["page_count"].astype(int).tolist()

    width = max(10, int(len(labels) * 0.6))
    fig, ax = plt.subplots(figsize=(width, 6))
    ax.bar(labels, counts)
    ax.set_title(title)
    ax.set_xlabel("Token range")
    ax.set_ylabel("Number of pages")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png_path, dpi=150)
    plt.close(fig)


def _url_to_possible_html_names(url: str) -> list[str]:
    if not url:
        return []

    parsed = urlparse(url.strip())
    host = (parsed.netloc or "").strip()
    path = (parsed.path or "").strip("/")

    if not host:
        return []

    host_part = host.replace(".", "_")
    base = f"{host_part}_{path}" if path else f"{host_part}_home"

    normalized = re.sub(r"[^A-Za-z0-9_\-/\.]+", "_", base)
    normalized = normalized.replace("/", "_").replace(".", "_")
    normalized = re.sub(r"_+", "_", normalized).strip("_")

    if not normalized:
        return []

    normalized_no_hyphen = normalized.replace("-", "_")

    candidates = [
        f"{normalized}.html",
        f"{normalized_no_hyphen}.html",
    ]

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)

    return deduped


def resolve_html_path(
    html_dir: Path,
    html_filename: str,
    html_file_name: str,
    web_url: str,
    available_html_by_lower_name: dict[str, Path],
) -> Path | None:
    csv_name = Path(str(html_filename)).name
    csv_stem = Path(csv_name).stem
    html_file_name_only = Path(str(html_file_name or "")).name

    candidate_names: list[str] = [
        csv_name,
        f"{csv_stem}.html",
    ]

    if html_file_name_only:
        candidate_names.append(html_file_name_only)
        candidate_names.append(f"{Path(html_file_name_only).stem}.html")

    candidate_names.extend(_url_to_possible_html_names(web_url))

    seen: set[str] = set()
    deduped_candidate_names: list[str] = []
    for name in candidate_names:
        clean = str(name).strip()
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped_candidate_names.append(clean)

    for candidate_name in deduped_candidate_names:
        resolved = available_html_by_lower_name.get(candidate_name.lower())
        if resolved is not None:
            return resolved

    return None


class TokenCounter:
    def __init__(self):
        self.method = "approx_chars_div_4"
        self._encoder = None

        try:
            import tiktoken  # type: ignore

            self._encoder = tiktoken.get_encoding("cl100k_base")
            self.method = "tiktoken_cl100k_base"
        except Exception:
            self._encoder = None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._encoder is not None:
            return len(self._encoder.encode(text))
        return max(1, round(len(text) / 4))


def calculate_token_histogram(
    dataset_csv_path: Path,
    html_dir: Path,
    output_dir: Path,
    want_screenshot: bool,
    want_ax_tree: bool,
    bin_size: int,
):
    if not dataset_csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_csv_path}")
    if not html_dir.exists():
        raise FileNotFoundError(f"HTML directory not found: {html_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_csv_path)
    grouped = df.groupby("html_file_path")

    available_html_by_lower_name = {
        p.name.lower(): p
        for p in html_dir.glob("*.html")
    }

    token_counter = TokenCounter()

    rows: list[dict[str, object]] = []
    skipped_missing_html = 0

    for html_filename, group in grouped:
        first_row = group.iloc[0]

        item_id = str(first_row.get("id", ""))
        web_url_id = str(first_row.get("web_URL_id", ""))
        html_file_name = str(first_row.get("html_file_name", "") or "")
        web_url = str(first_row.get("web_URL", "") or "")

        html_path = resolve_html_path(
            html_dir=html_dir,
            html_filename=str(html_filename),
            html_file_name=html_file_name,
            web_url=web_url,
            available_html_by_lower_name=available_html_by_lower_name,
        )

        if html_path is None:
            skipped_missing_html += 1
            continue

        html = html_path.read_text(encoding="utf-8", errors="replace")

        rendered_page = RenderedPage(
            web_url_id=web_url_id,
            html=html,
            screenshot_path="attached.png" if want_screenshot else None,
            ax_tree=None,  # for token estimation, only HTML + optional screenshot text is used
        )

        evidence_inputs = ["html"]
        if want_ax_tree:
            evidence_inputs.append("axtree")
        if want_screenshot:
            evidence_inputs.append("screenshot")

        user_prompt = PromptRecipe.build_user_prompt(
            page=rendered_page,
            evidence_inputs=evidence_inputs,
        )
        system_prompt = SYSTEM_PROMPT_EVIDENCE_FIRST

        user_tokens = token_counter.count(user_prompt)
        system_tokens = token_counter.count(system_prompt)
        total_input_tokens = user_tokens + system_tokens

        rows.append(
            {
                "item_id": item_id,
                "web_url_id": web_url_id,
                "html_file": html_path.name,
                "web_url": web_url,
                "user_prompt_chars": len(user_prompt),
                "system_prompt_chars": len(system_prompt),
                "user_prompt_tokens": user_tokens,
                "system_prompt_tokens": system_tokens,
                "total_input_tokens": total_input_tokens,
                "token_count_method": token_counter.method,
                "prompt_truncated_marker_present": "[TRUNCATED]" in user_prompt,
                "has_dom_section": "DOM (HTML):" in user_prompt,
                "has_schema_hint": "Return a strict JSON object" in user_prompt,
                "image_count": 1 if want_screenshot else 0,
            }
        )

    per_page_df = pd.DataFrame(rows)
    if per_page_df.empty:
        raise RuntimeError("No pages were resolved. Cannot build histogram.")

    per_page_csv = output_dir / "page_token_counts.csv"
    per_page_df.to_csv(per_page_csv, index=False)

    max_tokens = int(per_page_df["total_input_tokens"].max())
    upper = ((max_tokens // bin_size) + 1) * bin_size
    bin_edges = list(range(0, upper + bin_size, bin_size))

    if len(bin_edges) < 2:
        bin_edges = [0, bin_size]

    labels = [f"{bin_edges[i]}-{bin_edges[i+1]-1}" for i in range(len(bin_edges) - 1)]

    binned = pd.cut(
        per_page_df["total_input_tokens"],
        bins=bin_edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    hist_df = (
        binned.value_counts(sort=False)
        .rename_axis("token_bin")
        .reset_index(name="page_count")
    )

    hist_csv = output_dir / "token_histogram.csv"
    hist_df.to_csv(hist_csv, index=False)

    hist_png = output_dir / "token_histogram.png"
    _save_histogram_png(
        hist_df=hist_df,
        output_png_path=hist_png,
        title="Total input tokens per page",
    )

    summary_txt = output_dir / "token_histogram_summary.txt"
    summary_txt.write_text(
        "\n".join(
            [
                f"dataset_csv_path={dataset_csv_path}",
                f"html_dir={html_dir}",
                f"pages_processed={len(per_page_df)}",
                f"pages_skipped_missing_html={skipped_missing_html}",
                f"token_count_method={token_counter.method}",
                f"bin_size={bin_size}",
                f"min_total_tokens={int(per_page_df['total_input_tokens'].min())}",
                f"p50_total_tokens={int(per_page_df['total_input_tokens'].quantile(0.50))}",
                f"p90_total_tokens={int(per_page_df['total_input_tokens'].quantile(0.90))}",
                f"p95_total_tokens={int(per_page_df['total_input_tokens'].quantile(0.95))}",
                f"max_total_tokens={int(per_page_df['total_input_tokens'].max())}",
                f"per_page_csv={per_page_csv}",
                f"hist_csv={hist_csv}",
                f"hist_png={hist_png}",
            ]
        ),
        encoding="utf-8",
    )

    logger.info(
        "token_histogram_written",
        per_page_csv=str(per_page_csv),
        histogram_csv=str(hist_csv),
        histogram_png=str(hist_png),
        summary_txt=str(summary_txt),
        pages_processed=len(per_page_df),
        pages_skipped_missing_html=skipped_missing_html,
        token_count_method=token_counter.method,
    )


def main():
    parser = argparse.ArgumentParser(description="Calculate per-page prompt token histogram")
    parser.add_argument("--dataset-csv", default=CSV_PATH, help="Input dataset CSV path")
    parser.add_argument(
        "--html-dir",
        default="content/workspace/FullPipeline/html_pages_async",
        help="Directory containing local HTML files",
    )
    parser.add_argument(
        "--output-dir",
        default="experiment_results/token_analysis",
        help="Directory to write histogram outputs",
    )
    parser.add_argument("--bin-size", type=int, default=1000, help="Histogram bin size")
    parser.add_argument("--want-screenshot", action="store_true", help="Include screenshot note in prompt")
    parser.add_argument("--want-ax-tree", action="store_true", help="Include AX tree section (not rendered in this script)")

    args = parser.parse_args()

    calculate_token_histogram(
        dataset_csv_path=Path(args.dataset_csv),
        html_dir=Path(args.html_dir),
        output_dir=Path(args.output_dir),
        want_screenshot=bool(args.want_screenshot),
        want_ax_tree=bool(args.want_ax_tree),
        bin_size=int(args.bin_size),
    )


if __name__ == "__main__":
    main()
