import os
import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from config import CSV_PATH, logger
from evaluation import (
    COMPARISON_RESULTS_PATH,
    MODELS_RESULTS_DIR,
    RUN_RESULTS_DIR,
    export_diagnostics_report,
    log_results_quality_summary,
    run_evaluation,
    sanitize_model_name,
)
from llm_clients import LLMClient, lm_studio_client, ollama_client
from procecss import (
    calculate_metrics,
    extract_wcag_codes,
)
from prompt_builder import PromptRecipe
from render import PageRenderer

CSV_HEADERS = [
    "item_id", "model", "strategy", "duration_ms",
    "tp", "fp", "fn", "precision", "recall", "f1_score",
    "ground_truth", "predictions", "raw_output", "error"
]

def _url_to_possible_html_names(url: str) -> list[str]:
    """
    Gera possíveis nomes de arquivo local a partir da URL original.
    Ex.: https://www.aliexpress.com/ -> www_aliexpress_com_home.html
    """
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

    # Alguns nomes no dataset preservam '-', outros convertem para '_'.
    normalized_no_hyphen = normalized.replace("-", "_")

    candidates = [
        f"{normalized}.html",
        f"{normalized_no_hyphen}.html",
    ]

    # dedupe mantendo ordem
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)

    return deduped


def process_dataset(client: LLMClient, model: str, strategies: list[str], want_screenshot: bool =True, want_ax_tree: bool =True):
    """
    Lê o dataset CSV para obter ground truth e mapeia para arquivos HTML locais.
    """
    HTML_DIR = Path("./content/workspace/FullPipeline/html_pages_async")

    dataset_csv_path = Path(os.getenv("DATASET_CSV_PATH", CSV_PATH))

    if not dataset_csv_path.exists():
        logger.error("dataset_not_found", path=str(dataset_csv_path))
        return

    logger.info("dataset_ingestion_started", path=str(dataset_csv_path))

    try:
        df = pd.read_csv(dataset_csv_path)
    except Exception as e:
        logger.error("dataset_load_failed", error=str(e))
        return

    processed_count = 0

    # Group by the unique HTML identifier
    grouped = df.groupby('html_file_path')

    available_html_by_lower_name = {
        p.name.lower(): p
        for p in HTML_DIR.glob("*.html")
    }

    with PageRenderer(want_screenshot=want_screenshot, want_ax_tree=want_ax_tree) as renderer:
        for html_filename, group in grouped:
            # Aggregate all WCAG violations for this specific HTML file
            all_ground_truth = set()
            for _, row in group.iterrows():
                wcag_raw = str(row.get('wcag_reference', '')) or ""
                all_ground_truth.update(extract_wcag_codes(wcag_raw))

            # Resolve o arquivo HTML local com múltiplas estratégias.
            # O CSV contém mistura de nomes antigos (.txt) e nomes normalizados (.html).
            first_row = group.iloc[0]

            csv_name = Path(str(html_filename)).name
            csv_stem = Path(csv_name).stem
            html_file_name = Path(str(first_row.get("html_file_name", "") or "")).name
            web_url = str(first_row.get("web_URL", "") or "")

            candidate_names: list[str] = [
                csv_name,
                f"{csv_stem}.html",
            ]

            if html_file_name:
                candidate_names.append(html_file_name)
                candidate_names.append(f"{Path(html_file_name).stem}.html")

            candidate_names.extend(_url_to_possible_html_names(web_url))

            # Remove duplicados preservando ordem
            seen: set[str] = set()
            deduped_candidate_names: list[str] = []
            for name in candidate_names:
                name_clean = str(name).strip()
                if not name_clean:
                    continue
                key = name_clean.lower()
                if key in seen:
                    continue
                seen.add(key)
                deduped_candidate_names.append(name_clean)

            html_path: Path | None = None
            for candidate_name in deduped_candidate_names:
                resolved = available_html_by_lower_name.get(candidate_name.lower())
                if resolved is not None:
                    html_path = resolved
                    break

            if html_path is None:
                logger.warning(
                    "html_file_not_found",
                    csv_name=csv_name,
                    html_file_name=html_file_name,
                    web_url=web_url,
                    tried=deduped_candidate_names,
                )
                continue

            # Use the first ID or a combined ID for the item_id
            item_id = str(group.iloc[0]['id'])
            web_url_id = str(group.iloc[0]['web_URL_id'])

            # Render once per file
            rendered = renderer.render(web_url_id=web_url_id, html_path=html_path, out_dir=MODELS_RESULTS_DIR)

            if not all_ground_truth:
                logger.debug("skipping_file_no_ground_truth", item_id=item_id)
                continue

            evidence_inputs = ["html"]
            if want_ax_tree:
                evidence_inputs.append("axtree")
            if want_screenshot:
                evidence_inputs.append("screenshot")

            prompt = PromptRecipe.build_user_prompt(
                page=rendered,
                evidence_inputs=evidence_inputs
            )

            image_paths = [rendered.screenshot_path] if rendered.screenshot_path else []

            logger.info(
                "item_ready_for_inference",
                item_id=item_id,
                ground_truth=list(all_ground_truth),
                image_count=len(image_paths)
            )

            run_evaluation(
                client=client,
                model=model,
                item_id=item_id,
                ground_truth=all_ground_truth,
                text_prompt=prompt,
                images_paths=image_paths,
                strategies=strategies,
            )

            processed_count += 1

    logger.info("dataset_ingestion_completed", total_processed=processed_count)

def build_comparison_summary(results_root: Path, output_path: Path):
    """
    Consolida os `final_metrics.csv` de cada modelo em um único arquivo de comparação.
    """
    summary_frames = []

    for final_metrics_path in results_root.glob("by_model/*/final_metrics.csv"):
        try:
            model_summary = pd.read_csv(final_metrics_path)
            model_summary["source_file"] = str(final_metrics_path)
            summary_frames.append(model_summary)
        except Exception as e:
            logger.warning("failed_to_read_model_summary", path=str(final_metrics_path), error=str(e))

    if not summary_frames:
        logger.warning("no_model_summaries_found", path=str(results_root))
        return

    comparison_df = pd.concat(summary_frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path, index=False)



if __name__ == "__main__":
    logger.info("Iniciando o experimento...\n")

    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if provider in {"ollama", "local"}:
        selected_client = ollama_client
        selected_provider = "ollama"
    elif provider in {"lmstudio", "lm-studio", "openai_compat"}:
        selected_client = lm_studio_client
        selected_provider = "lmstudio"
    else:
        logger.warning("unknown_provider_fallback", provider=provider, fallback="ollama")
        selected_client = ollama_client
        selected_provider = "ollama"

    logger.info(
        "llm_provider_selected",
        provider=selected_provider,
        base_url=selected_client.base_url,
        force_json=selected_client.force_json,
        num_ctx=selected_client.num_ctx,
        include_images=selected_client.include_images,
    )

    MODELS_TO_TEST = [
        "gemma-3-27b-it",
        "llama-4-scout-17b-16e-instruct",
        "gemma-3-12b-it",
        "deepseek-r1-distill-llama-70b",
    ]

    #STRATEGIES_TO_TEST = ["zero-shot", "few-shot", "chain-of-thought"]

    STRATEGIES_TO_TEST = ["zero-shot"]

    for model in MODELS_TO_TEST:
        logger.info("model_run_started", model=model)
        process_dataset(
            client=selected_client,
            model=model,
            strategies=STRATEGIES_TO_TEST,
            want_screenshot=True,
            want_ax_tree=False
        )

        model_results_csv = MODELS_RESULTS_DIR / sanitize_model_name(model) / "metrics_output.csv"

        log_results_quality_summary(
            results_csv_path=model_results_csv,
            model=model,
        )

        export_diagnostics_report(
            results_csv_path=model_results_csv,
            output_csv_path=model_results_csv.parent / "diagnostics.csv",
            model=model,
        )

        calculate_metrics(
            results_csv_path=model_results_csv,
            output_csv_path=model_results_csv.parent / "final_metrics.csv",
        )
        logger.info("model_run_completed", model=model, results_dir=str(model_results_csv.parent))

    build_comparison_summary(RUN_RESULTS_DIR, COMPARISON_RESULTS_PATH)
    logger.info("comparison_written", path=str(COMPARISON_RESULTS_PATH))
