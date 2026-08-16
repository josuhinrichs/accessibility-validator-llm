from pathlib import Path

import pandas as pd

from config import CSV_PATH, logger
from evaluation import (
    COMPARISON_RESULTS_PATH,
    MODELS_RESULTS_DIR,
    RESULTS_ROOT_DIR,
    RUN_ID,
    RUN_RESULTS_DIR,
    run_evaluation,
    sanitize_model_name,
)
from llm_clients import LLMClient, ollama_client, openai_client
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

def process_dataset(client: LLMClient, model: str, strategies: list[str],want_screenshot: bool =True, want_ax_tree: bool =True):
    """
    Lê o dataset, prepara o payload de inferência e aciona o runner.
    Itera linha a linha para manter footprint de memória baixo.
    """
    if not Path(CSV_PATH).exists():
        logger.error("dataset_not_found", path=str(CSV_PATH))
        return

    logger.info("dataset_ingestion_started", path=str(CSV_PATH))
    
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        logger.error("dataset_load_failed", error=str(e))
        return

    total_rows = len(df)
    processed_count = 0

    
    with PageRenderer(want_screenshot=want_screenshot, want_ax_tree=want_ax_tree) as renderer:
        for index, row in df.iterrows():
            item_id = str(row['id'])
            web_url_id = str(row['web_URL_id'])
            wcag_raw = str(row.get('wcag_reference', '')) or ""
            html_path = Path(str(row.get('html_file_path', '')))
            
            ground_truth_codes = extract_wcag_codes(wcag_raw)
    
            rendered = renderer.render(web_url_id=web_url_id, html_path=html_path, out_dir=MODELS_RESULTS_DIR)
            
            if not ground_truth_codes:
                logger.debug("skipping_row_no_ground_truth", item_id=item_id)
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
    
            image_paths = (
                [rendered.screenshot_path]
                if rendered.screenshot_path
                else []
            )
    
            logger.info(
                "item_ready_for_inference", 
                item_id=item_id, 
                ground_truth=list(ground_truth_codes),
                image_count=len(image_paths),
                progress=f"{processed_count + 1}/{total_rows}"
            )

            run_evaluation(
                client=client,
                model=model,
                item_id=item_id,
                ground_truth=ground_truth_codes,
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
    
    MODELS_TO_TEST = [
        "qwen2.5vl",
    ]

    #STRATEGIES_TO_TEST = ["zero-shot", "few-shot", "chain-of-thought"]

    STRATEGIES_TO_TEST = ["zero-shot"]

    for model in MODELS_TO_TEST:
        logger.info("model_run_started", model=model) 
        process_dataset(
            client=ollama_client,
            model=model,
            strategies=STRATEGIES_TO_TEST,
            want_screenshot=True,
            want_ax_tree=False
        )

        model_results_csv = MODELS_RESULTS_DIR / sanitize_model_name(model) / "metrics_output.csv"
        calculate_metrics(
            results_csv_path=model_results_csv,
            output_csv_path=model_results_csv.parent / "final_metrics.csv",
        )
        logger.info("model_run_completed", model=model, results_dir=str(model_results_csv.parent))

    build_comparison_summary(RUN_RESULTS_DIR, COMPARISON_RESULTS_PATH)
    logger.info("comparison_written", path=str(COMPARISON_RESULTS_PATH))
