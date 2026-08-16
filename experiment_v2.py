import time
from pathlib import Path

# Removed redundant imports
# from typing import List, Set
import pandas as pd
from openai import OpenAI

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
from llm_clients import LLMClient
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




# def build_prompt_with_strategy(strategy: str, base_context: str) -> str:
#     # (Mantém a mesma implementação fornecida anteriormente)
#     base_instruction = "Liste todas as violações de acessibilidade (WCAG) encontradas. Retorne APENAS os códigos das diretrizes (ex: 1.1.1, 1.3.1, 1.4.6)."
    
#     if strategy == "zero-shot":
#         return f"{base_instruction}\n\n{base_context}"
#     elif strategy == "few-shot":
#         examples = "Exemplos de Saída:\n- <img src='logo.png'> -> 1.1.1\n- <div aria-hidden='true'>... -> 1.3.1\n"
#         return f"{base_instruction}\n{examples}\n\n{base_context}"
#     elif strategy == "chain-of-thought":
#         cot_instruction = "Analise o contexto passo a passo. 1) Identifique os elementos estruturais e visuais. 2) Avalie o contraste e atributos ARIA. 3) Determine a regra WCAG violada. 4) Por fim, extraia apenas os códigos numéricos."
#         return f"{cot_instruction}\n{base_instruction}\n\n{base_context}"
    
#     return f"{base_instruction}\n\n{base_context}"


# def run_evaluation(
#     client: LLMClient,
#     model: str,
#     item_id: str,
#     ground_truth: set[str],
#     text_prompt: str,
#     images_paths: list[str],
#     strategies: list[str]
# ):
#     """
#     Executa a inferência, grava logs estruturados e persiste os resultados no CSV.
#     """
#     results_csv_path = MODELS_RESULTS_DIR / sanitize_model_name(model) / "metrics_output.csv"

#     # Inicializa o CSV garantindo a presença do cabeçalho
#     init_csv_file(results_csv_path)

#     for strategy in strategies:
#         final_prompt = text_prompt
#         start_time = time.perf_counter()

#         logger.info("inference_started", item_id=item_id, model=model, strategy=strategy)

#         # Estrutura base do registro para o CSV
#         record = {
#             "item_id": item_id,
#             "model": model,
#             "strategy": strategy,
#             "duration_ms": 0,
#             "tp": 0,
#             "fp": 0,
#             "fn": 0,
#             "precision": 0.0,
#             "recall": 0.0,
#             "f1_score": 0.0,
#             "ground_truth": "|".join(ground_truth),  # Salva como string delimitada para não quebrar o CSV
#             "predictions": "",
#             "raw_output": "",
#             "error": "",
#         }

#         try:
#             raw_output = client.run(
#                 model=model,
#                 prompt=final_prompt,
#                 images=images_paths,
#             )

#             duration_ms = int((time.perf_counter() - start_time) * 1000)
#             predicted_codes = extract_predicted_wcag(raw_output)
#             metrics = calculate_advanced_metrics(ground_truth, predicted_codes)

#             # Atualiza o registro com sucesso
#             record.update({
#                 "duration_ms": duration_ms,
#                 "predictions": "|".join(predicted_codes),
#                 "raw_output": raw_output,
#                 **metrics,
#             })

#             logger.info("inference_success", item_id=item_id, model=model, strategy=strategy, metrics=metrics)

#         except Exception as e:
#             duration_ms = int((time.perf_counter() - start_time) * 1000)
#             error_msg = str(e)

#             # Atualiza o registro refletindo a falha
#             record.update({
#                 "duration_ms": duration_ms,
#                 "error": error_msg,
#             })

#             logger.error("inference_failed", item_id=item_id, model=model, strategy=strategy, error=error_msg)

#         finally:
#             # O bloco finally garante que a linha será salva no CSV, independentemente
#             # de sucesso (try) ou falha de rede/OOM (except).
#             append_to_csv(results_csv_path, record)


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
            wcag_raw = str(row.get('wcag_reference', ''))
            html_path = Path( str( row.get('html_file_path', '') ) )
            
            ground_truth_codes = extract_wcag_codes(wcag_raw)
    
            rendered = renderer.render(web_url_id=web_url_id, html_path=html_path, out_dir=MODELS_RESULTS_DIR)
            
            if not ground_truth_codes:
                logger.debug("skipping_row_no_ground_truth", item_id=item_id)
                continue
    
            # if want_ax_tree and rendered.ax_tree:
            #     prompt_payload = f"\nÁrvore de Acessibilidade (JSON):\n{json.dumps(rendered.ax_tree)[:60000]}\n"
    
            # if want_screenshot and rendered.screenshot_path:
            #     prompt_payload += f"\nUma captura de tela completa da página está anexada como uma imagem.\n"
    
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

openai_client = LLMClient(
    api_key="lm-studio",
    base_url="http://10.102.20.26:1234/v1",
    models=["qwen2.5vl"],
    include_images=True
)

ollama_client = LLMClient(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
    models=["qwen2.5vl"],
    include_images=True
)

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
            client=openai_client,
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
