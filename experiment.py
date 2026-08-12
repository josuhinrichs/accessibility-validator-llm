import csv
import base64
import re
from pathlib import Path
from typing import Dict, Set, List
import time

from llm_clients import LLMClient
from prompt_builder import PromptRecipe

import pandas as pd
from openai import OpenAI

from procecss import extract_wcag_codes, parse_supplementary_info, extract_predicted_wcag, calculate_metrics
from config import logger, CSV_PATH

def calculate_advanced_metrics(ground_truth: Set[str], predictions: Set[str]) -> Dict[str, float]:
    """
    Calcula TP, FP, FN e as métricas derivadas (Precision, Recall, F1).
    Inclui proteção contra divisão por zero.
    """
    tp = len(predictions.intersection(ground_truth))
    fp = len(predictions - ground_truth)
    fn = len(ground_truth - predictions)
    
    # Prevenção de divisão por zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4)
    }

def init_csv_file(filepath: Path):
    """
    Cria o arquivo CSV e escreve o cabeçalho caso ele ainda não exista.
    """
    headers = [
        "item_id", "model", "strategy", "duration_ms", 
        "tp", "fp", "fn", "precision", "recall", "f1_score", 
        "ground_truth", "predictions", "error"
    ]
    
    if not filepath.exists():
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

def append_to_csv(filepath: Path, record: Dict):
    """
    Adiciona uma única linha ao CSV. Abertura em modo 'a' (append) garante
    resiliência: se o script falhar, os dados processados até o momento estão salvos.
    """
    headers = [
        "item_id", "model", "strategy", "duration_ms", 
        "tp", "fp", "fn", "precision", "recall", "f1_score", 
        "ground_truth", "predictions", "error"
    ]
    
    with open(filepath, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writerow(record)

RESULTS_ROOT_DIR = Path("./experiment_results")
RUN_ID = time.strftime("%Y%m%d_%H%M%S")
RUN_RESULTS_DIR = RESULTS_ROOT_DIR / "runs" / RUN_ID
MODELS_RESULTS_DIR = RUN_RESULTS_DIR / "by_model"
COMPARISON_RESULTS_PATH = RUN_RESULTS_DIR / "model_comparison.csv"


def image_path_to_data_url(image_path: str) -> str:
    """
    Converte uma imagem local em uma data URL para envio ao OpenAI.
    """
    suffix = Path(image_path).suffix.lower()
    mime_type = "image/png" if suffix == ".png" else "image/jpeg"

    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"

def build_prompt_with_strategy(strategy: str, base_context: str) -> str:
    # (Mantém a mesma implementação fornecida anteriormente)
    base_instruction = "Liste todas as violações de acessibilidade (WCAG) encontradas. Retorne APENAS os códigos das diretrizes (ex: 1.1.1, 1.3.1, 1.4.6)."
    
    if strategy == "zero-shot":
        return f"{base_instruction}\n\n{base_context}"
    elif strategy == "few-shot":
        examples = "Exemplos de Saída:\n- <img src='logo.png'> -> 1.1.1\n- <div aria-hidden='true'>... -> 1.3.1\n"
        return f"{base_instruction}\n{examples}\n\n{base_context}"
    elif strategy == "chain-of-thought":
        cot_instruction = "Analise o contexto passo a passo. 1) Identifique os elementos estruturais e visuais. 2) Avalie o contraste e atributos ARIA. 3) Determine a regra WCAG violada. 4) Por fim, extraia apenas os códigos numéricos."
        return f"{cot_instruction}\n{base_instruction}\n\n{base_context}"
    
    return f"{base_instruction}\n\n{base_context}"


def run_evaluation(
    client: LLMClient,
    model: str,
    item_id: str,
    ground_truth: Set[str],
    text_prompt: str,
    images: List[str],
    strategies: List[str]
):
    """
    Executa a inferência, grava logs estruturados e persiste os resultados no CSV.
    """
    results_csv_path = MODELS_RESULTS_DIR / sanitize_model_name(model) / "metrics_output.csv"

    # Inicializa o CSV garantindo a presença do cabeçalho
    init_csv_file(results_csv_path)

    for strategy in strategies:
        final_prompt = build_prompt_with_strategy(strategy, text_prompt)
        start_time = time.perf_counter()

        logger.info("inference_started", item_id=item_id, model=model, strategy=strategy)

        # Estrutura base do registro para o CSV
        record = {
            "item_id": item_id,
            "model": model,
            "strategy": strategy,
            "duration_ms": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "ground_truth": "|".join(ground_truth),  # Salva como string delimitada para não quebrar o CSV
            "predictions": "",
            "error": "",
        }

        try:
            raw_output = client.run(
                model=model,
                prompt=final_prompt,
                images=images,
            )

            duration_ms = int((time.perf_counter() - start_time) * 1000)
            predicted_codes = extract_predicted_wcag(raw_output)
            metrics = calculate_advanced_metrics(ground_truth, predicted_codes)

            # Atualiza o registro com sucesso
            record.update({
                "duration_ms": duration_ms,
                "predictions": "|".join(predicted_codes),
                **metrics,
            })

            logger.info("inference_success", item_id=item_id, model=model, strategy=strategy, metrics=metrics)

        except Exception as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            error_msg = str(e)

            # Atualiza o registro refletindo a falha
            record.update({
                "duration_ms": duration_ms,
                "error": error_msg,
            })

            logger.error("inference_failed", item_id=item_id, model=model, strategy=strategy, error=error_msg)

        finally:
            # O bloco finally garante que a linha será salva no CSV, independentemente
            # de sucesso (try) ou falha de rede/OOM (except).
            append_to_csv(results_csv_path, record)


def process_dataset(client: LLMClient, model: str, strategies: List[str]):
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

    for index, row in df.iterrows():
        item_id = str(row['id'])
        wcag_raw = row.get('wcag_reference', '')
        affected_elements = row.get('affected_html_elements', '')
        supp_info_raw = row.get('supplementary_information', '')
        
        ground_truth_codes = extract_wcag_codes(wcag_raw)
        
        if not ground_truth_codes:
            logger.debug("skipping_row_no_ground_truth", item_id=item_id)
            continue
            
        image_paths, extra_text_context = parse_supplementary_info(supp_info_raw)
        
        prompt_payload = f"Abaixo estão os elementos HTML afetados por potenciais violações:\n"
        prompt_payload += f"```html\n{affected_elements}\n```\n"
        
        if extra_text_context:
            prompt_payload += f"\nContexto Suplementar:\n{extra_text_context}\n"

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
            text_prompt=prompt_payload,
            images=image_paths,
            strategies=strategies,
        )
        
        processed_count += 1

    logger.info("dataset_ingestion_completed", total_processed=processed_count)


def sanitize_model_name(model_name: str) -> str:
    """
    Converte o slug do modelo em um nome de pasta seguro para filesystem.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_")


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
)

ollama_client = LLMClient(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
    models=["qwen2.5vl"]
)

if __name__ == "__main__":
    logger.info("Iniciando o experimento...\n")
    
    MODELS_TO_TEST = [
        "qwen2.5vl",
    ]

    #STRATEGIES_TO_TEST = ["zero-shot", "few-shot", "chain-of-thought"]

    STRATEGIES_TO_TEST = ["zero-shot", "few-shot", "chain-of-thought"]

    for model in MODELS_TO_TEST:
        logger.info("model_run_started", model=model)
        process_dataset(
            client=ollama_client,
            model=model,
            strategies=STRATEGIES_TO_TEST,
        )

        model_results_csv = MODELS_RESULTS_DIR / sanitize_model_name(model) / "metrics_output.csv"
        calculate_metrics(
            results_csv_path=model_results_csv,
            output_csv_path=model_results_csv.parent / "final_metrics.csv",
        )
        logger.info("model_run_completed", model=model, results_dir=str(model_results_csv.parent))

    build_comparison_summary(RUN_RESULTS_DIR, COMPARISON_RESULTS_PATH)
    logger.info("comparison_written", path=str(COMPARISON_RESULTS_PATH))
