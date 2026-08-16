import csv
import re
import time
from pathlib import Path

from config import logger
from llm_clients import LLMClient
from procecss import extract_predicted_wcag

CSV_HEADERS = [
    "item_id", "model", "strategy", "duration_ms",
    "tp", "fp", "fn", "precision", "recall", "f1_score",
    "ground_truth", "predictions", "raw_output", "error"
]
RESULTS_ROOT_DIR = Path("./experiment_results")
RUN_ID = time.strftime("%Y%m%d_%H%M%S")
RUN_RESULTS_DIR = RESULTS_ROOT_DIR / "runs" / RUN_ID
MODELS_RESULTS_DIR = RUN_RESULTS_DIR / "by_model"
COMPARISON_RESULTS_PATH = RUN_RESULTS_DIR / "model_comparison.csv"

def sanitize_model_name(model_name: str) -> str:
    """
    Converte o slug do modelo em um nome de pasta seguro para filesystem.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_")

def calculate_advanced_metrics(ground_truth: set[str], predictions: set[str]) -> dict[str, float]:
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
    if not filepath.exists():
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)

def append_to_csv(filepath: Path, record: dict[str, str | int | float]):
    """
    Adiciona uma única linha ao CSV. Abertura em modo 'a' (append) garante
    resiliência: se o script falhar, os dados processados até o momento estão salvos.
    """
    with open(filepath, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writerow(record)

def run_evaluation(
    client: LLMClient,
    model: str,
    item_id: str,
    ground_truth: set[str],
    text_prompt: str,
    images_paths: list[str],
    strategies: list[str]
):
    """
    Executa a inferência, grava logs estruturados e persiste os resultados no CSV.
    """
    results_csv_path = MODELS_RESULTS_DIR / sanitize_model_name(model) / "metrics_output.csv"

    # Inicializa o CSV garantindo a presença do cabeçalho
    init_csv_file(results_csv_path)

    for strategy in strategies:
        final_prompt = text_prompt
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
            "raw_output": "",
            "error": "",
        }

        try:
            raw_output = client.run(
                model=model,
                prompt=final_prompt,
                images=images_paths,
            )

            duration_ms = int((time.perf_counter() - start_time) * 1000)
            predicted_codes = extract_predicted_wcag(raw_output)
            metrics = calculate_advanced_metrics(ground_truth, predicted_codes)

            # Atualiza o registro com sucesso
            record.update({
                "duration_ms": duration_ms,
                "predictions": "|".join(predicted_codes),
                "raw_output": raw_output,
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
