import csv
import os
import re
import time
from pathlib import Path

import pandas as pd

from config import logger
from llm_clients import LLMClient
from procecss import parse_llm_output

CSV_HEADERS = [
    "item_id", "model", "strategy", "duration_ms",
    "tp", "fp", "fn", "precision", "recall", "f1_score",
    "ground_truth", "predictions", "raw_output", "error",
    "schema_valid", "parse_failed", "empty_response", "prediction_source", "mapped_rule_ids"
]
RESULTS_ROOT_DIR = Path("./experiment_results")
RUN_ID = time.strftime("%Y%m%d_%H%M%S")
RUN_RESULTS_DIR = RESULTS_ROOT_DIR / "runs" / RUN_ID
MODELS_RESULTS_DIR = RUN_RESULTS_DIR / "by_model"
COMPARISON_RESULTS_PATH = RUN_RESULTS_DIR / "model_comparison.csv"


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        import tiktoken  # type: ignore

        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text))
    except Exception:
        # fallback aproximado
        return max(1, round(len(text) / 4))


def _get_max_input_tokens() -> int | None:
    raw = (os.getenv("MAX_INPUT_TOKENS", "") or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
        return value if value > 0 else None
    except ValueError:
        logger.warning("invalid_max_input_tokens_env", value=raw)
        return None

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

def append_to_csv(filepath: Path, record: dict[str, str | int | float | bool]):
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

    max_input_tokens = _get_max_input_tokens()

    for strategy in strategies:
        final_prompt = text_prompt
        start_time = time.perf_counter()

        logger.info("inference_started", item_id=item_id, model=model, strategy=strategy)
        system_prompt = str(getattr(client, "system_prompt", "") or "")
        user_tokens = _estimate_tokens(final_prompt)
        system_tokens = _estimate_tokens(system_prompt)
        total_input_tokens = user_tokens + system_tokens

        logger.info(
            "inference_payload_summary",
            item_id=item_id,
            model=model,
            strategy=strategy,
            prompt_chars=len(final_prompt),
            prompt_truncated_marker_present=("[TRUNCATED]" in final_prompt),
            has_dom_section=("DOM (HTML):" in final_prompt),
            has_axtree_section=("ACCESSIBILITY TREE (JSON):" in final_prompt),
            has_schema_hint=("Return a strict JSON object" in final_prompt),
            image_count=len(images_paths),
            image_paths=images_paths,
            estimated_user_tokens=user_tokens,
            estimated_system_tokens=system_tokens,
            estimated_total_input_tokens=total_input_tokens,
            max_input_tokens=max_input_tokens,
        )

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
            "schema_valid": False,
            "parse_failed": False,
            "empty_response": False,
            "prediction_source": "none",
            "mapped_rule_ids": "",
        }

        try:
            if max_input_tokens is not None and total_input_tokens > max_input_tokens:
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                record.update({
                    "duration_ms": duration_ms,
                    "error": f"max_input_tokens_exceeded:{total_input_tokens}>{max_input_tokens}",
                    "prediction_source": "skipped_input_too_large",
                })
                logger.warning(
                    "inference_skipped_input_token_limit",
                    item_id=item_id,
                    model=model,
                    strategy=strategy,
                    estimated_total_input_tokens=total_input_tokens,
                    max_input_tokens=max_input_tokens,
                )
                continue

            raw_output = client.run(
                model=model,
                prompt=final_prompt,
                images=images_paths,
            )

            duration_ms = int((time.perf_counter() - start_time) * 1000)
            parsed_output = parse_llm_output(raw_output)
            predicted_codes = set(parsed_output["predicted_codes"])
            metrics = calculate_advanced_metrics(ground_truth, predicted_codes)

            empty_response = bool(parsed_output["empty_response"])
            parse_failed = bool(parsed_output["parse_failed"])
            schema_valid = bool(parsed_output["schema_valid"])
            prediction_source = str(parsed_output["prediction_source"])
            mapped_rule_ids = "|".join(parsed_output["mapped_rule_ids"])

            error_msg = ""
            if empty_response:
                error_msg = "empty_response"
            elif not predicted_codes:
                error_msg = "no_wcag_codes_extracted"

            # Atualiza o registro com sucesso
            record.update({
                "duration_ms": duration_ms,
                "predictions": "|".join(sorted(predicted_codes)),
                "raw_output": raw_output,
                "error": error_msg,
                "schema_valid": schema_valid,
                "parse_failed": parse_failed,
                "empty_response": empty_response,
                "prediction_source": prediction_source,
                "mapped_rule_ids": mapped_rule_ids,
                **metrics,
            })

            logger.info(
                "inference_success",
                item_id=item_id,
                model=model,
                strategy=strategy,
                metrics=metrics,
                prediction_source=prediction_source,
                schema_valid=schema_valid,
                parse_failed=parse_failed,
                empty_response=empty_response,
                mapped_rule_ids=mapped_rule_ids,
            )

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


def log_results_quality_summary(results_csv_path: Path, model: str):
    """
    Gera um resumo de qualidade do parsing/predição para auditoria do run.
    """
    try:
        if not results_csv_path.exists():
            logger.warning("quality_summary_missing_results", path=str(results_csv_path), model=model)
            return

        df = pd.read_csv(results_csv_path)
        total = len(df)
        if total == 0:
            logger.warning("quality_summary_empty_results", path=str(results_csv_path), model=model)
            return

        empty_predictions = int(df["predictions"].fillna("").astype(str).str.strip().eq("").sum())

        parse_failed_series = (
            df["parse_failed"]
            if "parse_failed" in df.columns
            else pd.Series([False] * total)
        )
        schema_valid_series = (
            df["schema_valid"]
            if "schema_valid" in df.columns
            else pd.Series([False] * total)
        )
        empty_response_series = (
            df["empty_response"]
            if "empty_response" in df.columns
            else pd.Series([False] * total)
        )

        parse_failed = len(df[parse_failed_series.fillna(False).astype(bool)])
        schema_invalid = len(df[~schema_valid_series.fillna(False).astype(bool)])
        empty_response = len(df[empty_response_series.fillna(False).astype(bool)])

        logger.info(
            "results_quality_summary",
            model=model,
            path=str(results_csv_path),
            total_rows=total,
            empty_predictions=empty_predictions,
            empty_predictions_pct=round((empty_predictions / total) * 100, 2),
            parse_failed=parse_failed,
            parse_failed_pct=round((parse_failed / total) * 100, 2),
            schema_invalid=schema_invalid,
            schema_invalid_pct=round((schema_invalid / total) * 100, 2),
            empty_response=empty_response,
            empty_response_pct=round((empty_response / total) * 100, 2),
        )

    except Exception as e:
        logger.warning("quality_summary_failed", path=str(results_csv_path), model=model, error=str(e))


def export_diagnostics_report(results_csv_path: Path, output_csv_path: Path, model: str):
    """
    Exporta um diagnóstico por linha para facilitar análise de erros e parsing.
    """
    try:
        if not results_csv_path.exists():
            logger.warning("diagnostics_export_missing_results", model=model, path=str(results_csv_path))
            return

        df = pd.read_csv(results_csv_path)
        if df.empty:
            logger.warning("diagnostics_export_empty_results", model=model, path=str(results_csv_path))
            return

        # Garantir colunas esperadas mesmo em CSVs legados
        for col, default in {
            "schema_valid": False,
            "parse_failed": False,
            "empty_response": False,
            "prediction_source": "unknown",
            "mapped_rule_ids": "",
            "error": "",
            "predictions": "",
            "raw_output": "",
        }.items():
            if col not in df.columns:
                df[col] = default

        has_prediction = ~df["predictions"].fillna("").astype(str).str.strip().eq("")
        schema_valid = df["schema_valid"].fillna(False).astype(bool)
        parse_failed = df["parse_failed"].fillna(False).astype(bool)
        empty_response = df["empty_response"].fillna(False).astype(bool)
        error_text = df["error"].fillna("").astype(str).str.strip()

        error_class = pd.Series(["ok"] * len(df), index=df.index)
        error_class.loc[~has_prediction] = "no_prediction"
        error_class.loc[~schema_valid] = "schema_invalid"
        error_class.loc[parse_failed] = "parse_failed"
        error_class.loc[empty_response] = "empty_response"
        error_class.loc[error_text.ne("")] = error_text[error_text.ne("")]

        diagnostics_df = pd.DataFrame({
            "item_id": df.get("item_id", ""),
            "model": df.get("model", ""),
            "strategy": df.get("strategy", ""),
            "duration_ms": df.get("duration_ms", 0),
            "tp": df.get("tp", 0),
            "fp": df.get("fp", 0),
            "fn": df.get("fn", 0),
            "ground_truth": df.get("ground_truth", ""),
            "predictions": df.get("predictions", ""),
            "prediction_source": df.get("prediction_source", "unknown"),
            "mapped_rule_ids": df.get("mapped_rule_ids", ""),
            "schema_valid": schema_valid,
            "parse_failed": parse_failed,
            "empty_response": empty_response,
            "error": error_text,
            "error_class": error_class,
            "has_prediction": has_prediction,
            "raw_output_excerpt": df["raw_output"].fillna("").astype(str).str.slice(0, 600),
        })

        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_df.to_csv(output_csv_path, index=False)

        logger.info(
            "diagnostics_report_written",
            model=model,
            path=str(output_csv_path),
            rows=len(diagnostics_df),
        )

    except Exception as e:
        logger.warning(
            "diagnostics_report_failed",
            model=model,
            input_path=str(results_csv_path),
            output_path=str(output_csv_path),
            error=str(e),
        )
