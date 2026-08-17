import ast
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from config import logger, LOCAL_SCREENSHOTS_DIR


def parse_supplementary_info(supp_info: str) -> tuple[list[str], str]:
    """
    Analisa a coluna supplementary_information.
    Retorna uma tupla: (Lista de caminhos de imagens válidos, Texto suplementar).
    """
    images = []
    text_context = ""
    
    # Tratamento de valores nulos do Pandas (NaN)
    if pd.isna(supp_info) or not str(supp_info).strip():
        return images, text_context

    supp_str = str(supp_info)

    # Identifica se o campo contém caminhos de arquivos de imagem
    if ".png" in supp_str or ".jpg" in supp_str:
        raw_paths = [p.strip() for p in supp_str.split(",")]
        for rp in raw_paths:
            if rp.endswith((".png", ".jpg")):
                # Extrai apenas o nome do arquivo do path absoluto original do CSV
                filename = Path(rp).name 
                local_path = LOCAL_SCREENSHOTS_DIR / filename
                
                if local_path.exists():
                    images.append(str(local_path))
                else:
                    logger.warning("missing_image_file", expected_path=str(local_path))
    else:
        # Se não há extensão de imagem, assumimos que é contexto textual (ex: link-name)
        text_context = supp_str.strip()

    return images, text_context


WCAG_CODE_PATTERN = re.compile(r"\b[1-4]\.\d+\.\d+\b")

# Mapeamento pragmático para rule IDs recorrentes no dataset.
# Observação: pode (e deve) ser expandido conforme surgirem novos rule IDs.
RULE_ID_TO_WCAG: dict[str, set[str]] = {
    "047fe0": {"2.4.6"},
    "5b7ae0": {"3.1.1"},
    "5c01ea": {"4.1.2"},
    "674b10": {"4.1.2", "1.3.1"},
    "80f0bf": {"1.4.2"},
    "8fc3b6": {"1.1.1", "4.1.2"},
    "akn7bn": {"2.1.1"},
    "eac66b": {"1.2.1", "1.2.2"},
    "area-alt": {"1.1.1"},
    "aria-braille-equivalent": {"4.1.2"},
    "aria-meter-name": {"4.1.2"},
    "aria-tooltip-name": {"4.1.2"},
    "aria-treeitem-name": {"4.1.2"},
    "empty-table-header": {"1.3.1"},
    "form-field-multiple-labels": {"1.3.1", "3.3.2", "4.1.2"},
    "target-size": {"2.5.8", "2.5.5"},
}


def _strip_markdown_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _collect_wcag_codes_from_text(text: str) -> set[str]:
    return set(WCAG_CODE_PATTERN.findall(text or ""))


def _collect_rule_ids(node: Any) -> set[str]:
    rule_ids: set[str] = set()

    if isinstance(node, dict):
        for key, value in node.items():
            key_l = str(key).lower()
            if key_l in {"rule_id", "ruleid", "rule", "id"} and isinstance(value, str):
                rule_ids.add(value.strip())
            rule_ids.update(_collect_rule_ids(value))
    elif isinstance(node, list):
        for item in node:
            rule_ids.update(_collect_rule_ids(item))

    return rule_ids


def _collect_all_strings(node: Any) -> list[str]:
    strings: list[str] = []
    if isinstance(node, dict):
        for value in node.values():
            strings.extend(_collect_all_strings(value))
    elif isinstance(node, list):
        for item in node:
            strings.extend(_collect_all_strings(item))
    elif isinstance(node, str):
        strings.append(node)
    return strings


def parse_llm_output(llm_output: str) -> dict[str, Any]:
    """
    Parse robusto da saída do LLM:
    - tenta JSON primeiro,
    - extrai códigos WCAG por campos estruturados + regex textual,
    - normaliza rule_ids para WCAG via mapeamento.
    """
    result: dict[str, Any] = {
        "predicted_codes": set(),
        "schema_valid": False,
        "parse_failed": False,
        "empty_response": False,
        "prediction_source": "none",
        "mapped_rule_ids": [],
    }

    if not llm_output or not str(llm_output).strip():
        result["empty_response"] = True
        result["parse_failed"] = True
        return result

    raw_text = str(llm_output)
    cleaned = _strip_markdown_code_fences(raw_text)

    parsed_json: Any = None
    json_ok = False
    try:
        parsed_json = json.loads(cleaned)
        json_ok = True
    except Exception:
        result["parse_failed"] = True

    predicted_codes: set[str] = set()
    mapped_rule_ids: set[str] = set()

    if json_ok:
        strings = _collect_all_strings(parsed_json)
        for s in strings:
            predicted_codes.update(_collect_wcag_codes_from_text(s))

        rule_ids = _collect_rule_ids(parsed_json)
        for rule_id in rule_ids:
            norm_rule_id = rule_id.strip().lower()
            mapped = RULE_ID_TO_WCAG.get(norm_rule_id)
            if mapped:
                predicted_codes.update(mapped)
                mapped_rule_ids.add(norm_rule_id)

        # Critério de schema "válido": objeto com chave 'violations' em formato lista.
        result["schema_valid"] = (
            isinstance(parsed_json, dict)
            and isinstance(parsed_json.get("violations"), list)
        )

        if predicted_codes:
            result["prediction_source"] = "json"
        elif mapped_rule_ids:
            result["prediction_source"] = "rule_id_map"
        else:
            result["prediction_source"] = "json_no_codes"
    else:
        # Fallback texto livre
        predicted_codes.update(_collect_wcag_codes_from_text(raw_text))
        result["prediction_source"] = "regex_fallback" if predicted_codes else "none"

    result["predicted_codes"] = predicted_codes
    result["mapped_rule_ids"] = sorted(mapped_rule_ids)
    return result


def extract_predicted_wcag(llm_output: str) -> set[str]:
    """
    Compat wrapper para chamadas legadas.
    """
    parsed = parse_llm_output(llm_output)
    return set(parsed["predicted_codes"])

def extract_wcag_codes(wcag_string: str) -> set[str]:
    """Extrai os códigos WCAG (ex: 1.1.1) da string bruta."""
    try:
        wcag_list = ast.literal_eval(wcag_string)
        codes = set()
        for item in wcag_list:
            match = re.search(r"\b[1-4]\.\d+\.\d+\b", item)
            if match:
                codes.add(match.group())
        return codes
    except Exception as e:
        logger.warning("failed_to_parse_wcag", wcag_string=wcag_string, error=str(e))
        return set()
    
def calculate_metrics(
    results_csv_path: str | Path = "./experiment_results/metrics_output.csv",
    output_csv_path: str | Path = "./experiment_results/final_metrics.csv",
):
    df = pd.read_csv(results_csv_path)

    # Agrupamento e soma dos valores brutos
    global_metrics = df.groupby(['model', 'strategy'])[['tp', 'fp', 'fn']].sum().reset_index()

    global_metrics['precision'] = global_metrics['tp'] / (global_metrics['tp'] + global_metrics['fp']).replace(0, pd.NA)
    global_metrics['recall'] = global_metrics['tp'] / (global_metrics['tp'] + global_metrics['fn']).replace(0, pd.NA)
    global_metrics['f1_score'] = (2 * global_metrics['precision'] * global_metrics['recall']) / (global_metrics['precision'] + global_metrics['recall']).replace(0, pd.NA)
    global_metrics = global_metrics.fillna(0.0)

    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    global_metrics.to_csv(output_csv_path, index=False)
