import pandas as pd
import ast
import re
from pathlib import Path
from typing import Set, List, Tuple

from config import logger, LOCAL_SCREENSHOTS_DIR


def parse_supplementary_info(supp_info: str) -> Tuple[List[str], str]:
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


def extract_predicted_wcag(llm_output: str) -> Set[str]:
    """
    Extrai códigos numéricos WCAG da resposta livre (texto natural) do LLM.
    Não utiliza ast.literal_eval para garantir tolerância a alucinações de formato.
    """
    try:
        if not llm_output:
            return set()
            
        # Busca direta em qualquer parte do texto retornando o padrão X.Y.Z
        pattern = r"\b[1-4]\.\d+\.\d+\b"
        matches = re.findall(pattern, str(llm_output))
        return set(matches)
        
    except Exception as e:
        logger.warning("failed_to_extract_prediction", output=str(llm_output), error=str(e))
        return set()

def extract_wcag_codes(wcag_string: str) -> Set[str]:
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
