# accessibility-validator-llm

## How to run

### 1) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

### 2) Run the experiment

Default (Ollama):

```bash
python3 experiment_v2.py
```

Using LM Studio:

```bash
LLM_PROVIDER=lmstudio python3 experiment_v2.py
```

Using a custom dataset CSV (for calibration set, for example):

```bash
DATASET_CSV_PATH=content/calibration/calibration_set_50.csv python3 experiment_v2.py
```

Limit max input tokens per request (skip oversized prompts):

```bash
MAX_INPUT_TOKENS=12000 python3 experiment_v2.py
```

You can combine env vars:

```bash
LLM_PROVIDER=ollama DATASET_CSV_PATH=content/calibration/calibration_set_50.csv MAX_INPUT_TOKENS=12000 python3 experiment_v2.py
```

### 3) Generate calibration set (optional)

```bash
python3 create_calibration_set.py \
  --source-csv Original_full_data_new.csv \
  --output-csv content/calibration/calibration_set_50.csv \
  --pages 50 \
  --seed 42
```

### 4) Generate token histogram

```bash
python3 calculate_token_histogram.py --want-screenshot
```

Outputs are written to `experiment_results/token_analysis/`:
- `page_token_counts.csv`
- `token_histogram.csv`
- `token_histogram.png`
- `token_histogram_summary.txt`
