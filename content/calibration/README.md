# Calibration set

This folder contains a reproducible small calibration dataset derived from `Original_full_data_new.csv`.

## Files

- `calibration_set_50.csv`: sampled subset containing all rows for 50 unique pages.
- `calibration_set_50.meta.txt`: generation metadata (source, seed, row/page counts).

## Re-generate

```bash
python3 create_calibration_set.py \
  --source-csv Original_full_data_new.csv \
  --output-csv content/calibration/calibration_set_50.csv \
  --pages 50 \
  --seed 42
```

## Run experiment with calibration set

```bash
DATASET_CSV_PATH=content/calibration/calibration_set_50.csv python3 experiment_v2.py
```

You can combine with provider selection:

```bash
LLM_PROVIDER=ollama DATASET_CSV_PATH=content/calibration/calibration_set_50.csv python3 experiment_v2.py
```

```bash
LLM_PROVIDER=lmstudio DATASET_CSV_PATH=content/calibration/calibration_set_50.csv python3 experiment_v2.py
```
