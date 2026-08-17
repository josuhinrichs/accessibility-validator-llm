from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def create_calibration_set(
    source_csv: Path,
    output_csv: Path,
    sample_pages: int = 50,
    random_seed: int = 42,
):
    if not source_csv.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_csv}")

    df = pd.read_csv(source_csv)
    if "html_file_path" not in df.columns:
        raise ValueError("Expected column 'html_file_path' in source CSV")

    unique_pages = (
        df["html_file_path"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if unique_pages.empty:
        raise ValueError("No pages found in 'html_file_path'")

    n = min(sample_pages, len(unique_pages))
    sampled_pages = unique_pages.sample(n=n, random_state=random_seed)

    calibration_df = df[df["html_file_path"].astype(str).isin(set(sampled_pages))].copy()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    calibration_df.to_csv(output_csv, index=False)

    info_path = output_csv.with_suffix(".meta.txt")
    info_path.write_text(
        "\n".join(
            [
                f"source_csv={source_csv}",
                f"output_csv={output_csv}",
                f"sample_pages={n}",
                f"random_seed={random_seed}",
                f"rows={len(calibration_df)}",
                f"unique_pages={calibration_df['html_file_path'].astype(str).nunique()}",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Calibration set written: {output_csv}")
    print(f"Rows: {len(calibration_df)}")
    print(f"Unique pages: {calibration_df['html_file_path'].astype(str).nunique()}")
    print(f"Meta: {info_path}")


def main():
    parser = argparse.ArgumentParser(description="Create calibration subset from full dataset")
    parser.add_argument(
        "--source-csv",
        default="Original_full_data_new.csv",
        help="Path to full dataset CSV",
    )
    parser.add_argument(
        "--output-csv",
        default="content/calibration/calibration_set_50.csv",
        help="Path to output calibration CSV",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=50,
        help="Number of unique pages to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()
    create_calibration_set(
        source_csv=Path(args.source_csv),
        output_csv=Path(args.output_csv),
        sample_pages=args.pages,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
