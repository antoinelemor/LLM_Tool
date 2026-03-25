"""
Create an extra-extra-large dataset from the complete.csv source file.

Requirements:
- All rows from source except those in benchmark_annotations_by_annotator.csv
- Check id, date AND text to identify duplicates (same id can have different text)
"""

import pandas as pd
from pathlib import Path


def create_extra_extra_large_dataset(
    source_path: str,
    benchmark_path: str,
    output_path: str,
) -> pd.DataFrame:
    """
    Create a dataset with all source rows excluding benchmark sentences.

    Args:
        source_path: Path to the complete.csv source file
        benchmark_path: Path to benchmark_annotations_by_annotator.csv
        output_path: Path to save the output dataset

    Returns:
        The created DataFrame
    """
    # Load source data
    print(f"Loading source data from {source_path}...")
    source_df = pd.read_csv(source_path)
    print(f"  Source rows: {len(source_df)}")

    # Load benchmark data
    print(f"Loading benchmark data from {benchmark_path}...")
    benchmark_df = pd.read_csv(benchmark_path)
    print(f"  Benchmark rows: {len(benchmark_df)}")

    # Create a unique key from id, date and text for exclusion
    # This handles cases where same id has different text
    source_df["_key"] = (
        source_df["id"].astype(str)
        + "|"
        + source_df["date"].astype(str)
        + "|"
        + source_df["text"].astype(str)
    )
    benchmark_keys = set(
        benchmark_df["id"].astype(str)
        + "|"
        + benchmark_df["date"].astype(str)
        + "|"
        + benchmark_df["text"].astype(str)
    )

    # Filter out benchmark sentences
    print(f"Filtering out benchmark sentences...")
    result_df = source_df[~source_df["_key"].isin(benchmark_keys)].copy()
    result_df.drop(columns=["_key"], inplace=True)

    # Remove the 'n' column if present (it's just a row number from source)
    if "n" in result_df.columns:
        result_df = result_df.drop(columns=["n"])

    # Reset index
    result_df = result_df.reset_index(drop=True)

    print(f"\nFinal dataset size: {len(result_df)}")
    print(f"\nDistribution:")
    print(f"  Source: {result_df['source'].value_counts().to_dict()}")
    print(f"  Lang: {result_df['lang'].value_counts().to_dict()}")

    # Create output directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    result_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    return result_df


if __name__ == "__main__":
    # Paths
    project_root = Path(__file__).parent.parent.parent
    source_path = "/Users/antoine/Documents/GitHub/vitrine_pipeline/data/raw/subset/complete.csv"
    benchmark_path = project_root / "data" / "sets" / "benchmark_annotations_by_annotator.csv"
    output_path = project_root / "data" / "sets" / "extra-extra-large.csv"

    create_extra_extra_large_dataset(
        source_path=source_path,
        benchmark_path=str(benchmark_path),
        output_path=str(output_path),
    )
