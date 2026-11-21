# Name: Nizar Bahadur | Roll Number: DS-GRAD | Date: 2026-11-20
"""Pipeline 1: CSV verification, non-CSV format conversion, and raw ingestion."""

from pathlib import Path
import shutil
import pandas as pd


SUPPORTED_CONVERT_EXTS = {".xlsx", ".xls", ".json", ".parquet", ".txt", ".tsv"}


def convert_to_csv(source_path: str | Path, output_csv_path: str | Path) -> Path:
    """Convert supported tabular formats (Excel, JSON, Parquet, TXT/TSV) into a .csv file."""
    src = Path(source_path)
    out = Path(output_csv_path)
    ext = src.suffix.lower()

    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(src)
    elif ext == ".json":
        df = pd.read_json(src)
    elif ext == ".parquet":
        df = pd.read_parquet(src)
    elif ext in {".txt", ".tsv"}:
        df = pd.read_csv(src, sep=None, engine="python")
    else:
        raise ValueError(f"Unsupported format '{ext}' for CSV conversion.")

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def ingest_raw_csv(source_path: str | Path, raw_dir: str | Path = "data/raw") -> Path:
    """Verify .csv extension and copy an untouched raw dataset into data/raw/."""
    src = Path(source_path)
    if not src.exists():
        raise FileNotFoundError(f"Dataset file not found: {src}")
    if not src.is_file():
        raise ValueError(f"Expected a file path, got directory: {src}")
    if src.suffix.lower() != ".csv":
        raise ValueError(f"Expected '.csv' file, received '{src.suffix}'.")

    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    dest = raw_path / src.name
    shutil.copy2(src, dest)
    return dest